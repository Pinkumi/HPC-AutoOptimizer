"""Offline child-process backends. Auto mode never falls back to host execution."""
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
from time import perf_counter

from .evaluation_spec import ExecutionLimits


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    data: dict
    error: str | None = None
    stderr: str = ""
    process_wall_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _kill(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _capture(argv: list[str], cwd: Path, timeout: float, output_limit: int) -> ExecutionResult:
    start = perf_counter()
    stdout, stderr = bytearray(), bytearray()
    status, error = "ok", None
    env = {"PATH": "/usr/bin:/bin", "HOME": str(cwd), "TMPDIR": str(cwd), "LANG": "C.UTF-8"}
    try:
        process = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    except OSError as exc:
        return ExecutionResult("execution_error", {}, str(exc))
    try:
        with selectors.DefaultSelector() as selector:
            for pipe, buffer in ((process.stdout, stdout), (process.stderr, stderr)):
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ, buffer)
            while selector.get_map():
                remaining = timeout - (perf_counter() - start)
                if remaining <= 0:
                    status, error = "timeout", "Worker exceeded its wall-clock timeout"
                    break
                for key, _ in selector.select(min(0.05, remaining)):
                    chunk = os.read(key.fileobj.fileno(), 8192)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    available = output_limit - len(stdout) - len(stderr)
                    key.data.extend(chunk[:max(0, available)])
                    if len(chunk) > available:
                        status, error = "execution_error", "Worker output exceeded its byte limit"
                        break
                if error:
                    break
        if status == "ok":
            try:
                process.wait(timeout=max(0.001, timeout - (perf_counter() - start)))
            except subprocess.TimeoutExpired:
                status, error = "timeout", "Worker exceeded its wall-clock timeout"
        if status == "ok" and process.returncode != 0:
            status, error = "execution_error", f"Worker exited with status {process.returncode} (crash or resource limit)"
    finally:
        _kill(process)  # Also terminate descendants that retained the process group.
        process.wait()
        process.stdout.close()
        process.stderr.close()
    data = {}
    if status == "ok":
        try:
            data = json.loads(stdout.decode("utf-8"))
            if not isinstance(data, dict) or data.get("status") not in ("ok", "error", "unsupported"):
                raise ValueError("Invalid response envelope")
            if data["status"] == "error":
                status, error = "execution_error", str(data.get("error", "Worker error"))[:1000]
            elif data["status"] == "unsupported":
                status, error = "unsupported", str(data.get("error", "Unsupported output"))[:1000]
        except (ValueError, UnicodeError, RecursionError) as exc:
            status, error = "execution_error", f"Invalid worker response: {str(exc)[:200]}"
    return ExecutionResult(status, data if status == "ok" else {}, error,
                           stderr.decode("utf-8", errors="replace")[:2000], perf_counter() - start)


class ExecutionBackend:
    """Bubblewrap namespace isolation or explicitly trusted development fixtures."""
    def __init__(self, name: str = "auto", *, trusted_fixtures: bool = False):
        if name not in ("auto", "bubblewrap", "trusted"):
            raise ValueError("backend must be auto, bubblewrap, or trusted")
        if name == "trusted" and not trusted_fixtures:
            raise ValueError("Trusted mode requires explicit --trusted-fixtures opt-in; it is not a sandbox")
        if trusted_fixtures and name != "trusted":
            raise ValueError("--trusted-fixtures requires --backend trusted")
        self.name = "bubblewrap" if name == "auto" else name
        self.isolation_mode = "os_namespaces" if self.name == "bubblewrap" else "trusted_development_only"
        self.executable = shutil.which("bwrap") if self.name == "bubblewrap" else sys.executable
        self.python = Path(getattr(sys, "_base_executable", sys.executable)).resolve()

    def _sandbox_command(self, workspace: Path, temporary_bytes: int = 16777216) -> list[str]:
        command = [self.executable, "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL",
                   "--unshare-user", "--unshare-pid", "--unshare-net", "--disable-userns",
                   "--clearenv", "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "HOME", "/tmp",
                   "--setenv", "TMPDIR", "/tmp", "--ro-bind", "/usr", "/usr"]
        for directory in ("/bin", "/lib", "/lib64"):
            if Path(directory).is_symlink():
                command += ["--symlink", os.readlink(directory), directory]
            elif Path(directory).exists():
                command += ["--ro-bind", directory, directory]
        command += ["--proc", "/proc", "--dev", "/dev", "--size", str(temporary_bytes), "--tmpfs", "/tmp",
                    "--ro-bind", str(workspace), "/work", "--remount-ro", "/dev", "--remount-ro", "/proc",
                    "--remount-ro", "/", "--chdir", "/tmp", "--"]
        return command

    def availability(self) -> tuple[bool, str]:
        if not sys.platform.startswith("linux"):
            return False, "Evaluation currently requires Linux resource limits"
        if self.name == "trusted":
            return True, "Explicit trusted-fixture mode: no filesystem or network isolation"
        if not self.executable or os.geteuid() == 0:
            return False, "Rootless bubblewrap is unavailable; no host-execution fallback is permitted"
        if not self.python.is_relative_to("/usr"):
            return False, "Bubblewrap backend requires a Python runtime under /usr"
        with tempfile.TemporaryDirectory(prefix="hpc-probe-") as temporary:
            root = Path(temporary)
            # A trusted probe verifies namespace creation, rather than checking the binary alone.
            command = self._sandbox_command(root) + [str(self.python), "-I", "-S", "-c",
                                                    "print(" + repr(json.dumps({"status": "ok"})) + ")"]
            result = _capture(command, root, 3.0, 4096)
        return (True, "Bubblewrap namespaces available") if result.status == "ok" else (
            False, "Bubblewrap probe failed: " + (result.stderr or result.error or "unknown error")[:500])

    def run(self, source: str, function: str, cases: list[dict], limits: ExecutionLimits, *,
            mode: str = "correctness", warmup: int = 0, repetitions: int = 1) -> ExecutionResult:
        with tempfile.TemporaryDirectory(prefix="hpc-evaluate-") as temporary:
            root = Path(temporary)
            (root / "program.py").write_text(source, encoding="utf-8")
            shutil.copyfile(Path(__file__).with_name("_evaluation_worker.py"), root / "worker.py")
            request = {"function": function, "cases": cases, "mode": mode, "warmup": warmup,
                       "repetitions": repetitions, "limits": asdict(limits)}
            (root / "request.json").write_text(json.dumps(request, allow_nan=False), encoding="utf-8")
            if self.name == "bubblewrap":
                command = self._sandbox_command(root, min(limits.memory_bytes, 16777216)) + [
                    str(self.python), "-I", "-S", "-B", "/work/worker.py"]
            else:
                command = [sys.executable, "-I", "-S", "-B", str(root / "worker.py")]
            return _capture(command, root, limits.timeout_seconds, limits.output_bytes)
