"""Child-only worker. Never import this module to execute candidate code in the orchestrator."""
import contextlib
import copy
import ctypes
import importlib.util
import json
import math
from pathlib import Path
import resource
import sys
import time


class UnsupportedOutput(TypeError):
    pass


def encode(value, depth=0):
    if depth > 100:
        raise UnsupportedOutput("Output nesting exceeds 100 levels")
    kind = type(value)
    if value is None:
        return {"type": "none", "value": None}
    if kind in (bool, int, str):
        return {"type": kind.__name__, "value": value}
    if kind is float:
        data = "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf") if math.isinf(value) else value
        return {"type": "float", "value": data}
    if kind in (list, tuple):
        return {"type": kind.__name__, "value": [encode(item, depth + 1) for item in value]}
    if kind is dict and all(type(key) is str for key in value):
        return {"type": "dict", "value": {key: encode(item, depth + 1) for key, item in value.items()}}
    raise UnsupportedOutput(f"Unsupported output type: {kind.__name__}")


def main():
    root = Path(__file__).parent
    request = json.loads((root / "request.json").read_text())
    limits = request["limits"]
    if ctypes.CDLL(None, use_errno=True).prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise OSError("Unable to enforce no-new-privileges")
    for key, limit in ((resource.RLIMIT_AS, limits["memory_bytes"]), (resource.RLIMIT_CPU, limits["cpu_seconds"]),
                       (resource.RLIMIT_NPROC, limits["process_limit"]), (resource.RLIMIT_FSIZE, limits["file_bytes"]),
                       (resource.RLIMIT_NOFILE, 64), (resource.RLIMIT_CORE, 0)):
        resource.setrlimit(key, (limit, limit))
    try:
        with contextlib.redirect_stdout(sys.stderr):
            module_spec = importlib.util.spec_from_file_location("evaluated_program", root / "program.py")
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)  # Only inside the explicitly selected child backend.
            function = getattr(module, request["function"])
            if not callable(function):
                raise TypeError("Target is not callable")
            cases = request["cases"]
            if request["mode"] == "correctness":
                case = copy.deepcopy(cases[0])
                output = function(*case.get("args", []), **case.get("kwargs", {}))
                response = {"status": "ok", "output": encode(output)}
            else:
                for _ in range(request["warmup"]):
                    for case in copy.deepcopy(cases):
                        function(*case.get("args", []), **case.get("kwargs", {}))
                samples = []
                for _ in range(request["repetitions"]):
                    inputs = copy.deepcopy(cases)  # Input preparation is outside the measured workload.
                    start = time.perf_counter_ns()
                    for case in inputs:
                        function(*case.get("args", []), **case.get("kwargs", {}))
                    samples.append(time.perf_counter_ns() - start)
                response = {"status": "ok", "samples_ns": samples}
    except UnsupportedOutput as exc:
        response = {"status": "unsupported", "error": str(exc)[:1000]}
    except BaseException as exc:
        response = {"status": "error", "error": f"{type(exc).__name__}: {str(exc)[:1000]}"}
    sys.stdout.write(json.dumps(response, allow_nan=False))


if __name__ == "__main__":
    main()
