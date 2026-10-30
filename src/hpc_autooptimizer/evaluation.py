"""Explicit local evaluation and compatibility contracts from Phase 1."""
from __future__ import annotations
import ast
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
from typing import Protocol, TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from .evaluation_spec import EvaluationSpec
    from .execution import ExecutionBackend
    from .history import ExperimentHistory


@dataclass(frozen=True)
class IsolationPolicy:
    wall_timeout_seconds: float
    memory_bytes: int
    cpu_seconds: int
    process_limit: int
    output_bytes: int
    writable_directory: Path
    network_enabled: bool = False

    def __post_init__(self):
        for name in ("wall_timeout_seconds", "memory_bytes", "cpu_seconds", "process_limit", "output_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < float("inf"):
                raise ValueError(f"{name} must be finite and positive")
        if self.network_enabled:
            raise ValueError("Evaluation networking must remain disabled")


@dataclass(frozen=True)
class CorrectnessResult:
    status: str
    details: str


class CorrectnessEvaluator(Protocol):
    def evaluate(self, original: Path, candidate: Path, policy: IsolationPolicy) -> CorrectnessResult:
        """Require enforced isolation before reading or running untrusted artifacts."""


class DisabledCorrectnessEvaluator:
    def evaluate(self, original: Path, candidate: Path, policy: IsolationPolicy) -> CorrectnessResult:
        raise NotImplementedError("Candidate execution is disabled in Phase 1")


class LocalEvaluator:
    """Explicit offline evaluation. Generation never calls this workflow automatically."""
    def __init__(self, backend: ExecutionBackend, history: ExperimentHistory):
        self.backend = backend
        self.history = history

    def evaluate(self, original: Path, candidate: Path, spec: EvaluationSpec) -> tuple[Path, dict]:
        from .benchmark import CPUFunctionBenchmark
        from .correctness import CaseValidator

        original, candidate = Path(original).resolve(), Path(candidate).resolve()
        sources = {"original": original.read_text(encoding="utf-8"), "candidate": candidate.read_text(encoding="utf-8")}
        record = {"schema_version": 1, "experiment_type": "evaluation", "experiment_id": uuid4().hex,
                  "timestamp": datetime.now(timezone.utc).isoformat(), "original_source_path": str(original),
                  "candidate_source_path": str(candidate), "specification": spec.to_dict(),
                  "execution_backend": self.backend.name, "isolation_mode": self.backend.isolation_mode,
                  "environment": {"python": platform.python_version(), "platform": platform.platform()},
                  "source_hashes": {key: hashlib.sha256(value.encode()).hexdigest() for key, value in sources.items()},
                  "syntax_status": "pending", "correctness": {"status": "not_run", "cases": []},
                  "benchmark": {"status": "not_run", "speedup": None}, "decision": "evaluation_error", "error": None}
        try:
            for label, source in sources.items():
                tree = ast.parse(source)
                functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == spec.function]
                if not functions:
                    raise ValueError(f"{label}: top-level synchronous function {spec.function!r} not found")
            record["syntax_status"] = "syntax_valid"
        except (SyntaxError, ValueError, RecursionError) as exc:
            record.update(syntax_status="invalid_source", correctness={"status": "invalid_source", "cases": []},
                          error=str(exc)[:1000])
        else:
            available, reason = self.backend.availability()
            record["backend_availability"] = reason
            if not available:
                record.update(correctness={"status": "isolation_unavailable", "cases": []}, error=reason)
            else:
                record["correctness"] = CaseValidator(self.backend).validate(sources["original"], sources["candidate"], spec)
                if record["correctness"]["status"] == "passed":
                    record["benchmark"] = CPUFunctionBenchmark(self.backend).measure(
                        sources["original"], sources["candidate"], spec, correctness_status=record["correctness"]["status"])
                    record["decision"] = record["benchmark"].get("decision", "evaluation_error")
                elif record["correctness"]["status"] == "failed":
                    record["decision"] = "rejected_incorrect"
        record["correctness_status"] = record["correctness"]["status"]
        record["benchmark_status"] = record["benchmark"]["status"]
        directory = self.history.record(record)
        return directory, record
