"""Contracts for a future isolated correctness runner; no execution implementation."""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


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
