"""Explicit, bounded JSON contracts for CPU evaluation; no inferred test cases."""
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path


def _keys(data: dict, allowed: set[str], label: str) -> None:
    if not isinstance(data, dict) or set(data) - allowed:
        raise ValueError(f"Invalid or unknown {label} fields")


def _positive(value: object, label: str, *, integer: bool = False, zero: bool = False) -> None:
    try:
        valid = (type(value) is int if integer else type(value) in (int, float)) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid or (value < 0 if zero else value <= 0):
        raise ValueError(f"{label} must be a finite {'nonnegative' if zero else 'positive'} {'integer' if integer else 'number'}")


@dataclass(frozen=True)
class ComparisonSpec:
    mode: str = "exact"
    rtol: float = 1e-8
    atol: float = 1e-10
    equal_nan: bool = False
    allow_int_float: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ("exact", "numeric"):
            raise ValueError("comparison.mode must be exact or numeric")
        _positive(self.rtol, "rtol", zero=True)
        _positive(self.atol, "atol", zero=True)
        if type(self.equal_nan) is not bool or type(self.allow_int_float) is not bool:
            raise ValueError("equal_nan and allow_int_float must be booleans")


@dataclass(frozen=True)
class ExecutionLimits:
    timeout_seconds: float = 5.0
    memory_bytes: int = 268435456
    cpu_seconds: int = 3
    process_limit: int = 1
    output_bytes: int = 65536
    file_bytes: int = 1048576

    def __post_init__(self) -> None:
        for key, value in asdict(self).items():
            _positive(value, key, integer=key != "timeout_seconds")


@dataclass(frozen=True)
class BenchmarkSpec:
    cases: list[dict]
    warmup: int = 3
    repetitions: int = 10
    improvement_threshold: float = 0.05
    instability_threshold: float = 0.10

    def __post_init__(self) -> None:
        _cases(self.cases, "benchmark.cases")
        _positive(self.warmup, "warmup", integer=True, zero=True)
        _positive(self.repetitions, "repetitions", integer=True)
        _positive(self.improvement_threshold, "improvement_threshold", zero=True)
        _positive(self.instability_threshold, "instability_threshold", zero=True)
        if self.improvement_threshold >= 1:
            raise ValueError("improvement_threshold must be below 1")


def _cases(cases: object, label: str) -> None:
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{label} must be a nonempty array")
    for case in cases:
        _keys(case, {"args", "kwargs"}, label)
        if not isinstance(case.get("args", []), list) or not isinstance(case.get("kwargs", {}), dict):
            raise ValueError(f"{label} requires args array and kwargs object")
        if any(not isinstance(key, str) for key in case.get("kwargs", {})):
            raise ValueError("kwargs keys must be strings")
    # Reject non-JSON inputs and NaN/Infinity; arrays and GPU objects are future extensions.
    json.dumps(cases, allow_nan=False)


@dataclass(frozen=True)
class EvaluationSpec:
    function: str
    correctness_cases: list[dict]
    benchmark: BenchmarkSpec
    comparison: ComparisonSpec = field(default_factory=ComparisonSpec)
    limits: ExecutionLimits = field(default_factory=ExecutionLimits)

    def __post_init__(self) -> None:
        if not isinstance(self.function, str) or not self.function.isidentifier():
            raise ValueError("function must be a Python identifier")
        _cases(self.correctness_cases, "correctness_cases")

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationSpec":
        _keys(data, {"function", "correctness_cases", "comparison", "benchmark", "limits"}, "specification")
        if not {"function", "correctness_cases", "benchmark"} <= data.keys():
            raise ValueError("Specification requires function, correctness_cases, and benchmark")
        comparison, limits, benchmark = data.get("comparison", {}), data.get("limits", {}), data["benchmark"]
        _keys(comparison, set(ComparisonSpec.__dataclass_fields__), "comparison")
        _keys(limits, set(ExecutionLimits.__dataclass_fields__), "limits")
        _keys(benchmark, set(BenchmarkSpec.__dataclass_fields__), "benchmark")
        if "cases" not in benchmark:
            raise ValueError("benchmark.cases must explicitly specify timing inputs")
        return cls(data["function"], data["correctness_cases"], BenchmarkSpec(**benchmark),
                   ComparisonSpec(**comparison), ExecutionLimits(**limits))

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationSpec":
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate specification field: {key}")
                result[key] = value
            return result
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique))

    def to_dict(self) -> dict:
        return asdict(self)
