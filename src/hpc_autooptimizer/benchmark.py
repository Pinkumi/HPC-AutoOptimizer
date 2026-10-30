"""ParallelBench CLI planning and isolated CPU function timing orchestration."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import statistics
from typing import TYPE_CHECKING
from .evaluation import IsolationPolicy

if TYPE_CHECKING:
    from .execution import ExecutionBackend
    from .evaluation_spec import EvaluationSpec


@dataclass(frozen=True)
class BenchmarkPlan:
    argv: tuple[str, ...]
    results_path: Path
    candidate_supported: bool = False


class ParallelBenchAdapter:
    def plan_baseline(self, output_dir: Path, sizes: tuple[int, ...] = (64, 128),
                      repetitions: int = 20, warmup: int = 3) -> BenchmarkPlan:
        """Prepare a command for built-in CPU kernels, not a generated candidate."""
        if not sizes or any(type(size) is not int or size <= 0 for size in sizes):
            raise ValueError("sizes must be positive integers")
        if any(type(value) is not int or value <= 0 for value in (repetitions, warmup)):
            raise ValueError("repetitions and warmup must be positive integers")
        output_dir = Path(output_dir)
        return BenchmarkPlan(("parallelbench", "--device", "cpu", "--sizes", *map(str, sizes),
                              "--warmup", str(warmup), "--repetitions", str(repetitions),
                              "--output", str(output_dir)), output_dir / "results.json")

    def evaluate_candidate(self, candidate: Path, policy: IsolationPolicy):
        raise NotImplementedError("Requires an isolated runner and a versioned ParallelBench candidate interface")


def timing_statistics(samples_ns: list[int]) -> dict:
    """Summarize real worker samples only; process startup is excluded."""
    mean = statistics.mean(samples_ns)
    deviation = statistics.pstdev(samples_ns)
    return {"mean_ns": mean, "median_ns": statistics.median(samples_ns), "minimum_ns": min(samples_ns),
            "maximum_ns": max(samples_ns), "standard_deviation_ns": deviation,
            "coefficient_of_variation": deviation / mean if mean else None}


def decide_performance(baseline: dict, candidate: dict, *, repetitions: int,
                       improvement_threshold: float, instability_threshold: float) -> tuple[str, float | None]:
    base, proposed = baseline["median_ns"], candidate["median_ns"]
    if base <= 0 or proposed <= 0:
        return "inconclusive", None
    speedup = base / proposed
    unstable = any(item["coefficient_of_variation"] is None or
                   item["coefficient_of_variation"] > instability_threshold for item in (baseline, candidate))
    if repetitions < 3 or unstable:
        return "inconclusive", speedup
    if proposed < base and (base - proposed) / base >= improvement_threshold:
        return "accepted", speedup
    if proposed > base and (proposed - base) / base >= improvement_threshold:
        return "rejected_slower", speedup
    return "inconclusive", speedup


class CPUFunctionBenchmark:
    """Small function-timing bridge; ParallelBench remains the future research engine."""
    def __init__(self, backend: ExecutionBackend):
        self.backend = backend

    def measure(self, original: str, candidate: str, spec: EvaluationSpec, *, correctness_status: str = "not_run") -> dict:
        if correctness_status != "passed":
            raise ValueError("CPU timing requires a passed correctness result")
        samples, diagnostics = {}, {}
        for label, source in (("baseline", original), ("candidate", candidate)):
            result = self.backend.run(source, spec.function, spec.benchmark.cases, spec.limits,
                                      mode="benchmark", warmup=spec.benchmark.warmup,
                                      repetitions=spec.benchmark.repetitions)
            diagnostics[label] = {"status": result.status, "error": result.error, "stderr": result.stderr,
                                  "process_wall_seconds": result.process_wall_seconds}
            if result.status != "ok":
                return {"status": result.status, "execution": diagnostics, "speedup": None}
            values = result.data.get("samples_ns")
            if (not isinstance(values, list) or len(values) != spec.benchmark.repetitions or
                    any(type(value) is not int or value < 0 or value > 10 ** 18 for value in values)):
                return {"status": "execution_error", "error": "Invalid benchmark samples from worker",
                        "execution": diagnostics, "speedup": None}
            samples[label] = {"samples_ns": values, **timing_statistics(values)}
        decision, speedup = decide_performance(samples["baseline"], samples["candidate"],
                                               repetitions=spec.benchmark.repetitions,
                                               improvement_threshold=spec.benchmark.improvement_threshold,
                                               instability_threshold=spec.benchmark.instability_threshold)
        return {"status": "completed", **samples, "speedup": speedup, "decision": decision,
                "execution": diagnostics, "execution_order": ["baseline", "candidate"],
                "measurement": "One sample times the complete benchmark cases; preparation and startup excluded",
                "interpretation": "Observed timings only; statistical significance is not established"}
