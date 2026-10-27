"""ParallelBench CLI planning only; never launches processes."""
from dataclasses import dataclass
from pathlib import Path
from .evaluation import IsolationPolicy


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
