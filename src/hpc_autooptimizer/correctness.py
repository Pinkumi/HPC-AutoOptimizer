"""Explicit-case correctness checks. Never import evaluated modules in this process."""
from .comparison import compare_values, validate_value
from .evaluation_spec import EvaluationSpec
from .execution import ExecutionBackend, ExecutionResult


def _diagnostic(result: ExecutionResult) -> dict:
    return {"status": result.status, "error": result.error, "stderr": result.stderr,
            "process_wall_seconds": result.process_wall_seconds}


class CaseValidator:
    def __init__(self, backend: ExecutionBackend):
        self.backend = backend

    def validate(self, original: str, candidate: str, spec: EvaluationSpec) -> dict:
        results = []
        # Timing inputs are correctness-gated too, rather than trusting unseen benchmark cases.
        cases = [("correctness", case) for case in spec.correctness_cases]
        cases += [("benchmark", case) for case in spec.benchmark.cases]
        for index, (origin, case) in enumerate(cases):
            reference = self.backend.run(original, spec.function, [case], spec.limits)
            proposed = self.backend.run(candidate, spec.function, [case], spec.limits)
            entry = {"case_index": index, "input_group": origin,
                     "reference": _diagnostic(reference), "candidate": _diagnostic(proposed)}
            if reference.status != "ok" or proposed.status != "ok":
                entry["status"] = reference.status if reference.status != "ok" else proposed.status
            else:
                try:
                    left, right = reference.data.get("output"), proposed.data.get("output")
                    validate_value(left)
                    validate_value(right)
                    difference = compare_values(left, right, spec.comparison)
                    entry.update(status="failed" if difference else "passed", difference=difference)
                    if difference:
                        # Bounded previews are sufficient to diagnose a failing case.
                        entry.update(reference_output_preview=str(left)[:500], candidate_output_preview=str(right)[:500])
                except (ValueError, TypeError, RecursionError) as exc:
                    entry.update(status="unsupported", difference=str(exc)[:300])
            results.append(entry)
        reference_errors = [item["reference"]["status"] for item in results if item["reference"]["status"] != "ok"]
        failures = [item["status"] for item in results if item["status"] != "passed"]
        # A proven mismatch rejects the candidate even if another candidate case crashes.
        # Reference failures instead make the evaluation itself unreliable.
        if reference_errors:
            overall = reference_errors[0]
        elif "failed" in failures:
            overall = "failed"
        else:
            overall = failures[0] if failures else "passed"
        return {"status": overall, "cases": results,
                "failed_cases": [item["case_index"] for item in results if item["status"] != "passed"],
                "scope": "Returned values on explicit cases only; no proof for untested inputs or side effects"}
