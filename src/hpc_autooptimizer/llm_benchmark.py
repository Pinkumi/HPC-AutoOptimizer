"""Sequential inference measurements, independent of candidate program benchmarking."""
from datetime import datetime, timezone
from pathlib import Path
import statistics
from time import perf_counter
from uuid import uuid4

from .agent import build_prompt, code_hash
from .history import ExperimentHistory
from .providers import LLMProvider, OllamaLLMProvider, ProviderError
from .models import ResponseValidationError


BENCHMARK_SOURCE = """def sum_squares(values):
    total = 0.0
    for value in values:
        total += value * value
    return total
"""
BENCHMARK_PROMPT = build_prompt(
    BENCHMARK_SOURCE, "Improve numerical-computing efficiency while preserving behavior", "cpu",
    "sum_squares", ["Use the Python standard library only", "Do not change the function signature"])


def summarize(values: list[float]) -> dict:
    if not values:
        return {key: None for key in ("mean", "median", "minimum", "maximum", "standard_deviation")}
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "minimum": min(values), "maximum": max(values), "standard_deviation": statistics.pstdev(values)}


def run_llm_benchmark(provider: LLMProvider, history: ExperimentHistory, *, runs: int = 5,
                      warmup: bool = False) -> tuple[Path, dict]:
    """One model per invocation, sequential requests, release that model on completion."""
    if type(runs) is not int or runs <= 0:
        raise ValueError("runs must be a positive integer")
    record = {"schema_version": 2, "experiment_id": uuid4().hex, "experiment_type": "llm_benchmark",
              "timestamp": datetime.now(timezone.utc).isoformat(), "provider": provider.name,
              "model_name": provider.model, "inference_configuration": provider.inference_settings(),
              "prompt": BENCHMARK_PROMPT, "prompt_hash": code_hash(BENCHMARK_PROMPT),
              "requested_runs": runs, "warmup": None, "runs": [], "status": "running", "error": None,
              "correctness_status": "not_run", "benchmark_status": "not_run"}
    primary_error = None
    model_checked = False
    try:
        if isinstance(provider, OllamaLLMProvider):
            record["installed_model"] = provider.require_model()
            model_checked = True
        for index in range(-1 if warmup else 0, runs):
            sample = {"run_index": index, "status": "pending"}
            if index == -1:
                record["warmup"] = sample
            else:
                record["runs"].append(sample)
            start = perf_counter()
            try:
                response = provider.generate(BENCHMARK_PROMPT)
                sample.update(status="completed", metrics=response.metrics, raw_response=response.raw_response,
                              effective_settings=response.settings)
            except (ProviderError, ResponseValidationError) as exc:
                sample.update(status="failed", error=str(exc), metrics=exc.metrics, raw_response=exc.raw_response,
                              effective_settings=exc.settings)
                raise
            finally:
                sample["client_duration_seconds"] = perf_counter() - start
        record["status"] = "completed"
    except (ProviderError, ResponseValidationError) as exc:
        record.update(status="failed", error=str(exc), raw_response=exc.raw_response,
                      timing_metrics=exc.metrics)
        primary_error = exc
    finally:
        if isinstance(provider, OllamaLLMProvider) and model_checked:
            try:
                provider.unload()
                record["model_unloaded"] = True
            except ProviderError as exc:
                record["model_unloaded"] = False
                record["cleanup_error"] = str(exc)
        successful = [sample for sample in record["runs"] if sample["status"] == "completed"]
        record["statistics"] = {"client_duration_seconds": summarize([
            sample["client_duration_seconds"] for sample in successful])}
        for metric in ("output_tokens_per_second", "prompt_tokens_per_second", "total_inference_seconds",
                       "load_seconds", "prompt_processing_seconds", "output_generation_seconds", "eval_count"):
            record["statistics"][metric] = summarize([
                sample["metrics"][metric] for sample in successful if sample["metrics"].get(metric) is not None])
        directory = history.record(record)
    if primary_error is not None:
        raise ProviderError(f"LLM benchmark failed: {primary_error}. Results saved in {directory}") from primary_error
    return directory, record
