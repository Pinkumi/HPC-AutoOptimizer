"""Normalize optional Ollama nanosecond timings without inventing measurements."""
import math
from typing import Any


def inference_metrics(envelope: dict[str, Any], client_seconds: float) -> dict:
    def nonnegative(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        try:
            return value if math.isfinite(value) and value >= 0 else None
        except OverflowError:
            return None

    result = {key: nonnegative(envelope.get(key)) for key in (
        "total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration")}
    # Counts and durations are integers in the Ollama protocol.
    result = {key: value if type(value) is int else None for key, value in result.items()}
    result["done_reason"] = envelope.get("done_reason") if isinstance(envelope.get("done_reason"), str) else None
    for field, output in (("total_duration", "total_inference_seconds"), ("load_duration", "load_seconds"),
                          ("prompt_eval_duration", "prompt_processing_seconds"), ("eval_duration", "output_generation_seconds")):
        result[output] = result[field] / 1e9 if result[field] is not None else None
    for count, seconds, output in (("prompt_eval_count", "prompt_processing_seconds", "prompt_tokens_per_second"),
                                   ("eval_count", "output_generation_seconds", "output_tokens_per_second")):
        rate = result[count] / result[seconds] if result[count] is not None and result[seconds] else None
        result[output] = nonnegative(rate)
    result["total_generated_tokens"] = result["eval_count"]
    result["client_duration_seconds"] = nonnegative(client_seconds)
    return result
