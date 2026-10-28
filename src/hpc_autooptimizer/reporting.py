"""Readable Phase 2 reports; inference timings never imply program performance."""
import math


def _text(value: object) -> str:
    return " ".join(str(value).split()) if value is not None else "Not available"


def _metric(value: object, unit: str = "", decimals: int = 3) -> str:
    try:
        valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    except OverflowError:
        valid = False
    return f"{value:.{decimals}f}{unit}" if valid else "Not available"


def optimization_details(record: dict) -> list[tuple[str, str]]:
    """Shared display values for Markdown and terminal output; no metric recalculation."""
    metrics = record.get("timing_metrics") or {}
    syntax = {"syntax_valid": "Valid (syntax only)", "syntax_invalid": "Invalid"}.get(
        record.get("syntax_status"), "Not checked")
    return [
        ("Experiment ID", _text(record.get("experiment_id"))),
        ("Model", _text(record.get("model_name"))),
        ("Output file", _text(record.get("generated_source_path"))),
        ("Total inference time", _metric(metrics.get("total_inference_seconds"), " s")),
        ("Model loading time", _metric(metrics.get("load_seconds"), " s")),
        ("Prompt processing time", _metric(metrics.get("prompt_processing_seconds"), " s")),
        ("Generated tokens", _metric(metrics.get("total_generated_tokens"), decimals=0)),
        ("Generation speed", _metric(metrics.get("output_tokens_per_second"), " tokens/s")),
        ("Python syntax", syntax),
        ("Correctness verification", "Not evaluated"),
        ("Performance benchmark", "Not evaluated"),
    ]


def format_cli_summary(record: dict) -> str:
    omit = {"Model loading time", "Prompt processing time", "Generated tokens"}
    lines = [f"{label}: {value}" for label, value in optimization_details(record) if label not in omit]
    lines.append("Proposal only; behavioral equivalence and performance benefits are unverified.")
    return "\n".join(lines)


def _quoted(value: str) -> str:
    # Keep multiline model statements visibly attributed as quotations.
    return "\n".join("> " + line for line in value.splitlines())


def format_optimization_summary(record: dict, proposal: dict | None = None) -> str:
    details = dict(optimization_details(record))
    lines = ["# Optimization experiment", "",
             "Proposal only. Behavioral equivalence and performance benefits have not been evaluated.", "",
             "## General information", "",
             f"- Experiment ID: {details['Experiment ID']}",
             f"- Timestamp: {_text(record.get('timestamp'))}",
             f"- Original source file: {_text(record.get('original_source_path') or record.get('source_file'))}",
             f"- Selected LLM model: {details['Model']}",
             f"- Optimization objective: {_text(record.get('objective'))}",
             f"- Target hardware: {_text(record.get('target_hardware'))}",
             f"- Output file: {details['Output file']}", "",
             "## Proposed optimization", ""]
    if proposal is not None:
        lines += ["Model-proposed summary (unverified):", "", _quoted(proposal.get("summary") or
                  proposal.get("optimization_description") or "Not available"), ""]
        for title, key, legacy in (("Proposed changes (unverified)", "changes", None),
                                   ("Expected benefits (model hypotheses, not measurements)", "expected_benefits", "expected_performance_impact"),
                                   ("Potential correctness risks (model-reported)", "risks", "potential_risks")):
            items = proposal.get(key) or (proposal.get(legacy) if legacy else None) or []
            if isinstance(items, str):
                items = [items]
            lines += [f"### {title}", ""]
            lines += [_quoted(str(item)) + "\n" for item in items] if items else ["Not provided.", ""]
    else:
        lines += ["No validated proposal was produced.", ""]
    lines += ["## Inference metrics", ""]
    for label in ("Total inference time", "Model loading time", "Prompt processing time", "Generated tokens", "Generation speed"):
        lines.append(f"- {label}: {details[label]}")
    lines += ["", "## Validation status", ""]
    for label in ("Python syntax", "Correctness verification", "Performance benchmark"):
        lines.append(f"- {label}: {details[label]}")
    if record.get("error"):
        lines += ["", "## Generation error", "", _quoted(str(record["error"]))]
    return "\n".join(lines) + "\n"
