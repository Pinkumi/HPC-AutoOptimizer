"""Readable optimization and evaluation reports with explicit validation scope."""
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


def format_evaluation_report(record: dict, directory: object = None) -> str:
    benchmark = record["benchmark"]
    lines = [f"Evaluation ID: {record['experiment_id']}",
             f"Backend: {record['execution_backend']} ({record['isolation_mode']})",
             f"Syntax: {record['syntax_status']}", f"Correctness: {record['correctness']['status']}",
             f"Failed cases: {record['correctness'].get('failed_cases', [])}",
             f"CPU benchmark: {benchmark['status']}",
             f"Speedup (baseline median / candidate median): {_metric(benchmark.get('speedup'), 'x')}",
             f"Decision: {record['decision']}"]
    if record.get("error"):
        lines.append(f"Error: {_text(record['error'])}")
    if directory is not None:
        lines.append(f"Results: {directory}")
    if record["isolation_mode"] == "trusted_development_only":
        lines.append("TRUSTED FIXTURES ONLY: no filesystem or network isolation; never use this mode for LLM output.")
    lines.append("Case agreement is limited to the contract; measured timings do not establish statistical significance.")
    return "\n".join(lines)


def format_evaluation_summary(record: dict) -> str:
    lines = ["# Evaluation experiment", "", f"Timestamp: {record['timestamp']}", "",
             f"Original source: {record['original_source_path']}", "",
             f"Candidate source: {record['candidate_source_path']}", "",
             "## Results", "", "```text", format_evaluation_report(record), "```", "",
             "## CPU measurements", ""]
    for label in ("baseline", "candidate"):
        statistics = record["benchmark"].get(label)
        if statistics:
            lines.append(f"- {label}: median {_metric(statistics['median_ns'], ' ns')}, "
                         f"mean {_metric(statistics['mean_ns'], ' ns')}, "
                         f"standard deviation {_metric(statistics['standard_deviation_ns'], ' ns')}; "
                         f"{len(statistics['samples_ns'])} samples")
    if record["benchmark"]["status"] == "not_run":
        lines.append("Not evaluated: correctness must pass before any workload timing.")
    lines += ["", "Full contract, hashes, case diagnostics, limits, and timing samples are in record.json.",
              "The original optimization experiment has not been modified."]
    return "\n".join(lines) + "\n"
