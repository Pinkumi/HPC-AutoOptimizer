"""Command-line optimization proposals and inference-only benchmarks."""
import argparse
import json
from pathlib import Path
import sys
from .agent import OptimizationAgent
from .config import ExperimentConfig
from .history import ExperimentHistory
from .llm_benchmark import run_llm_benchmark
from .models import validate_python
from .providers import MockLLMProvider, OllamaLLMProvider, ProviderError


def _settings_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="TOML configuration")
    parser.add_argument("--env-file", type=Path, help="Environment file (defaults to .env if present)")
    parser.add_argument("--provider", choices=("mock", "ollama"))
    parser.add_argument("--model", help="Override the configured model")
    parser.add_argument("--history-dir", help="Experiment artifact directory")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate optimization proposals without executing source code")
    commands = parser.add_subparsers(dest="command", required=True)
    optimize = commands.add_parser("optimize", help="Propose a Python source optimization")
    optimize.add_argument("source", type=Path)
    optimize.add_argument("--objective", required=True)
    optimize.add_argument("--target", choices=("cpu", "gpu", "general"), default="cpu")
    optimize.add_argument("--function", "--target-function", dest="target_function")
    optimize.add_argument("--constraint", action="append", default=[])
    _settings_arguments(optimize)
    benchmark = commands.add_parser("llm-benchmark", help="Measure sequential LLM inference, not program speed")
    benchmark.add_argument("--runs", type=int, default=5)
    benchmark.add_argument("--warmup", action="store_true")
    _settings_arguments(benchmark)
    arguments = list(sys.argv[1:] if argv is None else argv)
    # Preserve the Phase 1 positional-source command.
    if arguments and not arguments[0].startswith("-") and arguments[0] not in {"optimize", "llm-benchmark"}:
        arguments.insert(0, "optimize")
    args = parser.parse_args(arguments)
    try:
        env_file = args.env_file if args.env_file is not None else (Path(".env") if Path(".env").is_file() else None)
        config = ExperimentConfig.load(args.config, env_file, provider=args.provider, model=args.model,
                                       history_dir=args.history_dir)
        provider = (MockLLMProvider(config.model) if config.provider == "mock" else
                    OllamaLLMProvider.from_config(config))
        history = ExperimentHistory(config.history_dir)
        if args.command == "llm-benchmark":
            directory, record = run_llm_benchmark(provider, history, runs=args.runs, warmup=args.warmup)
            print(json.dumps({"experiment_id": record["experiment_id"], "artifact_dir": str(directory),
                              "model": config.model, "status": record["status"], "statistics": record["statistics"],
                              "cleanup_error": record.get("cleanup_error")}))
        else:
            if not args.source.is_file():
                raise ValueError(f"Input file does not exist or is not a regular file: {args.source}")
            source = args.source.read_text(encoding="utf-8")
            validate_python(source, "Input source")
            result = OptimizationAgent(provider, history).optimize(
                source, args.objective, target=args.target, target_function=args.target_function,
                constraints=args.constraint, source_file=args.source)
            print(json.dumps({"experiment_id": result.experiment_id, "artifact_dir": str(result.artifact_dir),
                              "generated_source_path": str(result.artifact_dir / "optimized.py"),
                              "model": config.model, "summary": result.proposal.summary,
                              "validation_status": "schema_valid", "syntax_status": "syntax_valid",
                              "correctness_status": "not_run", "benchmark_status": "not_run",
                              "notice": "Proposal only; performance benefits are unverified."}))
        return 0
    except (ValueError, OSError, ProviderError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
