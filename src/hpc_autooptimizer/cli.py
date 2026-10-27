import argparse
import json
from pathlib import Path
import sys
from .agent import OptimizationAgent
from .config import ExperimentConfig
from .history import ExperimentHistory
from .providers import MockLLMProvider, OllamaLLMProvider, ProviderError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Propose code optimizations without executing code")
    parser.add_argument("source", type=Path)
    parser.add_argument("--objective", required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    try:
        config = ExperimentConfig.from_toml(args.config) if args.config else ExperimentConfig()
        provider = (MockLLMProvider(config.model) if config.provider == "mock" else
                    OllamaLLMProvider(config.model, config.base_url, config.timeout_seconds))
        result = OptimizationAgent(provider, ExperimentHistory(config.history_dir)).optimize(
            args.source.read_text(encoding="utf-8"), args.objective)
        print(json.dumps({"experiment_id": result.experiment_id, "artifact_dir": str(result.artifact_dir),
                          "validation_status": "schema_valid", "benchmark_status": "not_run"}))
        return 0
    except (ValueError, OSError, ProviderError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
