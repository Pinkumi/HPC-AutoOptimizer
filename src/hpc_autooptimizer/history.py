"""Immutable per-attempt JSON records; no shared read/modify/write index."""
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any
from .reporting import format_optimization_summary


def _compact_metadata(value: Any) -> Any:
    """Copy metadata, omitting Ollama continuation tokens only in raw responses."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == "raw_response" and isinstance(item, dict):
                item = {field: content for field, content in item.items() if field != "context"}
            result[key] = _compact_metadata(item)
        return result
    if isinstance(value, list):
        return [_compact_metadata(item) for item in value]
    return value


class ExperimentHistory:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def record(self, metadata: dict, proposal: dict | None = None, *,
               source_filename: str = "proposed_source.txt") -> Path:
        if source_filename not in {"proposed_source.txt", "optimized.py"}:
            raise ValueError("Unsupported candidate artifact name")
        experiment_id = metadata["experiment_id"]
        if not re.fullmatch(r"[a-f0-9]{32}", experiment_id):
            raise ValueError("Invalid experiment ID")
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / experiment_id
        staging = Path(tempfile.mkdtemp(prefix=".attempt-", dir=self.root))
        try:
            (staging / "record.json").write_text(
                json.dumps(_compact_metadata(metadata), indent=2, allow_nan=False) + "\n", encoding="utf-8")
            if proposal is not None:
                (staging / "proposal.json").write_text(json.dumps(proposal, indent=2) + "\n", encoding="utf-8")
                source = proposal["optimized_code"] if "optimized_code" in proposal else proposal["proposed_source_code"]
                (staging / source_filename).write_text(source, encoding="utf-8")
            if proposal is not None or metadata.get("experiment_type") == "optimization":
                (staging / "summary.md").write_text(
                    format_optimization_summary(metadata, proposal), encoding="utf-8")
            if destination.exists():
                raise FileExistsError(f"Experiment already exists: {experiment_id}")
            staging.rename(destination)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return destination

    def records(self) -> list[dict]:
        return [json.loads(path.read_text(encoding="utf-8"))
                for path in sorted(self.root.glob("*/record.json")) if not path.parent.name.startswith(".")]
