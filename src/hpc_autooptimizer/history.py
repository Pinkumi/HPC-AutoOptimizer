"""Immutable per-attempt JSON records; no shared read/modify/write index."""
import json
from pathlib import Path
import re
import shutil
import tempfile


class ExperimentHistory:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def record(self, metadata: dict, proposal: dict | None = None) -> Path:
        experiment_id = metadata["experiment_id"]
        if not re.fullmatch(r"[a-f0-9]{32}", experiment_id):
            raise ValueError("Invalid experiment ID")
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / experiment_id
        staging = Path(tempfile.mkdtemp(prefix=".attempt-", dir=self.root))
        try:
            (staging / "record.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            if proposal is not None:
                (staging / "proposal.json").write_text(json.dumps(proposal, indent=2) + "\n", encoding="utf-8")
                # A data extension discourages accidental execution or import.
                (staging / "proposed_source.txt").write_text(proposal["proposed_source_code"], encoding="utf-8")
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
