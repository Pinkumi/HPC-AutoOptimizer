"""Validated TOML experiment configuration."""
from dataclasses import dataclass, fields
from pathlib import Path
import math
import tomllib
from urllib.parse import urlsplit


@dataclass(frozen=True)
class ExperimentConfig:
    provider: str = "mock"
    model: str = "mock-v1"
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 120.0
    history_dir: str = "experiments"

    def __post_init__(self):
        if self.provider not in {"mock", "ollama"}:
            raise ValueError("provider must be mock or ollama")
        for key in ("model", "base_url", "history_dir"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError(f"{key} must be a nonempty string")
        if (isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float))
                or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0):
            raise ValueError("timeout_seconds must be a finite positive number")
        url = urlsplit(self.base_url)
        if (url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password
                or url.query or url.fragment):
            raise ValueError("base_url must be an HTTP(S) URL without credentials, query, or fragment")
        try:
            url.port
        except ValueError as exc:
            raise ValueError("base_url has an invalid port") from exc

    @classmethod
    def from_toml(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("rb") as handle:
            data = tomllib.load(handle)
        unknown = set(data) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
        return cls(**data)
