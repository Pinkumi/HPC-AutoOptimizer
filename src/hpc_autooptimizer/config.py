"""Validated TOML, environment, and deliberately simple .env configuration."""
from dataclasses import asdict, dataclass, fields
import math
import os
from pathlib import Path
import re
import tomllib
from typing import Mapping
from urllib.parse import urlsplit


ENV_FIELDS = {
    "HPC_LLM_PROVIDER": ("provider", str),
    "HPC_OLLAMA_BASE_URL": ("base_url", str),
    "HPC_OLLAMA_MODEL": ("model", str),
    "HPC_OLLAMA_HOST_HEADER": ("host_header", str),
    "HPC_OLLAMA_TIMEOUT": ("timeout_seconds", float),
    "HPC_OLLAMA_KEEP_ALIVE": ("keep_alive", str),
    "HPC_OLLAMA_NUM_CTX": ("num_ctx", int),
    "HPC_OLLAMA_NUM_PREDICT": ("num_predict", int),
    "HPC_OLLAMA_TEMPERATURE": ("temperature", float),
    "HPC_OLLAMA_THINK": ("think", str),
    "HPC_HISTORY_DIR": ("history_dir", str),
}


def read_env_file(path: Path) -> dict[str, str]:
    """Read KEY=value with optional surrounding quotes; never evaluate shell text."""
    result = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or key not in ENV_FIELDS:
            raise ValueError(f"Unsupported .env entry at line {number}: {key}")
        if value.startswith(('"', "'")):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f"Unclosed .env quote at line {number}")
            value = value[1:-1]
        result[key] = value
    return result


def _toml_values(path: str | Path) -> dict:
    with Path(path).open("rb") as handle:
        data = tomllib.load(handle)
    unknown = set(data) - {field.name for field in fields(ExperimentConfig)}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
    return data


@dataclass(frozen=True)
class ExperimentConfig:
    provider: str = "mock"
    model: str = "mock-v1"
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 180.0
    history_dir: str = "experiments"
    host_header: str | None = None
    keep_alive: str = "15m"
    num_ctx: int = 4096
    num_predict: int = 2048
    temperature: float = 0.2
    think: bool | None = False

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or self.provider not in {"mock", "ollama"}:
            raise ValueError("provider must be mock or ollama")
        for key in ("model", "base_url", "history_dir"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError(f"{key} must be a nonempty string")
        for key, positive in (("timeout_seconds", True), ("temperature", False)):
            value = getattr(self, key)
            try:
                finite = isinstance(value, (int, float)) and math.isfinite(value)
            except OverflowError:
                finite = False
            if (isinstance(value, bool) or not finite or (value <= 0 if positive else value < 0)):
                raise ValueError(f"{key} must be a finite {'positive' if positive else 'nonnegative'} number")
        for key in ("num_ctx", "num_predict"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.think is not None and type(self.think) is not bool:
            raise ValueError("think must be true, false, or null (model default)")
        if not isinstance(self.keep_alive, str) or not re.fullmatch(r"(?:0|[0-9]+(?:\.[0-9]+)?(?:ms|s|m|h))", self.keep_alive):
            raise ValueError("keep_alive must be 0 or a nonnegative duration such as 15m")
        if self.host_header is not None and (not isinstance(self.host_header, str) or not
                re.fullmatch(r"[A-Za-z0-9.-]+(?::[0-9]{1,5})?", self.host_header)):
            raise ValueError("host_header must be a hostname with optional port (no whitespace)")
        if any(ord(char) < 33 for char in self.base_url):
            raise ValueError("base_url must not contain whitespace or control characters")
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
        return cls(**_toml_values(path))

    @classmethod
    def load(cls, path: str | Path | None = None, env_file: str | Path | None = None,
             environ: Mapping[str, str] | None = None, **overrides) -> "ExperimentConfig":
        """Precedence: defaults < TOML < .env < process environment < explicit overrides."""
        data = _toml_values(path) if path else {}
        env = read_env_file(Path(env_file)) if env_file is not None else {}
        env.update(os.environ if environ is None else environ)
        for variable, (key, convert) in ENV_FIELDS.items():
            if variable not in env:
                continue
            value = env[variable]
            try:
                if key == "think":
                    if value.lower() not in {"true", "false", "null"}:
                        raise ValueError("use true, false, or null")
                    data[key] = {"true": True, "false": False, "null": None}[value.lower()]
                elif key == "host_header":
                    data[key] = value or None
                else:
                    data[key] = convert(value)
            except ValueError as exc:
                raise ValueError(f"Invalid {variable}: {exc}") from exc
        data.update({key: value for key, value in overrides.items() if value is not None})
        if data.get("provider") == "ollama" and "model" not in data:
            raise ValueError("Select an Ollama model with --model, model in TOML, or HPC_OLLAMA_MODEL")
        return cls(**data)

    def inference_settings(self) -> dict:
        return {key: value for key, value in asdict(self).items() if key not in {"history_dir", "provider", "model"}}
