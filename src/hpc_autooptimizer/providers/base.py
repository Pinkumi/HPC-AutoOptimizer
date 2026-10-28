from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from ..models import OptimizationProposal


class ProviderError(RuntimeError):
    """Inference transport or protocol failure with optional diagnostic data."""
    def __init__(self, message: str, *, raw_response: dict | str | None = None,
                 metrics: dict | None = None, settings: dict | None = None):
        super().__init__(message)
        self.raw_response = raw_response
        self.metrics = metrics or {}
        self.settings = settings or {}


class ProviderTimeoutError(ProviderError):
    """Inference exceeded the configured socket timeout."""


@dataclass(frozen=True)
class GenerationResponse:
    model: str
    proposal: OptimizationProposal
    metrics: dict = field(default_factory=dict)
    raw_response: dict | str | None = None
    reasoning: str | None = None
    settings: dict = field(default_factory=dict)


class LLMProvider(ABC):
    name = "unknown"

    def __init__(self, model: str):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty string")
        self.model = model

    def inference_settings(self) -> dict:
        return {}

    @abstractmethod
    def generate(self, prompt: str, *, model: str | None = None) -> GenerationResponse:
        """Generate and validate a structured optimization proposal."""
