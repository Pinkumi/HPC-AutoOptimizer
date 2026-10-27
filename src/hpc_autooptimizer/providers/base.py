from abc import ABC, abstractmethod
from dataclasses import dataclass
from ..models import OptimizationProposal


class ProviderError(RuntimeError):
    """Inference transport or protocol failure."""


class ProviderTimeoutError(ProviderError):
    """Inference exceeded the configured socket timeout."""


@dataclass(frozen=True)
class GenerationResponse:
    model: str
    proposal: OptimizationProposal


class LLMProvider(ABC):
    def __init__(self, model: str):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty string")
        self.model = model

    @abstractmethod
    def generate(self, prompt: str) -> GenerationResponse:
        """Generate and validate a structured optimization proposal."""
