import json
from .base import GenerationResponse, LLMProvider
from ..models import OptimizationProposal, ResponseValidationError


class MockLLMProvider(LLMProvider):
    """Deterministic fixture; the default proposal makes no optimization claim."""
    name = "mock"

    def __init__(self, model: str = "mock-v1", response: str | None = None):
        super().__init__(model)
        self.response = response

    def generate(self, prompt: str, *, model: str | None = None) -> GenerationResponse:
        selected = self.model if model is None else model
        if not isinstance(selected, str) or not selected.strip():
            raise ValueError("model must be a nonempty string")
        raw = self.response
        if raw is None:
            source = json.loads(prompt)["source_code"]
            raw = json.dumps({
                "optimized_code": source,
                "summary": "Mock baseline: source preserved unchanged.",
                "changes": [],
                "expected_benefits": ["Unknown; no measurements performed."],
                "assumptions": ["This is a deterministic workflow fixture."],
                "risks": ["Correctness and performance have not been evaluated."],
            })
        try:
            proposal = OptimizationProposal.from_json(raw)
        except ResponseValidationError as exc:
            exc.raw_response = raw
            raise
        return GenerationResponse(selected, proposal, raw_response=raw)
