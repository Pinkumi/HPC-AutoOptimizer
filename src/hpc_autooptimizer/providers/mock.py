import json
from .base import GenerationResponse, LLMProvider
from ..models import OptimizationProposal


class MockLLMProvider(LLMProvider):
    """Deterministic fixture; the default proposal makes no optimization claim."""
    def __init__(self, model: str = "mock-v1", response: str | None = None):
        super().__init__(model)
        self.response = response

    def generate(self, prompt: str) -> GenerationResponse:
        raw = self.response
        if raw is None:
            source = json.loads(prompt)["source_code"]
            raw = json.dumps({
                "proposed_source_code": source,
                "optimization_description": "Mock baseline: source preserved unchanged.",
                "expected_performance_impact": "Unknown; no measurements performed.",
                "assumptions": ["This is a deterministic workflow fixture."],
                "potential_risks": ["Correctness and performance have not been evaluated."],
            })
        return GenerationResponse(self.model, OptimizationProposal.from_json(raw))
