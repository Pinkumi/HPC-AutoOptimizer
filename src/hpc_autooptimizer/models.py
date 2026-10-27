"""Shared proposal schema and strict response validation."""
from dataclasses import asdict, dataclass
import json


class ResponseValidationError(ValueError):
    """The model output does not match the proposal schema."""


PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        **{key: {"type": "string", "minLength": 1} for key in (
            "proposed_source_code", "optimization_description", "expected_performance_impact")},
        **{key: {"type": "array", "items": {"type": "string", "minLength": 1}}
           for key in ("assumptions", "potential_risks")},
    },
    "required": ["proposed_source_code", "optimization_description", "expected_performance_impact",
                 "assumptions", "potential_risks"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class OptimizationProposal:
    proposed_source_code: str
    optimization_description: str
    expected_performance_impact: str
    assumptions: list[str]
    potential_risks: list[str]

    @classmethod
    def from_json(cls, raw: str) -> "OptimizationProposal":
        def unique_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ResponseValidationError(f"Duplicate response field: {key}")
                result[key] = value
            return result
        try:
            data = json.loads(raw, object_pairs_hook=unique_pairs)
        except (ValueError, TypeError) as exc:
            raise ResponseValidationError(f"Invalid proposal JSON: {exc}") from exc
        if not isinstance(data, dict) or set(data) != set(PROPOSAL_SCHEMA["required"]):
            raise ResponseValidationError("Proposal must contain exactly the five required schema fields")
        for key in PROPOSAL_SCHEMA["required"]:
            value = data[key]
            if key in ("assumptions", "potential_risks"):
                valid = isinstance(value, list) and all(isinstance(x, str) and x.strip() for x in value)
            else:
                valid = isinstance(value, str) and bool(value.strip())
            if not valid:
                raise ResponseValidationError(f"Invalid proposal field: {key}")
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)
