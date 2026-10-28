"""Strict structured proposals and Python syntax inspection without execution."""
import ast
from dataclasses import asdict, dataclass
import json


class ResponseValidationError(ValueError):
    """Malformed or incomplete output, optionally carrying the original server response."""

    def __init__(self, message: str, *, raw_response: dict | str | None = None,
                 metrics: dict | None = None, settings: dict | None = None):
        super().__init__(message)
        self.raw_response = raw_response
        self.metrics = metrics or {}
        self.settings = settings or {}


def validate_python(source: str, label: str = "Source") -> ast.Module:
    """Parse only: no import, execution, or bytecode compilation of the source."""
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ResponseValidationError(f"{label} is not syntactically valid Python: {exc}") from exc


PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        **{key: {"type": "string", "minLength": 1} for key in ("optimized_code", "summary")},
        **{key: {"type": "array", "items": {"type": "string", "minLength": 1}}
           for key in ("changes", "expected_benefits", "risks", "assumptions")},
    },
    "required": ["optimized_code", "summary", "changes", "expected_benefits", "risks", "assumptions"],
    "additionalProperties": False,
}
LEGACY_FIELDS = {"proposed_source_code", "optimization_description", "expected_performance_impact",
                 "assumptions", "potential_risks"}


@dataclass(frozen=True)
class OptimizationProposal:
    optimized_code: str
    summary: str
    changes: list[str]
    expected_benefits: list[str]
    risks: list[str]
    assumptions: list[str]

    @classmethod
    def from_json(cls, raw: str) -> "OptimizationProposal":
        def unique_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate response field: {key}")
                result[key] = value
            return result
        try:
            data = json.loads(raw, object_pairs_hook=unique_pairs)
        except (ValueError, TypeError, RecursionError) as exc:
            raise ResponseValidationError(f"Invalid proposal JSON: {exc}") from exc
        if isinstance(data, dict) and set(data) == LEGACY_FIELDS:
            # Read Phase 1 responses without weakening validation of either schema.
            for key in LEGACY_FIELDS - {"assumptions", "potential_risks"}:
                if not isinstance(data[key], str) or not data[key].strip():
                    raise ResponseValidationError(f"Invalid proposal field: {key}")
            data = {"optimized_code": data["proposed_source_code"], "summary": data["optimization_description"],
                    "changes": [], "expected_benefits": [data["expected_performance_impact"]],
                    "risks": data["potential_risks"], "assumptions": data["assumptions"]}
        if not isinstance(data, dict) or set(data) != set(PROPOSAL_SCHEMA["required"]):
            raise ResponseValidationError("Proposal must contain exactly: " + ", ".join(PROPOSAL_SCHEMA["required"]))
        for key in PROPOSAL_SCHEMA["required"]:
            value = data[key]
            if key in ("optimized_code", "summary"):
                valid = isinstance(value, str) and bool(value.strip())
            else:
                valid = isinstance(value, list) and all(isinstance(x, str) and x.strip() for x in value)
            if not valid:
                raise ResponseValidationError(f"Invalid proposal field: {key}")
        validate_python(data["optimized_code"], "optimized_code")
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)

    # Read-only compatibility aliases for Phase 1 clients.
    @property
    def proposed_source_code(self) -> str:
        return self.optimized_code

    @property
    def optimization_description(self) -> str:
        return self.summary

    @property
    def expected_performance_impact(self) -> str:
        return "; ".join(self.expected_benefits)

    @property
    def potential_risks(self) -> list[str]:
        return self.risks
