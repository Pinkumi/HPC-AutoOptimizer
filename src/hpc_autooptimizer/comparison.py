"""Compare typed worker values without coercing exact integers into floats."""
import math
from .evaluation_spec import ComparisonSpec


def validate_value(value: object, depth: int = 0) -> None:
    if depth > 100 or not isinstance(value, dict) or set(value) != {"type", "value"}:
        raise ValueError("Unsupported worker output encoding")
    kind, data = value["type"], value["value"]
    if kind in ("list", "tuple"):
        if not isinstance(data, list):
            raise ValueError("Invalid sequence encoding")
        for item in data:
            validate_value(item, depth + 1)
    elif kind == "dict":
        if not isinstance(data, dict):
            raise ValueError("Invalid mapping encoding")
        for item in data.values():
            validate_value(item, depth + 1)
    elif not ((kind == "none" and data is None) or (kind == "bool" and type(data) is bool)
              or (kind == "int" and type(data) is int) or (kind == "str" and type(data) is str)
              or (kind == "float" and (type(data) is float and math.isfinite(data)
                                       or type(data) is str and data in ("nan", "inf", "-inf")))):
        raise ValueError("Unsupported worker output type")


def compare_values(reference: dict, candidate: dict, spec: ComparisonSpec, path: str = "$") -> str | None:
    """Return the first difference; caller validates the bounded worker protocol first."""
    left_type, right_type = reference["type"], candidate["type"]
    left, right = reference["value"], candidate["value"]
    numeric = {left_type, right_type} <= {"int", "float"}
    if left_type != right_type and not (spec.mode == "numeric" and spec.allow_int_float and numeric):
        return f"{path}: type differs ({left_type} versus {right_type})"
    if left_type in ("list", "tuple", "dict"):
        if len(left) != len(right):
            return f"{path}: structure length differs"
        if left_type == "dict" and left.keys() != right.keys():
            return f"{path}: dictionary keys differ"
        for key in left if left_type == "dict" else range(len(left)):
            difference = compare_values(left[key], right[key], spec, f"{path}[{str(key)[:40]}]")
            if difference:
                return difference
        return None
    if numeric:
        if left_type == right_type == "int":
            equal = left == right  # Exact integer arithmetic, even in numeric mode.
        else:
            if left == "nan" or right == "nan":
                equal = spec.equal_nan and left == right == "nan"
            elif left in ("inf", "-inf") or right in ("inf", "-inf"):
                equal = left == right
            elif spec.mode == "numeric":
                try:
                    equal = math.isclose(left, right, rel_tol=spec.rtol, abs_tol=spec.atol)
                except OverflowError:
                    equal = False
            else:
                equal = left == right
    else:
        equal = left == right
    return None if equal else f"{path}: values differ"
