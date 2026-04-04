"""
Field validation helpers used by the graders.

Each validator returns True if the value matches the expected type,
False otherwise. These are intentionally simple and deterministic.
"""

import re
from typing import Any, Dict, List, Tuple


EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
PHONE_RE = re.compile(r"^\+?[1-9]\d{6,14}$")
URL_RE = re.compile(r"^https?://[^\s]+$")
ISO_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(Z|[+-]\d{2}:\d{2})$"
)


def validate_email(value: Any) -> bool:
    return isinstance(value, str) and bool(EMAIL_RE.match(value))


def validate_phone(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    cleaned = value.replace(" ", "").replace("-", "")
    return bool(PHONE_RE.match(cleaned))


def validate_url(value: Any) -> bool:
    return isinstance(value, str) and bool(URL_RE.match(value))


def validate_datetime(value: Any) -> bool:
    return isinstance(value, str) and bool(ISO_DATETIME_RE.match(value))


def validate_enum(value: Any, allowed_values: List[str]) -> bool:
    return isinstance(value, str) and value in allowed_values


def validate_field_type(value: Any, expected_type: str) -> bool:
    """Check if a value matches the expected type string from the spec.

    Supported types: string, integer, float, boolean, email, datetime,
    url, phone, enum:val1,val2, object, array.
    """
    if value is None:
        return False

    if expected_type == "string":
        return isinstance(value, str)
    elif expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    elif expected_type == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    elif expected_type == "boolean":
        return isinstance(value, bool)
    elif expected_type == "email":
        return validate_email(value)
    elif expected_type == "datetime":
        return validate_datetime(value)
    elif expected_type == "url":
        return validate_url(value)
    elif expected_type == "phone":
        return validate_phone(value)
    elif expected_type.startswith("enum:"):
        allowed = expected_type.split(":", 1)[1].split(",")
        return validate_enum(value, allowed)
    elif expected_type == "object":
        return isinstance(value, dict)
    elif expected_type == "array":
        return isinstance(value, list)
    else:
        # Unknown type, accept anything non-None
        return True


def validate_request_against_spec(
    request: Dict[str, Any],
    spec: Dict[str, Any],
) -> Tuple[float, str]:
    """Validate a request body against its spec.

    Returns (score, feedback_string) where score is 0.0 to 1.0
    based on how many checks pass.
    """
    checks = []
    total = 0
    passed = 0

    # Check required fields are present and non-null
    for field in spec["required_fields"]:
        total += 1
        if field in request and request[field] is not None:
            passed += 1
            checks.append(f"  {field}: PRESENT")
        else:
            checks.append(f"  {field}: MISSING")

    # Check field types for fields that are present
    for field, expected_type in spec["field_types"].items():
        if field not in request or request[field] is None:
            continue
        total += 1
        if validate_field_type(request[field], expected_type):
            passed += 1
            checks.append(f"  {field} type: VALID ({expected_type})")
        else:
            checks.append(f"  {field} type: INVALID (expected {expected_type})")

    # Check no extra unknown fields
    all_known = set(spec["required_fields"]) | set(spec.get("optional_fields", []))
    for field in request:
        if field not in all_known:
            total += 1
            checks.append(f"  {field}: UNKNOWN FIELD (not in spec)")

    score = passed / total if total > 0 else 0.0
    feedback = f"Validation: {passed}/{total} checks passed.\n" + "\n".join(checks)
    return round(score, 4), feedback


def validate_headers_against_spec(
    headers: Dict[str, str],
    spec: Dict[str, Any],
) -> Tuple[float, str]:
    """Validate request headers against the spec's required_headers.

    Returns (score, feedback_string).
    """
    required = spec.get("required_headers", {})
    if not required:
        return 1.0, "No required headers."

    total = len(required)
    passed = 0
    checks = []

    for header_name in required:
        if header_name in headers:
            passed += 1
            checks.append(f"  {header_name}: PRESENT")
        else:
            checks.append(f"  {header_name}: MISSING")

    score = passed / total if total > 0 else 1.0
    feedback = f"Headers: {passed}/{total} present.\n" + "\n".join(checks)
    return round(score, 4), feedback
