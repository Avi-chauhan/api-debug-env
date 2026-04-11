"""
10 error injection functions for the API Debug Environment.

Each injector takes a valid request + headers + spec + RNG and returns:
  (broken_request, broken_headers, ground_truth)

ground_truth contains the error_type, affected_fields, and the original
valid request/headers so the grader knows the correct answer.
"""

import copy
import random as random_module
from typing import Any, Dict, List, Tuple

GroundTruth = Dict[str, Any]
InjectorResult = Tuple[Dict[str, Any], Dict[str, str], GroundTruth]


def _ground_truth(
    error_type: str,
    affected_fields: List[str],
    valid_request: Dict[str, Any],
    valid_headers: Dict[str, str],
) -> GroundTruth:
    """Build a standard ground truth dict."""
    return {
        "error_type": error_type,
        "affected_fields": affected_fields,
        "valid_request": valid_request,
        "valid_headers": valid_headers,
    }


# =========================================================================
# 1. missing_required_field
# =========================================================================

def inject_missing_required_field(
    request: Dict[str, Any],
    headers: Dict[str, str],
    spec: Dict[str, Any],
    rng: random_module.Random,
) -> InjectorResult:
    """Remove a random required field from the request."""
    broken = copy.deepcopy(request)
    candidates = [f for f in spec["required_fields"] if f in broken]
    if not candidates:
        return broken, headers, _ground_truth(
            "missing_required_field", [], request, headers
        )
    field = rng.choice(candidates)
    del broken[field]
    return broken, headers, _ground_truth(
        "missing_required_field", [field], request, headers
    )


# =========================================================================
# 2. wrong_field_type
# =========================================================================

def inject_wrong_field_type(
    request: Dict[str, Any],
    headers: Dict[str, str],
    spec: Dict[str, Any],
    rng: random_module.Random,
) -> InjectorResult:
    """Change a field's value to the wrong type (e.g. int to string)."""
    broken = copy.deepcopy(request)
    candidates = [f for f in spec["required_fields"] if f in broken]
    if not candidates:
        return broken, headers, _ground_truth(
            "wrong_field_type", [], request, headers
        )
    field = rng.choice(candidates)
    original = broken[field]

    # Pick a wrong type based on what the original is
    if isinstance(original, int):
        broken[field] = str(original)
    elif isinstance(original, float):
        broken[field] = str(original)
    elif isinstance(original, bool):
        broken[field] = "true"
    elif isinstance(original, str):
        broken[field] = 12345
    elif isinstance(original, list):
        broken[field] = "should_be_array"
    elif isinstance(original, dict):
        broken[field] = "should_be_object"
    else:
        broken[field] = "wrong_type"

    return broken, headers, _ground_truth(
        "wrong_field_type", [field], request, headers
    )


# =========================================================================
# 3. invalid_email_format
# =========================================================================

def inject_invalid_email_format(
    request: Dict[str, Any],
    headers: Dict[str, str],
    spec: Dict[str, Any],
    rng: random_module.Random,
) -> InjectorResult:
    """Corrupt an email field to an invalid format."""
    broken = copy.deepcopy(request)
    email_fields = [
        f for f in spec["field_types"]
        if spec["field_types"][f] == "email" and f in broken
    ]
    if not email_fields:
        # Fallback: inject a missing field instead
        return inject_missing_required_field(request, headers, spec, rng)

    field = rng.choice(email_fields)
    bad_emails = ["not-an-email", "user@", "@domain.com", "user@.com", "user space@example.com"]
    broken[field] = rng.choice(bad_emails)
    return broken, headers, _ground_truth(
        "invalid_email_format", [field], request, headers
    )


# =========================================================================
# 4. missing_auth_header
# =========================================================================

def inject_missing_auth_header(
    request: Dict[str, Any],
    headers: Dict[str, str],
    spec: Dict[str, Any],
    rng: random_module.Random,
) -> InjectorResult:
    """Remove the Authorization header."""
    broken_headers = copy.deepcopy(headers)
    if "Authorization" in broken_headers:
        del broken_headers["Authorization"]
        return request, broken_headers, _ground_truth(
            "missing_auth_header", ["Authorization"], request, headers
        )
    # If no auth header exists in spec, remove Content-Type instead
    if "Content-Type" in broken_headers:
        del broken_headers["Content-Type"]
        return request, broken_headers, _ground_truth(
            "missing_auth_header", ["Content-Type"], request, headers
        )
    return request, broken_headers, _ground_truth(
        "missing_auth_header", [], request, headers
    )


# =========================================================================
# 5. extra_unknown_field
# =========================================================================

def inject_extra_unknown_field(
    request: Dict[str, Any],
    headers: Dict[str, str],
    spec: Dict[str, Any],
    rng: random_module.Random,
) -> InjectorResult:
    """Add a field that is not in the spec."""
    broken = copy.deepcopy(request)
    unknown_fields = [
        ("unknown_field", "unexpected_value"),
        ("debug_mode", True),
        ("internal_id", 99999),
        ("_private", "should_not_exist"),
        ("extra_data", {"nested": "bad"}),
    ]
    field_name, field_value = rng.choice(unknown_fields)
    broken[field_name] = field_value
    return broken, headers, _ground_truth(
        "extra_unknown_field", [field_name], request, headers
    )


# =========================================================================
# 6. null_value_in_required
# =========================================================================

def inject_null_value_in_required(
    request: Dict[str, Any],
    headers: Dict[str, str],
    spec: Dict[str, Any],
    rng: random_module.Random,
) -> InjectorResult:
    """Set a required field to null."""
    broken = copy.deepcopy(request)
    candidates = [f for f in spec["required_fields"] if f in broken]
    if not candidates:
        return broken, headers, _ground_truth(
            "null_value_in_required", [], request, headers
        )
    field = rng.choice(candidates)
    broken[field] = None
    return broken, headers, _ground_truth(
        "null_value_in_required", [field], request, headers
    )


# =========================================================================
# 7. wrong_http_method
# =========================================================================

def inject_wrong_http_method(
    request: Dict[str, Any],
    headers: Dict[str, str],
    spec: Dict[str, Any],
    rng: random_module.Random,
) -> InjectorResult:
    """Indicate the wrong HTTP method was used.

    The error is stored in the ground truth. The request body stays the same
    but the observation will show a different http_method.
    """
    all_methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    correct = spec["http_method"]
    wrong_methods = [m for m in all_methods if m != correct]
    wrong = rng.choice(wrong_methods)

    gt = _ground_truth("wrong_http_method", ["http_method"], request, headers)
    gt["wrong_method"] = wrong
    gt["correct_method"] = correct
    return request, headers, gt


# =========================================================================
# 8. malformed_json_value
# =========================================================================

def inject_malformed_json_value(
    request: Dict[str, Any],
    headers: Dict[str, str],
    spec: Dict[str, Any],
    rng: random_module.Random,
) -> InjectorResult:
    """Corrupt a field value so it looks like broken JSON.

    Since we work with Python dicts (already parsed), we simulate this
    by inserting strings that look like malformed JSON fragments.
    """
    broken = copy.deepcopy(request)
    candidates = [f for f in spec["required_fields"] if f in broken]
    if not candidates:
        return broken, headers, _ground_truth(
            "malformed_json_value", [], request, headers
        )
    field = rng.choice(candidates)
    bad_values = [
        "{broken",
        "[unclosed",
        "value with 'mixed\" quotes",
        "undefined",
        "NaN",
    ]
    broken[field] = rng.choice(bad_values)
    return broken, headers, _ground_truth(
        "malformed_json_value", [field], request, headers
    )


# =========================================================================
# 9. invalid_enum_value
# =========================================================================

def inject_invalid_enum_value(
    request: Dict[str, Any],
    headers: Dict[str, str],
    spec: Dict[str, Any],
    rng: random_module.Random,
) -> InjectorResult:
    """Use a value not in the enum list for an enum field."""
    broken = copy.deepcopy(request)
    enum_fields = [
        f for f in spec["field_types"]
        if spec["field_types"][f].startswith("enum:") and f in broken
    ]
    if not enum_fields:
        # Fallback: inject wrong type instead
        return inject_wrong_field_type(request, headers, spec, rng)

    field = rng.choice(enum_fields)
    broken[field] = "INVALID_ENUM_VALUE"
    return broken, headers, _ground_truth(
        "invalid_enum_value", [field], request, headers
    )


# =========================================================================
# 10. datetime_format_error
# =========================================================================

def inject_datetime_format_error(
    request: Dict[str, Any],
    headers: Dict[str, str],
    spec: Dict[str, Any],
    rng: random_module.Random,
) -> InjectorResult:
    """Replace a datetime field with the wrong format."""
    broken = copy.deepcopy(request)
    datetime_fields = [
        f for f in spec["field_types"]
        if spec["field_types"][f] == "datetime" and f in broken
    ]
    if not datetime_fields:
        # Fallback: inject wrong type instead
        return inject_wrong_field_type(request, headers, spec, rng)

    field = rng.choice(datetime_fields)
    bad_formats = [
        "04/01/2026",
        "2026.04.01",
        "April 1, 2026",
        "1711929600",
        "2026-04-01 09:00",
    ]
    broken[field] = rng.choice(bad_formats)
    return broken, headers, _ground_truth(
        "datetime_format_error", [field], request, headers
    )


# =========================================================================
# 11. wrong_content_type
# =========================================================================

def inject_wrong_content_type(
    request: Dict[str, Any],
    headers: Dict[str, str],
    spec: Dict[str, Any],
    rng: random_module.Random,
) -> InjectorResult:
    """Change Content-Type to an incorrect value."""
    broken_headers = copy.deepcopy(headers)
    wrong_types = [
        "text/plain",
        "application/xml",
        "multipart/form-data",
        "text/html",
        "application/x-www-form-urlencoded",
    ]
    if "Content-Type" in broken_headers:
        broken_headers["Content-Type"] = rng.choice(wrong_types)
    else:
        broken_headers["Content-Type"] = rng.choice(wrong_types)
    return request, broken_headers, _ground_truth(
        "wrong_content_type", ["Content-Type"], request, headers
    )


# =========================================================================
# 12. expired_auth_token
# =========================================================================

def inject_expired_token(
    request: Dict[str, Any],
    headers: Dict[str, str],
    spec: Dict[str, Any],
    rng: random_module.Random,
) -> InjectorResult:
    """Replace the Authorization token with an expired/malformed one."""
    broken_headers = copy.deepcopy(headers)
    bad_tokens = [
        "Bearer expired_token_abc123",
        "Bearer ",
        "Basic dXNlcjpwYXNz",
        "Token invalid",
        "Bearer eyJhbGciOiJub25lIn0.e30.",
    ]
    if "Authorization" in broken_headers:
        broken_headers["Authorization"] = rng.choice(bad_tokens)
        return request, broken_headers, _ground_truth(
            "expired_auth_token", ["Authorization"], request, headers
        )
    # If no auth header in spec, inject wrong content type instead
    return inject_wrong_content_type(request, headers, spec, rng)


# =========================================================================
# 13. wrong_status_code (for response validation / chained scenarios)
# =========================================================================

def inject_wrong_status_code(
    request: Dict[str, Any],
    headers: Dict[str, str],
    spec: Dict[str, Any],
    rng: random_module.Random,
) -> InjectorResult:
    """Record that the wrong HTTP status code would be returned.

    Simulates a server returning an unexpected status code.
    The ground truth stores the wrong code and the expected code.
    """
    correct_status = 200 if spec["http_method"] == "GET" else 201
    wrong_codes = [
        (301, "Moved Permanently - resource redirected"),
        (302, "Found - temporary redirect to different endpoint"),
        (400, "Bad Request - but request is actually valid"),
        (403, "Forbidden - incorrect permissions applied"),
        (404, "Not Found - wrong endpoint routing"),
        (429, "Too Many Requests - rate limit misconfigured"),
        (500, "Internal Server Error - server-side issue"),
        (502, "Bad Gateway - upstream service down"),
        (503, "Service Unavailable - maintenance mode"),
    ]
    wrong_status, description = rng.choice(wrong_codes)
    gt = _ground_truth("wrong_status_code", ["status_code"], request, headers)
    gt["wrong_status"] = wrong_status
    gt["correct_status"] = correct_status
    gt["description"] = description
    return request, headers, gt


# =========================================================================
# 14. redirect_loop
# =========================================================================

def inject_redirect_loop(
    request: Dict[str, Any],
    headers: Dict[str, str],
    spec: Dict[str, Any],
    rng: random_module.Random,
) -> InjectorResult:
    """Simulate a redirect chain issue.

    The agent must identify that the endpoint redirects and provide
    the correct target endpoint.
    """
    redirect_scenarios = [
        {
            "from": spec["endpoint"],
            "to": spec["endpoint"].rstrip("/") + "/v2",
            "reason": "API version upgrade - v1 redirects to v2",
        },
        {
            "from": spec["endpoint"],
            "to": spec["endpoint"].replace("/api/", "/api/v2/"),
            "reason": "Base path migration",
        },
        {
            "from": spec["endpoint"],
            "to": spec["endpoint"] + "?format=json",
            "reason": "Content negotiation redirect",
        },
    ]
    scenario = rng.choice(redirect_scenarios)
    gt = _ground_truth("redirect_loop", ["endpoint"], request, headers)
    gt["redirect_from"] = scenario["from"]
    gt["redirect_to"] = scenario["to"]
    gt["reason"] = scenario["reason"]
    return request, headers, gt


# =========================================================================
# 15. rate_limit_headers
# =========================================================================

def inject_rate_limit_headers(
    request: Dict[str, Any],
    headers: Dict[str, str],
    spec: Dict[str, Any],
    rng: random_module.Random,
) -> InjectorResult:
    """Inject missing or wrong rate limit headers.

    Real APIs require headers like X-RateLimit-Limit, Retry-After.
    The agent must identify the rate limiting issue and provide correct headers.
    """
    broken_headers = copy.deepcopy(headers)
    # Add rate limit headers that indicate the client is being throttled
    broken_headers["X-RateLimit-Remaining"] = "0"
    broken_headers["X-RateLimit-Reset"] = "1712000000"
    broken_headers["Retry-After"] = "60"

    gt = _ground_truth(
        "rate_limit_headers",
        ["X-RateLimit-Remaining", "Retry-After"],
        request, headers,
    )
    gt["issue"] = "Client is rate-limited, must wait or reduce request frequency"
    return request, broken_headers, gt


# =========================================================================
# Registry and helpers
# =========================================================================

# Header-only error types (used by the headers task)
HEADER_ERROR_TYPES = [
    "missing_auth_header",
    "wrong_content_type",
    "expired_auth_token",
]

ERROR_TYPES = [
    "missing_required_field",
    "wrong_field_type",
    "invalid_email_format",
    "missing_auth_header",
    "extra_unknown_field",
    "null_value_in_required",
    "wrong_http_method",
    "malformed_json_value",
    "invalid_enum_value",
    "datetime_format_error",
    "wrong_content_type",
    "expired_auth_token",
    "wrong_status_code",
    "redirect_loop",
    "rate_limit_headers",
]

INJECTOR_MAP = {
    "missing_required_field": inject_missing_required_field,
    "wrong_field_type": inject_wrong_field_type,
    "invalid_email_format": inject_invalid_email_format,
    "missing_auth_header": inject_missing_auth_header,
    "extra_unknown_field": inject_extra_unknown_field,
    "null_value_in_required": inject_null_value_in_required,
    "wrong_http_method": inject_wrong_http_method,
    "malformed_json_value": inject_malformed_json_value,
    "invalid_enum_value": inject_invalid_enum_value,
    "datetime_format_error": inject_datetime_format_error,
    "wrong_content_type": inject_wrong_content_type,
    "expired_auth_token": inject_expired_token,
    "wrong_status_code": inject_wrong_status_code,
    "redirect_loop": inject_redirect_loop,
    "rate_limit_headers": inject_rate_limit_headers,
}


def inject_error(
    error_type: str,
    request: Dict[str, Any],
    headers: Dict[str, str],
    spec: Dict[str, Any],
    rng: random_module.Random,
) -> InjectorResult:
    """Inject a single error of the specified type."""
    injector = INJECTOR_MAP[error_type]
    return injector(request, headers, spec, rng)


# Chain patterns for realistic multi-step debugging scenarios
CHAIN_PATTERNS = [
    # Pattern 1: Auth gate -> body errors
    # Real-world: API returns 401 first, body validation only runs after auth passes
    {
        "name": "auth_gate",
        "gate_types": ["missing_auth_header", "expired_auth_token"],
        "body_pool": None,  # uses all body types
    },
    # Pattern 2: Content-type gate -> type mismatches
    # Real-world: Wrong Content-Type causes parser to misinterpret the body
    {
        "name": "content_type_gate",
        "gate_types": ["wrong_content_type"],
        "body_pool": ["wrong_field_type", "malformed_json_value", "invalid_enum_value"],
    },
    # Pattern 3: Method + endpoint chain
    # Real-world: Wrong method returns 405, then wrong fields for the correct method
    {
        "name": "method_chain",
        "gate_types": ["wrong_http_method"],
        "body_pool": ["missing_required_field", "extra_unknown_field", "null_value_in_required"],
    },
    # Pattern 4: Rate limit + auth
    # Real-world: Rate limited, and when retrying the token has expired
    {
        "name": "rate_limit_chain",
        "gate_types": ["rate_limit_headers"],
        "body_pool": ["expired_auth_token", "missing_required_field"],
    },
    # Pattern 5: Redirect + body errors
    # Real-world: Endpoint moved, client follows redirect but sends wrong body format
    {
        "name": "redirect_chain",
        "gate_types": ["redirect_loop"],
        "body_pool": ["wrong_field_type", "datetime_format_error", "invalid_email_format"],
    },
]


def inject_chained_errors(
    request: Dict[str, Any],
    headers: Dict[str, str],
    spec: Dict[str, Any],
    rng: random_module.Random,
    count: int = 2,
) -> Tuple[Dict[str, Any], Dict[str, str], List[GroundTruth]]:
    """Inject errors in a realistic dependency chain.

    Picks a chain pattern, injects the gate error first, then body errors.
    Ground truths are ordered: gate errors first, body errors second.
    This ordering lets the environment progressively reveal errors.
    """
    broken_req = copy.deepcopy(request)
    broken_hdrs = copy.deepcopy(headers)
    chain: List[GroundTruth] = []

    # Pick a random chain pattern
    pattern = rng.choice(CHAIN_PATTERNS)

    # Inject the gate error
    gate_type = rng.choice(pattern["gate_types"])
    injector = INJECTOR_MAP[gate_type]
    broken_req, broken_hdrs, gt = injector(broken_req, broken_hdrs, spec, rng)
    chain.append(gt)

    # Inject body errors from the pattern's pool (or all body types)
    body_pool = pattern["body_pool"]
    if body_pool is None:
        body_pool = [t for t in ERROR_TYPES if t not in HEADER_ERROR_TYPES
                     and t not in ("wrong_status_code", "redirect_loop", "rate_limit_headers")]

    body_count = max(1, count - 1)
    available = [t for t in body_pool if t in INJECTOR_MAP]
    chosen = rng.sample(available, min(body_count, len(available)))
    for err_type in chosen:
        injector = INJECTOR_MAP[err_type]
        broken_req, broken_hdrs, gt = injector(broken_req, broken_hdrs, spec, rng)
        chain.append(gt)

    return broken_req, broken_hdrs, chain


def inject_multiple_errors(
    request: Dict[str, Any],
    headers: Dict[str, str],
    spec: Dict[str, Any],
    rng: random_module.Random,
    count: int = 2,
) -> Tuple[Dict[str, Any], Dict[str, str], List[GroundTruth]]:
    """Inject multiple errors sequentially. Returns list of ground truths."""
    broken_req = copy.deepcopy(request)
    broken_hdrs = copy.deepcopy(headers)
    all_truths = []

    chosen_types = rng.sample(ERROR_TYPES, min(count, len(ERROR_TYPES)))
    for err_type in chosen_types:
        injector = INJECTOR_MAP[err_type]
        broken_req, broken_hdrs, gt = injector(broken_req, broken_hdrs, spec, rng)
        all_truths.append(gt)

    return broken_req, broken_hdrs, all_truths
