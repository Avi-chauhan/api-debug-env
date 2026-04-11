"""
Response templates for the response validation task.

Each template defines what a correct API response looks like for a given
request type. The environment generates a broken response by injecting
issues (wrong status code, missing fields, wrong types, extra fields).

Response issue types:
- wrong_status_code: Response has incorrect HTTP status code
- missing_response_field: Required field missing from response body
- wrong_response_type: Field present but wrong data type
- extra_response_field: Unexpected field in response (data leak risk)
- inconsistent_error_format: Error response doesn't follow spec format
"""

import copy
import random
from typing import Any, Dict, List, Tuple

# Response issue types the agent must identify
RESPONSE_ISSUE_TYPES = [
    "wrong_status_code",
    "missing_response_field",
    "wrong_response_type",
    "extra_response_field",
    "inconsistent_error_format",
]


# Maps API operation type to expected success response
RESPONSE_TEMPLATES = [
    {
        "name": "Create Resource",
        "success_status": 201,
        "success_body": {
            "id": "res_abc123",
            "status": "created",
            "created_at": "2025-01-15T10:30:00Z",
        },
        "required_response_fields": ["id", "status", "created_at"],
        "field_types": {"id": "string", "status": "string", "created_at": "string"},
        "error_status": 400,
        "error_body": {
            "error": {"code": "VALIDATION_ERROR", "message": "Invalid input", "details": []},
        },
    },
    {
        "name": "List Resources",
        "success_status": 200,
        "success_body": {
            "data": [{"id": "res_1", "name": "Item 1"}, {"id": "res_2", "name": "Item 2"}],
            "total": 2,
            "page": 1,
            "per_page": 20,
        },
        "required_response_fields": ["data", "total", "page", "per_page"],
        "field_types": {"data": "array", "total": "integer", "page": "integer", "per_page": "integer"},
        "error_status": 401,
        "error_body": {
            "error": {"code": "UNAUTHORIZED", "message": "Invalid API key", "details": []},
        },
    },
    {
        "name": "Update Resource",
        "success_status": 200,
        "success_body": {
            "id": "res_abc123",
            "status": "updated",
            "updated_at": "2025-01-15T12:00:00Z",
        },
        "required_response_fields": ["id", "status", "updated_at"],
        "field_types": {"id": "string", "status": "string", "updated_at": "string"},
        "error_status": 404,
        "error_body": {
            "error": {"code": "NOT_FOUND", "message": "Resource not found", "details": []},
        },
    },
    {
        "name": "Delete Resource",
        "success_status": 204,
        "success_body": {},
        "required_response_fields": [],
        "field_types": {},
        "error_status": 403,
        "error_body": {
            "error": {"code": "FORBIDDEN", "message": "Insufficient permissions", "details": []},
        },
    },
    {
        "name": "Batch Operation",
        "success_status": 200,
        "success_body": {
            "processed": 5,
            "failed": 0,
            "results": [
                {"id": "item_1", "status": "success"},
                {"id": "item_2", "status": "success"},
            ],
        },
        "required_response_fields": ["processed", "failed", "results"],
        "field_types": {"processed": "integer", "failed": "integer", "results": "array"},
        "error_status": 422,
        "error_body": {
            "error": {"code": "UNPROCESSABLE", "message": "Batch validation failed", "details": []},
        },
    },
    {
        "name": "Authentication",
        "success_status": 200,
        "success_body": {
            "access_token": "eyJhbGciOiJIUzI1NiJ9.token",
            "token_type": "Bearer",
            "expires_in": 3600,
        },
        "required_response_fields": ["access_token", "token_type", "expires_in"],
        "field_types": {"access_token": "string", "token_type": "string", "expires_in": "integer"},
        "error_status": 401,
        "error_body": {
            "error": {"code": "INVALID_CREDENTIALS", "message": "Bad credentials", "details": []},
        },
    },
    {
        "name": "File Upload",
        "success_status": 201,
        "success_body": {
            "file_id": "file_xyz789",
            "filename": "report.pdf",
            "size_bytes": 1048576,
            "url": "https://cdn.example.com/files/file_xyz789",
        },
        "required_response_fields": ["file_id", "filename", "size_bytes", "url"],
        "field_types": {"file_id": "string", "filename": "string", "size_bytes": "integer", "url": "string"},
        "error_status": 413,
        "error_body": {
            "error": {"code": "PAYLOAD_TOO_LARGE", "message": "File exceeds 10MB limit", "details": []},
        },
    },
    {
        "name": "Search Query",
        "success_status": 200,
        "success_body": {
            "query": "test search",
            "results": [{"id": "doc_1", "score": 0.95, "title": "Test Document"}],
            "total_results": 1,
            "search_time_ms": 42,
        },
        "required_response_fields": ["query", "results", "total_results", "search_time_ms"],
        "field_types": {"query": "string", "results": "array", "total_results": "integer", "search_time_ms": "integer"},
        "error_status": 400,
        "error_body": {
            "error": {"code": "INVALID_QUERY", "message": "Query syntax error", "details": []},
        },
    },
]


def get_random_response_template(rng: random.Random) -> Dict[str, Any]:
    """Pick a random response template."""
    return copy.deepcopy(rng.choice(RESPONSE_TEMPLATES))


def inject_response_issues(
    template: Dict[str, Any],
    rng: random.Random,
    issue_count: int = 1,
) -> Tuple[Dict[str, Any], int, List[Dict[str, str]]]:
    """Inject issues into a response and return (broken_body, broken_status, ground_truths).

    Each ground truth has: issue_type, description, affected_field (if applicable).
    """
    # Decide if we break a success response or an error response
    use_success = rng.random() < 0.6
    if use_success:
        body = copy.deepcopy(template["success_body"])
        status = template["success_status"]
    else:
        body = copy.deepcopy(template["error_body"])
        status = template["error_status"]

    ground_truths: List[Dict[str, str]] = []
    available_issues = list(RESPONSE_ISSUE_TYPES)
    rng.shuffle(available_issues)

    injected = 0
    for issue_type in available_issues:
        if injected >= issue_count:
            break

        if issue_type == "wrong_status_code":
            wrong_codes = [200, 201, 204, 301, 400, 401, 403, 404, 422, 429, 500, 502, 503]
            wrong_codes = [c for c in wrong_codes if c != status]
            old_status = status
            status = rng.choice(wrong_codes)
            ground_truths.append({
                "issue_type": "wrong_status_code",
                "description": f"Expected status {old_status}, got {status}",
                "affected_field": "status_code",
                "correct_value": str(old_status),
            })
            injected += 1

        elif issue_type == "missing_response_field" and template["required_response_fields"]:
            fields = list(template["required_response_fields"])
            rng.shuffle(fields)
            field = fields[0]
            if field in body:
                del body[field]
                ground_truths.append({
                    "issue_type": "missing_response_field",
                    "description": f"Required field '{field}' missing from response",
                    "affected_field": field,
                })
                injected += 1

        elif issue_type == "wrong_response_type" and template["field_types"]:
            typed_fields = [f for f in template["field_types"] if f in body]
            if typed_fields:
                field = rng.choice(typed_fields)
                original_type = template["field_types"][field]
                # Replace with wrong type
                if original_type == "string":
                    body[field] = 12345
                elif original_type == "integer":
                    body[field] = "not_a_number"
                elif original_type == "array":
                    body[field] = "should_be_array"
                else:
                    body[field] = [1, 2, 3]
                ground_truths.append({
                    "issue_type": "wrong_response_type",
                    "description": f"Field '{field}' should be {original_type}",
                    "affected_field": field,
                })
                injected += 1

        elif issue_type == "extra_response_field":
            leak_fields = {
                "internal_id": "int_9f8a2b",
                "debug_trace": "stack trace at line 42",
                "db_query": "SELECT * FROM users WHERE id=123",
                "server_ip": "10.0.0.42",
                "session_token": "sess_leaked_abc",
            }
            field, value = rng.choice(list(leak_fields.items()))
            body[field] = value
            ground_truths.append({
                "issue_type": "extra_response_field",
                "description": f"Unexpected field '{field}' in response (potential data leak)",
                "affected_field": field,
            })
            injected += 1

        elif issue_type == "inconsistent_error_format" and not use_success:
            # Break the error format -- flatten it or use wrong keys
            variants = [
                {"msg": body.get("error", {}).get("message", "error"), "err_code": "UNKNOWN"},
                {"error_message": "Something went wrong", "status": "error"},
                {"errors": [{"msg": "bad request"}]},
            ]
            body = rng.choice(variants)
            ground_truths.append({
                "issue_type": "inconsistent_error_format",
                "description": "Error response doesn't follow standard format: {error: {code, message, details}}",
                "affected_field": "error",
            })
            injected += 1

    return body, status, ground_truths
