"""
FastAPI application for the API Debug Environment.

Uses OpenEnv's create_app() to generate all endpoints:
POST /reset, POST /step, GET /state, GET /schema, WS /ws, GET /health
"""

from fastapi.responses import JSONResponse
from openenv.core.env_server.http_server import create_app

try:
    from ..models import APIDebugAction, APIDebugObservation
    from .environment import APIDebugEnvironment
except ImportError:
    from models import APIDebugAction, APIDebugObservation
    from server.environment import APIDebugEnvironment

app = create_app(
    APIDebugEnvironment,
    APIDebugAction,
    APIDebugObservation,
    env_name="api_debug",
    max_concurrent_envs=10,
)


@app.get("/tasks")
def list_tasks():
    """List all available tasks, their configuration, and supported error types."""
    return JSONResponse({
        "tasks": [
            {
                "name": "easy",
                "max_steps": 3,
                "error_count": 1,
                "grading": "deterministic",
                "description": "Identify the error type and affected fields",
            },
            {
                "name": "medium",
                "max_steps": 5,
                "error_count": 1,
                "grading": "deterministic",
                "description": "Fix the broken request to match the API spec",
            },
            {
                "name": "hard",
                "max_steps": 7,
                "error_count": "2-3",
                "grading": "70% deterministic + 30% LLM-as-judge",
                "description": "Fix the request and explain the fix for developers",
            },
        ],
        "error_types": [
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
        ],
        "api_spec_count": 30,
    })


def main():
    """Run the server directly."""
    import sys
    import uvicorn

    port = 8000
    if len(sys.argv) > 1:
        port = int(sys.argv[1])

    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
