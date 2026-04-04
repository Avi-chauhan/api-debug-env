"""
FastAPI application for the API Debug Environment.

Uses OpenEnv's create_app() to generate all endpoints:
POST /reset, POST /step, GET /state, GET /schema, WS /ws, GET /health
"""

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
