"""
Client SDK for the API Debug Environment.

Implements the three required abstract methods from EnvClient:
- _step_payload: converts APIDebugAction to JSON dict
- _parse_result: converts server response to StepResult
- _parse_state: converts server state to State object

Usage:
    async with APIDebugEnv(base_url="http://localhost:8000") as env:
        result = await env.reset(task="easy")
        result = await env.step(APIDebugAction(error_type="missing_required_field"))
"""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import APIDebugAction, APIDebugObservation


class APIDebugEnv(EnvClient[APIDebugAction, APIDebugObservation, State]):

    def _step_payload(self, action: APIDebugAction) -> Dict[str, Any]:
        """Convert action to JSON dict, including only non-None fields."""
        payload = {}
        if action.error_type is not None:
            payload["error_type"] = action.error_type
        if action.affected_fields is not None:
            payload["affected_fields"] = action.affected_fields
        if action.fixed_request is not None:
            payload["fixed_request"] = action.fixed_request
        if action.fixed_headers is not None:
            payload["fixed_headers"] = action.fixed_headers
        if action.explanation is not None:
            payload["explanation"] = action.explanation
        return payload

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[APIDebugObservation]:
        """Convert server JSON response to StepResult.

        The server sends:
        {
            "observation": { ...fields except reward/done/metadata... },
            "reward": float,
            "done": bool,
        }
        """
        obs_data = payload.get("observation", {})
        observation = APIDebugObservation(
            task=obs_data.get("task", "easy"),
            api_name=obs_data.get("api_name", ""),
            http_method=obs_data.get("http_method", "POST"),
            endpoint=obs_data.get("endpoint", ""),
            broken_request=obs_data.get("broken_request", ""),
            broken_headers=obs_data.get("broken_headers", {}),
            api_spec=obs_data.get("api_spec", ""),
            error_count=obs_data.get("error_count", 1),
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", 3),
            feedback=obs_data.get("feedback", ""),
            message=obs_data.get("message", ""),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """Convert server state JSON to State object."""
        return State(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
        )
