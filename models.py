"""
Pydantic models for the API Debug Environment.

APIDebugAction: What the agent sends each step.
APIDebugObservation: What the environment returns each step.

All Action fields are Optional so the agent can submit only what it has.
For example, on an easy task the agent only needs error_type and affected_fields.
On medium, it needs fixed_request. On hard, it needs everything plus explanation.
"""

from typing import Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class APIDebugAction(Action):
    """Agent's response at each step of the debugging episode."""

    error_type: Optional[str] = Field(
        default=None,
        description="Diagnosed error type, e.g. 'missing_required_field'"
    )
    error_types: Optional[List[str]] = Field(
        default=None,
        description="All diagnosed error types (for classify task with multiple errors)"
    )
    affected_fields: Optional[List[str]] = Field(
        default=None,
        description="List of field names affected by the error"
    )
    fixed_request: Optional[str] = Field(
        default=None,
        description="JSON string of the corrected request body"
    )
    fixed_headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="Corrected HTTP headers if applicable"
    )
    explanation: Optional[str] = Field(
        default=None,
        description="Developer-facing explanation of the fix (hard task only)"
    )
    response_issues: Optional[List[str]] = Field(
        default=None,
        description="Issues found in the API response (response task only)"
    )
    expected_status_code: Optional[int] = Field(
        default=None,
        description="Correct HTTP status code for the response (response task only)"
    )


class APIDebugObservation(Observation):
    """Environment's response at each step.

    Inherits done, reward, and metadata from Observation base class.
    """

    task: str = Field(
        default="easy",
        description="Current task: easy, classify, medium, headers, hard, response"
    )
    api_name: str = Field(
        default="",
        description="Name of the API being debugged"
    )
    http_method: str = Field(
        default="POST",
        description="HTTP method of the broken request"
    )
    endpoint: str = Field(
        default="",
        description="API endpoint path"
    )
    broken_request: str = Field(
        default="",
        description="JSON string of the malformed request body"
    )
    broken_headers: Dict[str, str] = Field(
        default_factory=dict,
        description="HTTP headers sent with the broken request"
    )
    api_spec: str = Field(
        default="",
        description="JSON string of the API specification"
    )
    error_count: int = Field(
        default=1,
        description="Number of errors injected in this episode"
    )
    step_number: int = Field(
        default=0,
        description="Current step in this episode"
    )
    max_steps: int = Field(
        default=3,
        description="Maximum steps allowed for this task"
    )
    response_body: str = Field(
        default="",
        description="JSON string of the API response to validate (response task only)"
    )
    response_status_code: int = Field(
        default=0,
        description="HTTP status code of the response (response task only)"
    )
    feedback: str = Field(
        default="",
        description="Structured validation feedback from the last action"
    )
    message: str = Field(
        default="",
        description="Human-readable status message"
    )
