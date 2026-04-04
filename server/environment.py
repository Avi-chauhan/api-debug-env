"""
Core environment for the API Debug Environment.

Implements the OpenEnv Environment interface with:
- 3 task difficulty levels (easy, medium, hard)
- Multi-turn episodes with structured feedback
- Deterministic grading for easy/medium, LLM-as-judge for hard
- Step reward decay to encourage efficient debugging
"""

import copy
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import APIDebugAction, APIDebugObservation
except ImportError:
    from models import APIDebugAction, APIDebugObservation

from .api_specs import get_random_spec
from .error_injectors import (
    ERROR_TYPES,
    inject_error,
    inject_multiple_errors,
)
from .validators import (
    validate_field_type,
    validate_headers_against_spec,
    validate_request_against_spec,
)


# Task configuration: max steps and error count per difficulty
TASK_CONFIG = {
    "easy": {"max_steps": 3, "error_count": 1},
    "medium": {"max_steps": 5, "error_count": 1},
    "hard": {"max_steps": 7, "min_errors": 2, "max_errors": 3},
}


class APIDebugEnvironment(Environment):
    """API Contract Validation environment.

    An LLM agent receives a broken API request and must:
    - Easy: Identify the error type and affected fields
    - Medium: Fix the request to match the API spec
    - Hard: Fix the request and explain the fix for developers

    Each episode allows multiple attempts. Perfect answers on early
    steps get full reward. Later steps get decayed reward.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.task = "easy"
        self.spec: Dict[str, Any] = {}
        self.broken_request: Dict[str, Any] = {}
        self.broken_headers: Dict[str, str] = {}
        self.ground_truths: List[Dict[str, Any]] = []
        self.current_step = 0
        self.max_steps = 3
        self.episode_done = False
        self.best_reward = 0.0
        self.rng = random.Random()
        # For wrong_http_method error: the method shown to the agent
        self.shown_http_method = ""

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: str = "easy",
        **kwargs,
    ) -> APIDebugObservation:
        """Start a new debugging episode.

        Args:
            seed: Random seed for reproducible episodes.
            episode_id: Custom episode identifier.
            task: Difficulty level (easy, medium, hard).
        """
        # Initialize RNG
        if seed is not None:
            self.rng = random.Random(seed)
        else:
            self.rng = random.Random()

        # Validate task
        self.task = task if task in TASK_CONFIG else "easy"
        config = TASK_CONFIG[self.task]
        self.max_steps = config["max_steps"]
        self.current_step = 0
        self.episode_done = False
        self.best_reward = 0.0

        # Fresh state
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        # Pick random spec and build valid request
        self.spec = copy.deepcopy(get_random_spec(self.rng))
        valid_request = copy.deepcopy(self.spec["valid_example"])
        valid_headers = copy.deepcopy(self.spec["required_headers"])

        # Inject errors based on difficulty
        if self.task == "hard":
            error_count = self.rng.randint(config["min_errors"], config["max_errors"])
            self.broken_request, self.broken_headers, self.ground_truths = (
                inject_multiple_errors(
                    valid_request, valid_headers, self.spec, self.rng, error_count
                )
            )
        else:
            error_type = self.rng.choice(ERROR_TYPES)
            self.broken_request, self.broken_headers, gt = inject_error(
                error_type, valid_request, valid_headers, self.spec, self.rng
            )
            self.ground_truths = [gt]

        # Handle wrong_http_method: show the wrong method to the agent
        self.shown_http_method = self.spec["http_method"]
        for gt in self.ground_truths:
            if gt["error_type"] == "wrong_http_method":
                self.shown_http_method = gt.get("wrong_method", self.spec["http_method"])
                break

        error_count = len(self.ground_truths)
        return APIDebugObservation(
            task=self.task,
            api_name=self.spec["api_name"],
            http_method=self.shown_http_method,
            endpoint=self.spec["endpoint"],
            broken_request=json.dumps(self.broken_request, indent=2),
            broken_headers=self.broken_headers,
            api_spec=self._build_spec_string(),
            error_count=error_count,
            step_number=0,
            max_steps=self.max_steps,
            feedback="",
            message=(
                f"Debug this {self.shown_http_method} {self.spec['endpoint']} request. "
                f"It contains {error_count} error(s). "
                f"You have {self.max_steps} steps."
            ),
            done=False,
            reward=0.0,
        )

    def step(
        self,
        action: APIDebugAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> APIDebugObservation:
        """Process the agent's debugging attempt.

        The agent can submit a partial or complete response.
        The grader evaluates whatever fields are present.
        """
        self.current_step += 1
        self._state.step_count = self.current_step

        if self.episode_done:
            return self._make_observation(
                feedback="Episode already ended.",
                reward=0.0,
                done=True,
            )

        # Grade based on task type
        if self.task == "easy":
            raw_score, feedback = self._grade_easy(action)
        elif self.task == "medium":
            raw_score, feedback = self._grade_medium(action)
        else:
            raw_score, feedback = self._grade_hard(action)

        # Apply step decay: step 1 = 1.0x, step 2 = 0.9x, etc. Floor at 0.3x
        step_multiplier = max(1.0 - 0.1 * (self.current_step - 1), 0.3)
        reward = round(raw_score * step_multiplier, 4)

        # Track best reward across all steps
        self.best_reward = max(self.best_reward, reward)

        # Episode ends if score is near-perfect or out of steps
        near_perfect = raw_score >= 0.95
        out_of_steps = self.current_step >= self.max_steps
        done = near_perfect or out_of_steps

        if done:
            self.episode_done = True
            # Return best reward achieved during the episode
            reward = self.best_reward

        return self._make_observation(
            feedback=feedback,
            reward=reward,
            done=done,
        )

    @property
    def state(self) -> State:
        return self._state

    # =====================================================================
    # Grading methods
    # =====================================================================

    def _grade_easy(self, action: APIDebugAction) -> Tuple[float, str]:
        """Grade error identification. Fully deterministic.

        Scoring: 0.6 for correct error_type + 0.4 for correct affected_fields.
        Fields use Jaccard similarity for partial credit.
        """
        score = 0.0
        parts = []

        # Collect all ground truth error types and affected fields
        gt_types = {gt["error_type"] for gt in self.ground_truths}
        gt_fields: set = set()
        for gt in self.ground_truths:
            gt_fields.update(gt.get("affected_fields", []))

        # Check error type (0.6 weight)
        if action.error_type and action.error_type in gt_types:
            score += 0.6
            parts.append("error_type: CORRECT")
        else:
            given = action.error_type or "(none)"
            parts.append(f"error_type: INCORRECT (you said '{given}')")

        # Check affected fields using Jaccard similarity (0.4 weight)
        agent_fields = set(action.affected_fields or [])
        if gt_fields and agent_fields:
            intersection = gt_fields & agent_fields
            union = gt_fields | agent_fields
            jaccard = len(intersection) / len(union) if union else 0.0
            score += 0.4 * jaccard
            parts.append(
                f"affected_fields: {len(intersection)}/{len(gt_fields)} correct, "
                f"{len(agent_fields - gt_fields)} extra"
            )
        elif not agent_fields:
            parts.append("affected_fields: MISSING (none provided)")
        else:
            parts.append("affected_fields: INCORRECT (0 matches)")

        return round(score, 4), "; ".join(parts)

    def _grade_medium(self, action: APIDebugAction) -> Tuple[float, str]:
        """Grade request fix. Fully deterministic per-field validation.

        Validates the fixed request against the spec: required fields present,
        field types correct, headers present. Each check is equally weighted.
        """
        if not action.fixed_request:
            return 0.0, "No fixed_request provided."

        try:
            fixed = json.loads(action.fixed_request)
        except (json.JSONDecodeError, TypeError):
            return 0.0, "fixed_request is not valid JSON."

        if not isinstance(fixed, dict):
            return 0.0, "fixed_request must be a JSON object."

        # Validate request body against spec
        body_score, body_feedback = validate_request_against_spec(fixed, self.spec)

        # Validate headers if provided
        header_score = 0.0
        header_feedback = ""
        has_header_errors = any(
            gt["error_type"] == "missing_auth_header" for gt in self.ground_truths
        )

        if has_header_errors and action.fixed_headers:
            header_score, header_feedback = validate_headers_against_spec(
                action.fixed_headers, self.spec
            )
            # Blend: 80% body + 20% headers when header errors exist
            total_score = 0.8 * body_score + 0.2 * header_score
            feedback = body_feedback + "\n" + header_feedback
        elif has_header_errors and not action.fixed_headers:
            feedback = body_feedback + "\nHeaders: NOT PROVIDED (header fix needed)"
            total_score = 0.8 * body_score
        else:
            total_score = body_score
            feedback = body_feedback

        return round(total_score, 4), feedback

    def _grade_hard(self, action: APIDebugAction) -> Tuple[float, str]:
        """Grade fix + explanation. 70% deterministic fix, 30% explanation.

        The explanation is scored by LLM-as-judge if available,
        with a heuristic fallback if the LLM is not reachable.
        """
        # Deterministic fix scoring (same as medium)
        fix_score, fix_feedback = self._grade_medium(action)

        # Explanation scoring
        explain_score = 0.0
        explain_feedback = "No explanation provided."

        if action.explanation and len(action.explanation.strip()) > 10:
            explain_score = self._score_explanation(action.explanation)
            explain_feedback = f"Explanation quality: {explain_score:.2f}/1.0"

        total = 0.7 * fix_score + 0.3 * explain_score
        feedback = (
            f"Fix score: {fix_score:.2f} (70% weight)\n"
            f"{fix_feedback}\n"
            f"{explain_feedback}"
        )
        return round(total, 4), feedback

    def _score_explanation(self, explanation: str) -> float:
        """Score an explanation using LLM-as-judge with heuristic fallback.

        Tries to call the LLM via the HF router. If that fails for any
        reason, falls back to a keyword + length heuristic.
        """
        # Try LLM-as-judge first
        try:
            llm_score = self._llm_judge_explanation(explanation)
            if llm_score is not None:
                return llm_score
        except Exception:
            pass

        # Heuristic fallback
        return self._heuristic_score_explanation(explanation)

    def _llm_judge_explanation(self, explanation: str) -> Optional[float]:
        """Call LLM to score the explanation. Returns None if unavailable."""
        api_base = os.getenv("API_BASE_URL")
        api_key = os.getenv("HF_TOKEN")
        model = os.getenv("MODEL_NAME")

        if not all([api_base, api_key, model]):
            return None

        from openai import OpenAI

        client = OpenAI(base_url=api_base, api_key=api_key)

        error_types = [gt["error_type"] for gt in self.ground_truths]
        prompt = (
            "Rate this API debugging explanation on a 0.0 to 1.0 scale.\n\n"
            "Criteria:\n"
            "- Correctly identifies root cause (0 to 0.4)\n"
            "- Provides actionable fix guidance (0 to 0.3)\n"
            "- Includes prevention advice for developers (0 to 0.3)\n\n"
            f"API: {self.spec['api_name']} {self.spec['endpoint']}\n"
            f"Errors present: {json.dumps(error_types)}\n"
            f"Explanation: {explanation}\n\n"
            'Return ONLY a JSON object: {"score": 0.0}'
        )

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.0,
        )
        text = response.choices[0].message.content or ""

        # Parse score from response
        result = json.loads(text)
        raw_score = float(result["score"])
        return max(0.0, min(1.0, raw_score))

    def _heuristic_score_explanation(self, explanation: str) -> float:
        """Simple heuristic scoring based on length and keyword presence.

        This is the fallback when LLM-as-judge is not available.
        Not perfect, but ensures non-zero scores for reasonable explanations.
        """
        keywords = [
            "because", "should", "instead", "required", "missing",
            "type", "format", "expected", "invalid", "correct",
            "field", "header", "value", "fix", "error",
        ]
        keyword_hits = sum(1 for k in keywords if k in explanation.lower())
        keyword_score = min(keyword_hits / 6.0, 1.0)

        # Length score: reward explanations between 50 and 500 chars
        length = len(explanation.strip())
        if length < 20:
            length_score = 0.1
        elif length < 50:
            length_score = 0.3
        elif length <= 500:
            length_score = 0.6
        else:
            length_score = 0.5  # Slightly penalize very long explanations

        return round(0.5 * keyword_score + 0.5 * length_score, 2)

    # =====================================================================
    # Helpers
    # =====================================================================

    def _build_spec_string(self) -> str:
        """Build a JSON string of the spec info the agent needs to see."""
        visible_spec = {
            "required_fields": self.spec["required_fields"],
            "optional_fields": self.spec.get("optional_fields", []),
            "field_types": self.spec["field_types"],
            "required_headers": list(self.spec.get("required_headers", {}).keys()),
        }
        return json.dumps(visible_spec, indent=2)

    def _make_observation(
        self,
        feedback: str,
        reward: float,
        done: bool,
    ) -> APIDebugObservation:
        """Build an observation with the current episode state."""
        if done and not feedback:
            msg = "Episode complete."
        elif done:
            msg = f"Episode complete. Final reward: {reward:.2f}"
        else:
            remaining = self.max_steps - self.current_step
            msg = f"{remaining} step(s) remaining. Use the feedback to improve."

        return APIDebugObservation(
            task=self.task,
            api_name=self.spec.get("api_name", ""),
            http_method=self.shown_http_method,
            endpoint=self.spec.get("endpoint", ""),
            broken_request=json.dumps(self.broken_request, indent=2),
            broken_headers=self.broken_headers,
            api_spec=self._build_spec_string(),
            error_count=len(self.ground_truths),
            step_number=self.current_step,
            max_steps=self.max_steps,
            feedback=feedback,
            message=msg,
            done=done,
            reward=reward,
        )
