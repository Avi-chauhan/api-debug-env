"""
Baseline inference script for the API Debug Environment.

MANDATORY:
- Must be named inference.py and placed in the root directory.
- Must use OpenAI Client for all LLM calls.
- Must read env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN.
- Must emit [START], [STEP], [END] structured logs to stdout.

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import re
import textwrap
from typing import List, Optional

from openai import OpenAI

from client import APIDebugEnv
from models import APIDebugAction

# Environment variables (mandatory for hackathon evaluation)
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_URL = os.getenv("ENV_URL") or "https://avichauhan-api-debug-env.hf.space"
IMAGE_NAME = os.getenv("IMAGE_NAME")

# Task configuration
TASKS = ["easy", "medium", "hard"]
EPISODES_PER_TASK = 3
MAX_STEPS = {"easy": 3, "medium": 5, "hard": 7}
BENCHMARK_NAME = "api_debug"


# =========================================================================
# Structured logging (exact format required by evaluator)
# =========================================================================

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    done_val = str(done).lower()
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool, steps: int, score: float, rewards: List[float]
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# =========================================================================
# System prompts per task
# =========================================================================

SYSTEM_PROMPTS = {
    "easy": textwrap.dedent("""
        You are an API debugging expert. You receive a broken API request and its specification.
        Your job: identify the error type and the affected fields.

        Respond with ONLY a JSON object in this format:
        {"error_type": "<type>", "affected_fields": ["field1", "field2"]}

        Valid error types:
        missing_required_field, wrong_field_type, invalid_email_format,
        missing_auth_header, extra_unknown_field, null_value_in_required,
        wrong_http_method, malformed_json_value, invalid_enum_value,
        datetime_format_error
    """).strip(),

    "medium": textwrap.dedent("""
        You are an API debugging expert. You receive a broken API request and its specification.
        Your job: fix the request so it matches the spec.

        Respond with ONLY a JSON object in this format:
        {"fixed_request": "<valid JSON string matching the spec>", "fixed_headers": {"Header": "value"}}

        The fixed_request must be a valid JSON string. Include all required fields with correct types.
    """).strip(),

    "hard": textwrap.dedent("""
        You are an API debugging expert. You receive a broken API request with multiple errors.
        Your job: diagnose the errors, fix the request, and explain the fix for a developer.

        Respond with ONLY a JSON object in this format:
        {
            "error_type": "<primary error type>",
            "affected_fields": ["field1"],
            "fixed_request": "<valid JSON string>",
            "fixed_headers": {"Header": "value"},
            "explanation": "Clear explanation of what was wrong and how to fix it."
        }
    """).strip(),
}


# =========================================================================
# Prompt building
# =========================================================================

def build_user_prompt(obs, step_num: int) -> str:
    """Build the user prompt from the observation."""
    parts = [
        f"API: {obs.http_method} {obs.endpoint} ({obs.api_name})",
        f"Error count: {obs.error_count}",
        f"Step {step_num}/{obs.max_steps}",
        f"\nBroken request body:\n{obs.broken_request}",
        f"\nRequest headers: {json.dumps(obs.broken_headers)}",
        f"\nAPI Specification:\n{obs.api_spec}",
    ]
    if obs.feedback:
        parts.append(f"\nFeedback from previous attempt:\n{obs.feedback}")
    return "\n".join(parts)


# =========================================================================
# LLM response parsing
# =========================================================================

def parse_llm_response(text: str) -> dict:
    """Extract a JSON object from the LLM response.

    Handles cases where the LLM wraps JSON in markdown code blocks
    or adds extra text around it.
    """
    if not text:
        return {}

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding any JSON object in the text
    brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


def build_action(data: dict) -> APIDebugAction:
    """Convert parsed JSON dict to APIDebugAction."""
    # Handle fixed_request: if it's a dict, serialize to JSON string
    fixed_req = data.get("fixed_request")
    if isinstance(fixed_req, dict):
        fixed_req = json.dumps(fixed_req)

    return APIDebugAction(
        error_type=data.get("error_type"),
        affected_fields=data.get("affected_fields"),
        fixed_request=fixed_req,
        fixed_headers=data.get("fixed_headers"),
        explanation=data.get("explanation"),
    )


# =========================================================================
# Episode runner
# =========================================================================

async def run_episode(
    env: APIDebugEnv,
    llm_client: OpenAI,
    task: str,
) -> float:
    """Run a single episode for the given task. Returns the final score."""
    log_start(task=task, env=BENCHMARK_NAME, model=MODEL_NAME)

    result = await env.reset(task=task)
    obs = result.observation
    rewards: List[float] = []
    steps_taken = 0

    max_steps = MAX_STEPS[task]

    for step in range(1, max_steps + 1):
        if result.done:
            break

        user_prompt = build_user_prompt(obs, step)

        # Call the LLM
        try:
            completion = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPTS[task]},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=500,
                temperature=0.0,
            )
            llm_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"[DEBUG] LLM request failed: {exc}", flush=True)
            llm_text = ""

        # Parse LLM output into action
        parsed = parse_llm_response(llm_text)
        action = build_action(parsed)

        # Step the environment
        result = await env.step(action)
        obs = result.observation
        reward = result.reward or 0.0
        done = result.done

        rewards.append(reward)
        steps_taken = step

        # Build a short action summary for the log
        action_summary = _action_summary(action, task)
        log_step(step=step, action=action_summary, reward=reward, done=done, error=None)

        if done:
            break

    # Final score is the max reward achieved (environment already tracks best)
    score = max(rewards) if rewards else 0.0
    score = min(max(score, 0.0), 1.0)
    success = score >= 0.5

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


def _action_summary(action: APIDebugAction, task: str) -> str:
    """Short summary of the action for logging."""
    if task == "easy":
        return f"diagnose:{action.error_type or 'none'}"
    elif task == "medium":
        fix_len = len(action.fixed_request or "")
        return f"fix:len={fix_len}"
    else:
        fix_len = len(action.fixed_request or "")
        exp_len = len(action.explanation or "")
        return f"fix:len={fix_len}+explain:len={exp_len}"


# =========================================================================
# Main
# =========================================================================

async def main() -> None:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Connect to environment (via Docker image or direct URL)
    # Use longer timeout for HF Spaces (LLM calls can be slow)
    if IMAGE_NAME:
        env = await APIDebugEnv.from_docker_image(IMAGE_NAME)
    else:
        env = APIDebugEnv(base_url=ENV_URL, message_timeout_s=120.0)

    all_scores: dict = {}

    try:
        for task in TASKS:
            task_scores = []
            for ep in range(EPISODES_PER_TASK):
                try:
                    score = await run_episode(env, llm_client, task)
                except Exception as exc:
                    print(f"[DEBUG] Episode failed: {exc}", flush=True)
                    # Reconnect on WebSocket failure
                    try:
                        await env.close()
                    except Exception:
                        pass
                    env = APIDebugEnv(base_url=ENV_URL, message_timeout_s=120.0)
                    score = 0.0
                task_scores.append(score)
            avg = sum(task_scores) / len(task_scores)
            all_scores[task] = avg

        # Print summary
        print("\n--- Baseline Scores ---", flush=True)
        for task, avg in all_scores.items():
            print(f"  {task}: {avg:.3f}", flush=True)
        overall = sum(all_scores.values()) / len(all_scores)
        print(f"  overall: {overall:.3f}", flush=True)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
