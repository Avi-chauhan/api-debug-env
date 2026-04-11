import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

"""
GRPO Training on API Debug Environment
=======================================
Trains a small LLM (Qwen 0.5B) to debug malformed API requests using
reward signals from the live HuggingFace Space environment.

Supports curriculum learning: starts on easy task, promotes to classify
and medium as the agent improves.

Run on Colab (free T4 GPU):
    pip install -r training/requirements.txt
    python training/train.py
"""

import json
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openenv import generate_rollout_completions

from client import APIDebugEnv
from models import APIDebugAction

# -- GPU check ----------------------------------------------------------------
print(f"GPU available   : {torch.cuda.is_available()}")
print(f"GPU name        : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None (CPU)'}")

has_gpu       = torch.cuda.is_available()
supports_bf16 = has_gpu and torch.cuda.is_bf16_supported()

# -- Config -------------------------------------------------------------------
MODEL_ID    = "Qwen/Qwen2.5-0.5B-Instruct"
ENV_URL     = "https://avichauhan-api-debug-env.hf.space"
MAX_TURNS   = 3   # easy task: 3 steps max
NUM_SAMPLES = 64

# -- Curriculum state ---------------------------------------------------------
# Tracks which task the agent is currently training on.
# Promotes when rolling average reward exceeds threshold.
CURRICULUM = {
    "easy":     {"next": "classify", "threshold": 0.7, "max_turns": 3},
    "classify": {"next": "medium",   "threshold": 0.6, "max_turns": 4},
    "medium":   {"next": "headers",  "threshold": 0.6, "max_turns": 5},
    "headers":  {"next": "response", "threshold": 0.5, "max_turns": 4},
    "response": {"next": "hard",     "threshold": 0.5, "max_turns": 4},
    "hard":     {"next": None,       "threshold": None, "max_turns": 7},
}
current_task = "easy"
recent_rewards: list[float] = []
WINDOW_SIZE = 10

SYSTEM_PROMPT = """You are an API debugging expert. You receive a broken API request and its specification.
Your job: identify the error type and the affected fields.

Respond with ONLY a JSON object in this format:
{"error_type": "<type>", "affected_fields": ["field1", "field2"]}

Valid error types:
missing_required_field, wrong_field_type, invalid_email_format,
missing_auth_header, extra_unknown_field, null_value_in_required,
wrong_http_method, malformed_json_value, invalid_enum_value,
datetime_format_error, wrong_content_type, expired_auth_token"""

CLASSIFY_PROMPT = """You are an API debugging expert. This request has MULTIPLE errors.
Identify ALL error types and ALL affected fields.

Respond with ONLY a JSON object:
{"error_types": ["type1", "type2"], "affected_fields": ["field1", "field2"]}"""

MEDIUM_PROMPT = """You are an API debugging expert. Fix the broken request to match the API spec.

Respond with ONLY a JSON object:
{"fixed_request": {"field": "value"}, "fixed_headers": {"Header": "value"}}"""

HEADERS_PROMPT = """You are an API debugging expert. This request has ONLY header-level errors.
Identify the error type and fix the headers to match the API spec.

Respond with ONLY a JSON object:
{"error_type": "<type>", "fixed_headers": {"Header-Name": "correct_value"}}

Common header error types: wrong_content_type, expired_auth_token, missing_auth_header"""

RESPONSE_PROMPT = """You are an API response validation expert. You receive an API request, its spec, and the server response.
Identify issues in the response: wrong status codes, missing fields, wrong types, extra fields, inconsistent error format.

Respond with ONLY a JSON object:
{"response_issues": ["issue_type1"], "affected_fields": ["field1"], "expected_status_code": 200}

Valid issue types: wrong_status_code, missing_response_field, wrong_response_type, extra_response_field, inconsistent_error_format"""

HARD_PROMPT = """You are an API debugging expert. This request has MULTIPLE errors across headers and body.
Some errors are chained -- fixing one may reveal others. Fix everything and explain your reasoning.

Respond with ONLY a JSON object:
{"fixed_request": {"field": "value"}, "fixed_headers": {"Header": "value"}, "explanation": "why each fix was needed"}"""

TASK_PROMPTS = {
    "easy": SYSTEM_PROMPT,
    "classify": CLASSIFY_PROMPT,
    "medium": MEDIUM_PROMPT,
    "headers": HEADERS_PROMPT,
    "response": RESPONSE_PROMPT,
    "hard": HARD_PROMPT,
}

# -- Environment client -------------------------------------------------------
print(f"Connecting to environment: {ENV_URL}")
env_client = APIDebugEnv(base_url=ENV_URL)


# -- JSON parser (reused from inference.py) -----------------------------------
def parse_llm_response(text: str) -> dict:
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            pass
    brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def build_action(data) -> APIDebugAction:
    if not isinstance(data, dict):
        return APIDebugAction()
    fixed_req = data.get("fixed_request")
    if isinstance(fixed_req, dict):
        fixed_req = json.dumps(fixed_req)
    return APIDebugAction(
        error_type=data.get("error_type"),
        error_types=data.get("error_types"),
        affected_fields=data.get("affected_fields"),
        fixed_request=fixed_req,
        fixed_headers=data.get("fixed_headers"),
        response_issues=data.get("response_issues"),
        expected_status_code=data.get("expected_status_code"),
    )


# -- Curriculum learning ------------------------------------------------------
def maybe_promote():
    """Check if agent should be promoted to next difficulty."""
    global current_task
    config = CURRICULUM[current_task]
    if config["next"] is None or config["threshold"] is None:
        return
    if len(recent_rewards) < WINDOW_SIZE:
        return
    avg = sum(recent_rewards[-WINDOW_SIZE:]) / WINDOW_SIZE
    if avg >= config["threshold"]:
        old_task = current_task
        current_task = config["next"]
        recent_rewards.clear()
        print(f"[CURRICULUM] Promoted: {old_task} -> {current_task} (avg_reward={avg:.3f})")


# -- Rollout function ---------------------------------------------------------
def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict:
    tokenizer = trainer.processing_class
    all_prompt_ids     = []
    all_completion_ids = []
    all_logprobs       = []
    all_rewards        = []

    task = current_task
    max_turns = CURRICULUM[task]["max_turns"]
    system_prompt = TASK_PROMPTS[task]

    for base_prompt in prompts:
        with env_client.sync() as env:
            obs = env.reset(task=task)
            episode_reward     = 0.0
            episode_prompt_ids = []
            episode_comp_ids   = []
            episode_logprobs   = []

            for turn in range(max_turns):
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": (
                        f"{base_prompt}\n\n"
                        f"API: {obs.observation.http_method} {obs.observation.endpoint} "
                        f"({obs.observation.api_name})\n"
                        f"Error count: {obs.observation.error_count}\n"
                        f"Step {turn + 1}/{max_turns}\n\n"
                        f"Broken request:\n{obs.observation.broken_request}\n\n"
                        f"Headers: {json.dumps(obs.observation.broken_headers)}\n\n"
                        f"API Spec:\n{obs.observation.api_spec}\n"
                        + (f"\nFeedback:\n{obs.observation.feedback}" if obs.observation.feedback else "")
                    )},
                ]
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=False,
                )

                outputs = generate_rollout_completions(trainer, [prompt_text])[0]
                completion_text = tokenizer.decode(
                    outputs["completion_ids"], skip_special_tokens=True
                ).strip()

                episode_prompt_ids.extend(outputs["prompt_ids"])
                episode_comp_ids.extend(outputs["completion_ids"])
                episode_logprobs.extend(outputs["logprobs"])

                # Parse LLM output into action
                parsed = parse_llm_response(completion_text)
                action = build_action(parsed)
                obs = env.step(action)
                episode_reward = float(obs.reward or 0.0)

                if obs.done:
                    break

            all_prompt_ids.append(episode_prompt_ids)
            all_completion_ids.append(episode_comp_ids)
            all_logprobs.append(episode_logprobs)
            all_rewards.append(episode_reward)

            # Track for curriculum
            recent_rewards.append(episode_reward)

    # Check if agent should be promoted
    maybe_promote()

    return {
        "prompt_ids":     all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs":       all_logprobs,
        "env_reward":     all_rewards,
    }


# -- Reward function ----------------------------------------------------------
def reward_from_env(completions, **kwargs):
    env_rewards = kwargs.get("env_reward", [])
    return [float(r) for r in env_rewards] if env_rewards else [0.0] * len(completions)


# -- Dataset ------------------------------------------------------------------
dataset = Dataset.from_dict({
    "prompt": ["Debug this broken API request."] * NUM_SAMPLES
})

# -- Trainer ------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, attn_implementation="eager")

grpo_args = GRPOConfig(
    use_vllm=True,
    vllm_mode="colocate",
    num_train_epochs=1,
    num_generations=2,
    max_completion_length=128,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-6,
    output_dir="./outputs/api-debug-grpo",
    logging_steps=1,
    report_to="none",
    bf16=supports_bf16,
    fp16=has_gpu and not supports_bf16,
    gradient_checkpointing=True,
    vllm_gpu_memory_utilization=0.3,
    dataloader_pin_memory=False,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=reward_from_env,
    train_dataset=dataset,
    rollout_func=rollout_func,
    args=grpo_args,
)

if __name__ == "__main__":
    print("Starting GRPO training on API Debug Environment...")
    print(f"Model      : {MODEL_ID}")
    print(f"Environment: {ENV_URL}")
    print(f"Episodes   : {NUM_SAMPLES}")
    print(f"Task       : {current_task} (with curriculum learning)")
    print(f"bf16       : {supports_bf16}")
    print(f"fp16       : {has_gpu and not supports_bf16}")
    trainer.train()
    print(f"Training complete! Final task: {current_task}")
    print("Model saved to ./outputs/api-debug-grpo")
