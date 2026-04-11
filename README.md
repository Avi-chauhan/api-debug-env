---
title: API Debug Environment
emoji: "🔧"
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - rl-environment
  - api-debugging
---

# API Debug Environment

An OpenEnv reinforcement learning environment where LLM agents learn to debug malformed API requests. The agent receives a broken request and its API specification, then must diagnose the error, fix the request, and explain the fix.

Built for the Meta PyTorch OpenEnv Hackathon x Scaler School of Technology 2026.

## Infinite Unique Scenarios

Unlike fixed-fixture environments where every episode presents the same scenario, this environment generates a unique broken request each episode. With 45 API spec templates across 9 domains and 15 error injection functions (including chained multi-step errors), the environment produces tens of thousands of distinct training scenarios. An agent cannot memorize answers after one run - it must learn a generalizable debugging strategy. This is critical for real RL training value: agents learn transferable skills rather than dataset-specific shortcuts.

## Why This Domain

Developers spend significant time debugging API contract mismatches. Research from Calendar Gym shows that malformed tool arguments caused more than half of agent failures. This environment trains agents to identify and fix these errors systematically.

## How It Works

1. On `reset()`, the environment picks a random API spec from 45 templates and injects 1-3 errors
2. The agent receives the broken request, headers, and the API specification
3. The agent submits a fix attempt via `step()`
4. The environment grades the attempt and returns structured feedback
5. The agent can iterate (multi-turn) using the feedback to improve

Each episode allows multiple attempts. Perfect answers on early steps earn full reward. Later steps get decayed reward, encouraging efficient debugging.

## Tasks

| Task | Difficulty | Max Steps | Errors | Grading |
|------|-----------|-----------|--------|---------|
| easy | Identify error type and affected fields | 3 | 1 | Deterministic: 0.6 x type_match + 0.4 x fields_match |
| classify | Identify ALL error types across multiple errors | 4 | 2-3 | Deterministic: 0.6 x Jaccard(types) + 0.4 x Jaccard(fields) |
| medium | Fix the broken request | 5 | 1 | Deterministic: per-field validation against spec |
| headers | Fix header-level errors (auth, content-type, tokens) | 4 | 1 | Deterministic: 0.7 x header_fix + 0.3 x type_match |
| response | Validate API response for issues | 4 | 1-2 | Deterministic: 0.5 x Jaccard(issues) + 0.3 x Jaccard(fields) + 0.2 x status_code |
| hard | Fix request + explain for developers | 7 | 2-3 | 70% deterministic fix + 30% LLM-as-judge explanation (gpt-4o-mini) |

## Error Types

| Error Type | Description | Example |
|-----------|-------------|---------|
| missing_required_field | A required field is removed | `email` missing from Create User |
| wrong_field_type | Field has wrong type | `amount` sent as `"2500"` instead of `2500` |
| invalid_email_format | Email field is malformed | `user@` instead of `user@example.com` |
| missing_auth_header | Authorization header removed | No `Bearer` token |
| extra_unknown_field | Unknown field added | `debug_mode: true` in production request |
| null_value_in_required | Required field set to null | `name: null` |
| wrong_http_method | Wrong HTTP method used | `GET` instead of `POST` |
| malformed_json_value | Corrupted field value | `{broken` as a value |
| invalid_enum_value | Value not in allowed list | `currency: "xyz"` |
| datetime_format_error | Wrong date format | `04/01/2026` instead of ISO 8601 |
| wrong_content_type | Wrong Content-Type header | `text/plain` instead of `application/json` |
| expired_auth_token | Expired or invalid auth token | `Bearer expired_token_2024` |
| wrong_status_code | Wrong HTTP status code in response | `200` instead of `201` for resource creation |
| redirect_loop | Redirect configuration error | Version upgrade redirect loop |
| rate_limit_headers | Rate limit exceeded headers | `X-RateLimit-Remaining: 0` |

## API Spec Domains (45 templates)

| Domain | Count | Examples |
|--------|-------|---------|
| Payment (Stripe-like) | 5 | Create Customer, Create Charge, Process Refund |
| User Management | 5 | Create User, Update Profile, Reset Password |
| Content (GitHub-like) | 5 | Create Repository, Create Issue, Merge PR |
| Messaging (Twilio-like) | 5 | Send SMS, Send Email, Create Webhook |
| E-Commerce | 5 | Create Order, Process Payment, Create Shipping Label |
| Calendar and Auth | 5 | Create Event, OAuth Token, Create API Key |
| Analytics/Monitoring | 5 | Create Dashboard, Add Metric, Create Alert |
| DevOps/Infrastructure | 5 | Create Deployment, Scale Service, Create DNS Record |
| AI/ML APIs | 5 | Submit Inference, Create Fine-tune Job, Upload Dataset |

## Action Space

The agent sends an `APIDebugAction` with these fields (all optional, submit what you have):

| Field | Type | Used In | Description |
|-------|------|---------|-------------|
| error_type | string | easy, headers, hard | Diagnosed error type |
| error_types | list[string] | classify | All diagnosed error types (multi-error) |
| affected_fields | list[string] | easy, classify, response, hard | Fields affected by the error |
| fixed_request | string (JSON) | medium, hard | Corrected request body |
| fixed_headers | dict | medium, headers, hard | Corrected HTTP headers |
| explanation | string | hard | Developer-facing explanation |
| response_issues | list[string] | response | Issue types found in API response |
| expected_status_code | int | response | Correct HTTP status code |

## Observation Space

The environment returns an `APIDebugObservation` with:

| Field | Type | Description |
|-------|------|-------------|
| task | string | Current task: easy, classify, medium, headers, response, hard |
| api_name | string | Name of the API (e.g. "Create Customer") |
| http_method | string | HTTP method of the request |
| endpoint | string | API endpoint path |
| broken_request | string (JSON) | The malformed request body |
| broken_headers | dict | HTTP headers sent with the request |
| api_spec | string (JSON) | API specification with required fields and types |
| response_body | string (JSON) | Server response body (response task only) |
| response_status_code | int | HTTP status code of response (response task only) |
| error_count | int | Number of errors injected |
| step_number | int | Current step in the episode |
| max_steps | int | Maximum steps allowed |
| feedback | string | Structured validation feedback from last action |
| message | string | Human-readable status |
| done | bool | Whether the episode has ended |
| reward | float | Reward signal in open interval (0, 1) |

## Reward Design

Rewards are shaped per-step with decay to encourage efficient debugging:

```
reward = raw_score x max(1.0 - 0.1 x (step - 1), 0.3)
```

- Step 1: full reward (1.0x multiplier)
- Step 2: 0.9x multiplier
- Step 5: 0.6x multiplier
- Step 7+: 0.3x floor (agent still gets credit for late fixes)

At episode end, the best reward achieved across all steps is returned.

## Hard Task: LLM-as-Judge

The hard task uses an LLM (gpt-4o-mini via OpenAI API) to evaluate explanation quality. The judge receives the actual ground truth (error types and affected fields) and scores the agent's explanation on three criteria:

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Root cause identification | 0.4 | Does the explanation correctly name the error types and affected fields? |
| Fix guidance | 0.3 | Does it explain the correct remediation? |
| Developer clarity | 0.3 | Is the explanation actionable and clear for a developer? |

If the LLM judge is unavailable, the environment falls back to a keyword + length heuristic that ensures non-zero scores for reasonable explanations. A 10-second timeout on the judge call prevents blocking the episode if the LLM is slow.

To use a dedicated judge model (recommended to avoid the agent grading itself):

```bash
export JUDGE_MODEL=gpt-4o-mini
export JUDGE_API_BASE=https://api.openai.com/v1
export JUDGE_API_KEY=sk-your-key
```

If not set, the judge falls back to the agent's model (`MODEL_NAME`), then to the heuristic.

## Baseline Scores

Scores from running inference.py against the live HF Space (3 episodes per task, LLM-as-judge active for hard):

| Task | Qwen2.5-72B-Instruct | gpt-4o-mini |
|------|----------------------|-------------|
| easy | 0.999 | 0.799 |
| classify | 0.600 | 0.660 |
| medium | 0.999 | 0.999 |
| headers | 0.700 | 0.760 |
| response | 0.518 | 0.521 |
| hard | 0.830 | 0.713 |
| **overall** | **0.774** | **0.742** |

Hard task uses LLM-as-judge (gpt-4o-mini) for explanation quality scoring, which is stricter than a heuristic baseline. The agent must fix 2-3 simultaneous errors and provide a developer-facing explanation to score high. Larger models perform better on the hard task, showing meaningful difficulty progression.

## Chained Multi-Step Errors

The hard task supports chained error scenarios where errors depend on each other. Fixing one error reveals the next, simulating real-world API debugging:

| Chain Pattern | Gate Error | Body Errors |
|---------------|-----------|-------------|
| auth_gate | missing_auth_header, expired_auth_token | (any body error) |
| content_type_gate | wrong_content_type | wrong_field_type, malformed_json_value, invalid_enum_value |
| method_chain | wrong_http_method | missing_required_field, extra_unknown_field, null_value_in_required |
| rate_limit_chain | rate_limit_headers | expired_auth_token, missing_required_field |
| redirect_chain | redirect_loop | wrong_field_type, datetime_format_error, invalid_email_format |

## GRPO Training with Curriculum Learning

The `training/` directory contains a GRPO training script that trains a small LLM (Qwen 0.5B) using reward signals from the live environment:

```bash
pip install -r training/requirements.txt
python training/train.py
```

The training auto-promotes through 6 difficulty levels based on rolling average reward:

| Level | Task | Threshold | Max Turns |
|-------|------|-----------|-----------|
| 1 | easy | 0.7 | 3 |
| 2 | classify | 0.6 | 4 |
| 3 | medium | 0.6 | 5 |
| 4 | headers | 0.5 | 4 |
| 5 | response | 0.5 | 4 |
| 6 | hard | -- | 7 |

The environment also supports `task="auto"` which lets the environment itself manage curriculum progression based on session history.

### Training Results

Trained on Google Colab (free T4 GPU) with 64 episodes on the easy task:

| Metric | Value |
|--------|-------|
| Runtime | 7m 43s (8 steps) |
| Mean reward (easy) | 0.172 |
| Mean completion length | 62 tokens |
| Loss | -0.003 (converging) |
| GPU | Tesla T4, bf16 |

The trained model is available at: [avichauhan/api-debug-grpo-qwen-0.5b](https://huggingface.co/avichauhan/api-debug-grpo-qwen-0.5b)

A Colab notebook is provided at `training/train_colab.ipynb` for one-click training.

## Setup

### Prerequisites

- Python 3.12+
- Docker (for container deployment)
- uv (Python package manager)

### Local Development

```bash
git clone https://github.com/Avi-chauhan/api-debug-env.git
cd api-debug-env
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Run Server Locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
docker build -t api-debug-env:latest .
docker run -p 7860:7860 api-debug-env:latest
```

### Test

```python
import asyncio
from client import APIDebugEnv
from models import APIDebugAction

async def test():
    async with APIDebugEnv(base_url="http://localhost:8000") as env:
        result = await env.reset(task="easy")
        print(result.observation.message)

        action = APIDebugAction(
            error_type="missing_required_field",
            affected_fields=["email"]
        )
        result = await env.step(action)
        print(f"Reward: {result.reward}, Feedback: {result.observation.feedback}")

asyncio.run(test())
```

## Project Structure

```
api-debug-env/
├── Dockerfile              # Root level (required for HF Spaces)
├── requirements.txt
├── inference.py            # Baseline inference script (mandatory)
├── openenv.yaml            # OpenEnv manifest
├── pyproject.toml
├── README.md
├── models.py               # APIDebugAction, APIDebugObservation
├── client.py               # APIDebugEnv(EnvClient)
├── validate-submission.sh  # Pre-submission validator
├── __init__.py
├── tests/
│   ├── __init__.py
│   └── test_environment.py  # 109 unit tests
├── training/
│   ├── __init__.py
│   ├── train.py            # GRPO training with 6-level curriculum
│   ├── requirements.txt
│   └── README.md
└── server/
    ├── __init__.py
    ├── app.py              # FastAPI app via create_app()
    ├── environment.py      # Core logic: 6 tasks, graders, LLM judge, auto-curriculum
    ├── api_specs.py        # 45 API spec templates across 9 domains
    ├── error_injectors.py  # 15 error types + 5 chain patterns
    ├── response_specs.py   # 8 response templates + 5 issue injection types
    └── validators.py       # Field type validation helpers
```

## Deployment

### HuggingFace Spaces

```bash
openenv push --repo-id avichauhan/api-debug-env
```

HF Space URL: https://avichauhan-api-debug-env.hf.space

### Validation

```bash
./validate-submission.sh https://avichauhan-api-debug-env.hf.space .
```

## License

MIT
