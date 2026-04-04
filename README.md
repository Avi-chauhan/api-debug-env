---
title: API Debug Environment
emoji: "🔧"
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# API Debug Environment

An OpenEnv reinforcement learning environment where LLM agents learn to debug malformed API requests. The agent receives a broken request and its API specification, then must diagnose the error, fix the request, and explain the fix.

Built for the Meta PyTorch OpenEnv Hackathon x Scaler School of Technology 2026.

## Why This Domain

Developers spend significant time debugging API contract mismatches. Research from Calendar Gym shows that malformed tool arguments caused more than half of agent failures. This environment trains agents to identify and fix these errors systematically.

**Real-world applications:** API gateway validation, automated debugging assistants, developer tooling, CI/CD request validation, LLM tool-use reliability.

## How It Works

1. On `reset()`, the environment picks a random API spec from 30 templates and injects 1-3 errors
2. The agent receives the broken request, headers, and the API specification
3. The agent submits a fix attempt via `step()`
4. The environment grades the attempt and returns structured feedback
5. The agent can iterate (multi-turn) using the feedback to improve

Each episode allows multiple attempts. Perfect answers on early steps earn full reward. Later steps get decayed reward, encouraging efficient debugging.

## Tasks

| Task | Difficulty | Max Steps | Errors | Grading |
|------|-----------|-----------|--------|---------|
| easy | Identify error type and affected fields | 3 | 1 | Deterministic: 0.6 x type_match + 0.4 x fields_match |
| medium | Fix the broken request | 5 | 1 | Deterministic: per-field validation against spec |
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

## API Spec Domains (30 templates)

| Domain | Count | Examples |
|--------|-------|---------|
| Payment (Stripe-like) | 5 | Create Customer, Create Charge, Process Refund |
| User Management | 5 | Create User, Update Profile, Reset Password |
| Content (GitHub-like) | 5 | Create Repository, Create Issue, Merge PR |
| Messaging (Twilio-like) | 5 | Send SMS, Send Email, Create Webhook |
| E-Commerce | 5 | Create Order, Process Payment, Create Shipping Label |
| Calendar and Auth | 5 | Create Event, OAuth Token, Create API Key |

## Action Space

The agent sends an `APIDebugAction` with these fields (all optional, submit what you have):

| Field | Type | Used In | Description |
|-------|------|---------|-------------|
| error_type | string | easy, hard | Diagnosed error type |
| affected_fields | list[string] | easy, hard | Fields affected by the error |
| fixed_request | string (JSON) | medium, hard | Corrected request body |
| fixed_headers | dict | medium, hard | Corrected HTTP headers |
| explanation | string | hard | Developer-facing explanation |

## Observation Space

The environment returns an `APIDebugObservation` with:

| Field | Type | Description |
|-------|------|-------------|
| task | string | Current difficulty: easy, medium, hard |
| api_name | string | Name of the API (e.g. "Create Customer") |
| http_method | string | HTTP method of the request |
| endpoint | string | API endpoint path |
| broken_request | string (JSON) | The malformed request body |
| broken_headers | dict | HTTP headers sent with the request |
| api_spec | string (JSON) | API specification with required fields and types |
| error_count | int | Number of errors injected |
| step_number | int | Current step in the episode |
| max_steps | int | Maximum steps allowed |
| feedback | string | Structured validation feedback from last action |
| message | string | Human-readable status |
| done | bool | Whether the episode has ended |
| reward | float | Reward signal (0.0 to 1.0) |

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

## Baseline Scores

Scores from running inference.py against the live HF Space (3 episodes per task, LLM-as-judge active for hard):

| Task | Episodes | Qwen2.5-72B-Instruct | gpt-4o-mini |
|------|----------|----------------------|-------------|
| easy | 3 | 1.000 | 0.667 |
| medium | 3 | 1.000 | 1.000 |
| hard | 3 | 0.780 | 0.760 |
| **overall** | **9** | **0.927** | **0.809** |

Hard task uses LLM-as-judge (gpt-4o-mini) for explanation quality scoring, which is stricter than a heuristic baseline. The agent must fix 2-3 simultaneous errors and provide a developer-facing explanation to score high. Larger models perform better on the hard task, showing meaningful difficulty progression.

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
└── server/
    ├── __init__.py
    ├── app.py              # FastAPI app via create_app()
    ├── environment.py      # Core logic: reset(), step(), graders + LLM judge
    ├── api_specs.py        # 30 API spec templates
    ├── error_injectors.py  # 10 error injection functions
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
