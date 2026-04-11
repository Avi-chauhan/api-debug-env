# API Debug Environment - Technical Deep Dive

A complete behind-the-scenes breakdown of the environment: what it does, how every piece works at the code level, what makes it unique, the bugs that almost killed the submission, and where it can go next.

---

## Table of Contents

1. [What We Built](#what-we-built)
2. [Terminology Reference](#terminology-reference)
3. [Architecture Overview](#architecture-overview)
4. [Code-Level Walkthrough](#code-level-walkthrough)
5. [The Grading System](#the-grading-system)
6. [Reward Shaping and Step Decay](#reward-shaping-and-step-decay)
7. [LLM-as-Judge: How It Actually Works](#llm-as-judge-how-it-actually-works)
8. [Error Injection System](#error-injection-system)
9. [Infinite Unique Scenarios](#infinite-unique-scenarios)
10. [The Bugs That Almost Killed It](#the-bugs-that-almost-killed-it)
11. [Unique Features and Benefits](#unique-features-and-benefits)
12. [Baseline Results](#baseline-results)
13. [Scope for Advancement (Round 2)](#scope-for-advancement-round-2)
14. [Practical Applications: Where the Trained LLM Sits](#practical-applications-where-the-trained-llm-sits)

---

## What We Built

An RL (Reinforcement Learning) environment where an LLM agent learns to debug malformed API requests and validate API responses. The agent receives a broken HTTP request along with its API specification, and must handle 6 progressively harder tasks:

- **Easy**: Identify the error type and affected fields (single error)
- **Classify**: Identify ALL error types and fields across multiple simultaneous errors
- **Medium**: Fix the broken request body to match the spec
- **Headers**: Fix header-level errors (auth, content-type, expired tokens)
- **Response**: Validate API responses for wrong status codes, missing fields, data leaks
- **Hard**: Fix multi-error requests (including chained dependencies) and explain the fix

Built on the **OpenEnv framework** (by Meta/PyTorch), with 45 API specs across 9 domains, 15 error types, 5 chained error patterns, GRPO training pipeline, and 6-level curriculum learning.

---

## Terminology Reference

| Term | Meaning in this project |
|------|------------------------|
| **OpenEnv** | Meta/PyTorch framework for building RL environments. Provides `Environment` base class, `create_app()` for FastAPI, `EnvClient` for agents. |
| **Episode** | One complete debugging session. Starts with `reset()`, ends when `done=True`. |
| **Step** | One agent attempt within an episode. Agent submits an action, gets observation + reward + feedback. |
| **Observation** | What the environment returns to the agent: broken request, spec, feedback, reward, done flag. |
| **Action** | What the agent sends: error type, affected fields, fixed request, headers, explanation. |
| **Ground truth** | The correct answer stored internally. Includes the original valid request, error type, and affected fields. Never exposed to the agent. |
| **Reward** | Float in (0, 1) representing how good the agent's attempt was. Higher = better. |
| **Step decay** | Multiplier that reduces reward for later steps: 1.0x at step 1, down to 0.3x floor at step 7+. |
| **best_reward** | Highest reward achieved across all steps in an episode. Returned as final score. |
| **Jaccard similarity** | Set similarity metric: size of intersection divided by size of union. Used for partial credit on field identification. |
| **LLM-as-judge** | Using an LLM (gpt-4o-mini) to score the quality of the agent's explanation. Only for hard task. |
| **Heuristic fallback** | Keyword + length scoring when the LLM judge is unavailable. Ensures hard task never blocks. |
| **Reward clamping** | `max(0.001, min(0.999, reward))` - keeps rewards in open interval (0,1) because the evaluator rejects boundary values. |
| **create_app()** | OpenEnv function that takes your Environment class and generates a FastAPI app with all required endpoints. |
| **EnvClient** | OpenEnv SDK class that agents use to connect. Handles WebSocket, auto-connect via `_ensure_connected()`. |
| **GRPO** | Group Relative Policy Optimization. An RL algorithm suitable for training LLMs with environment reward signals. |
| **Deterministic grading** | Scoring that produces the same result every time for the same input. No randomness, no LLM calls. |
| **Procedural generation** | Creating scenarios algorithmically (random spec + random error + random field) rather than from a fixed dataset. |
| **HF Space** | HuggingFace Spaces - cloud deployment for the environment. Runs Docker container on port 7860. |
| **Phase 1** | Automated check: Docker builds, `openenv validate` passes, health endpoint responds. |
| **Phase 2** | Automated check: evaluator's agent runs against the environment, scores must be in (0, 1), no crashes. |
| **Phase 3** | Human judges review environment quality, RL training value, code quality, documentation. |

---

## Architecture Overview

```
                    +--------------------+
                    |   LLM Agent        |
                    |  (Qwen2.5-72B /    |
                    |   any model)       |
                    +--------+-----------+
                             |
                     WebSocket / HTTP
                             |
                    +--------v-----------+
                    |   OpenEnv SDK       |
                    |   create_app()      |
                    | POST /reset         |
                    | POST /step          |
                    | WS /ws             |
                    | GET /health        |
                    +--------+-----------+
                             |
              +--------------+---------------+
              |              |               |
    +---------v--+  +--------v---+  +--------v--------+
    | api_specs  |  | error_     |  | environment.py  |
    | .py        |  | injectors  |  | (core logic)    |
    | 45 specs   |  | .py        |  | reset(), step() |
    | 9 domains  |  | 15 types   |  | 6 graders       |
    +------------+  +------------+  +---------+-------+
                                              |
                                    +---------v-------+
                                    | validators.py   |
                                    | Field type      |
                                    | checking,       |
                                    | spec validation |
                                    +-----------------+
```

**Key files:**

| File | Lines | Purpose |
|------|-------|---------|
| `server/environment.py` | 472 | Core environment: reset(), step(), 3 graders, LLM judge, reward clamping |
| `server/api_specs.py` | 640 | 30 API spec templates across 6 domains |
| `server/error_injectors.py` | 389 | 10 error injection functions + multi-error injector |
| `server/validators.py` | 151 | 12 field type validators + request/header validation |
| `server/app.py` | 84 | FastAPI app via OpenEnv's `create_app()` + `/tasks` endpoint |
| `models.py` | 97 | Pydantic models: APIDebugAction (5 optional fields), APIDebugObservation (13 fields) |
| `client.py` | 80 | EnvClient SDK: `_step_payload`, `_parse_result`, `_parse_state` |
| `inference.py` | 331 | Baseline agent: LLM prompting, JSON parsing, episode runner, structured logging |
| `tests/test_environment.py` | 579 | 79 unit tests covering all graders, edge cases, reward bounds |

---

## Code-Level Walkthrough

### 1. Server Startup (`server/app.py`)

```python
app = create_app(
    APIDebugEnvironment,    # Our environment class
    APIDebugAction,         # What the agent sends
    APIDebugObservation,    # What the environment returns
    env_name="api_debug",
    max_concurrent_envs=10, # Supports 10 parallel sessions
)
```

`create_app()` is from OpenEnv SDK. It auto-generates all endpoints: `/reset`, `/step`, `/state`, `/schema`, `/ws`, `/health`. We just hand it our classes and it wires everything up. The WebSocket endpoint at `/ws` is what the evaluator's agent connects to.

We also added a custom `/tasks` endpoint that returns all task configs, error types, and spec count. This helps external agents understand what the environment offers without reading docs.

### 2. Episode Start: `reset()` (`server/environment.py:77-158`)

When an agent calls `reset(task="medium")`:

1. **Initialize RNG**: If a seed is provided, the episode is reproducible. Otherwise random.
2. **Load task config**: `TASK_CONFIG["medium"]` gives `{"max_steps": 5, "error_count": 1}`
3. **Pick random spec**: `get_random_spec(self.rng)` picks one of 30 API templates
4. **Deep copy the valid example**: We never mutate the template itself
5. **Inject errors**: For easy/medium, picks 1 random error type. For hard, picks 2-3 distinct error types using `rng.sample()` (no duplicates)
6. **Build observation**: Returns the broken request, headers, API spec (field types + required fields), error count, and a message like "Debug this POST /v1/customers request. It contains 1 error(s). You have 5 steps."

The agent never sees the ground truth. It only sees the broken request and the spec.

### 3. Agent Action: `step()` (`server/environment.py:160-213`)

When the agent submits a fix attempt:

1. **Increment step counter**: Tracks which step we're on
2. **Guard against stepping after done**: Returns 0 reward if episode already ended
3. **Route to correct grader**: `_grade_easy()`, `_grade_medium()`, or `_grade_hard()` based on task
4. **Apply step decay**: `multiplier = max(1.0 - 0.1 * (step - 1), 0.3)` - step 1 gets full reward, step 7+ gets 30% floor
5. **Clamp reward**: `max(0.001, min(0.999, reward))` - open interval (0, 1) because the evaluator rejects exactly 0.0 or 1.0
6. **Track best reward**: `self.best_reward = max(self.best_reward, reward)` - at episode end, returns the best reward across all steps
7. **Check termination**: Episode ends if `raw_score >= 0.95` (near-perfect) or all steps exhausted

### 4. Pydantic Models (`models.py`)

```python
class APIDebugAction(Action):
    error_type: Optional[str]        # "missing_required_field"
    affected_fields: Optional[List[str]]  # ["email"]
    fixed_request: Optional[str]     # JSON string of corrected body
    fixed_headers: Optional[Dict[str, str]]  # {"Authorization": "Bearer ..."}
    explanation: Optional[str]       # Developer-facing text
```

All fields are Optional. The agent submits only what's needed for the current task:
- Easy: `error_type` + `affected_fields`
- Medium: `fixed_request` + `fixed_headers`
- Hard: All five fields

```python
class APIDebugObservation(Observation):
    task, api_name, http_method, endpoint,     # Context
    broken_request, broken_headers, api_spec,  # The problem
    error_count, step_number, max_steps,       # Episode state
    feedback, message,                          # Grader output
    done, reward                                # From Observation base
```

### 5. Client SDK (`client.py`)

The client implements three abstract methods from `EnvClient`:

- **`_step_payload(action)`**: Converts `APIDebugAction` to a JSON dict, only including non-None fields
- **`_parse_result(payload)`**: Converts the server's JSON response into a `StepResult[APIDebugObservation]`
- **`_parse_state(payload)`**: Converts server state to an `episode_id` + `step_count`

The `EnvClient` base class handles WebSocket connection management. It has `_ensure_connected()` which auto-calls `connect()` if the WebSocket is not yet open. This means you can call `env.reset()` directly without explicitly opening a connection first.

---

## The Grading System

### Easy Grader (`_grade_easy`, line 223)

Fully deterministic. Two components:

| Component | Weight | How it works |
|-----------|--------|-------------|
| Error type match | 0.6 | Exact string match against ground truth error type(s) |
| Affected fields | 0.4 | Jaccard similarity: `|intersection| / |union|` of predicted vs actual fields |

Jaccard similarity gives partial credit. If the ground truth is `["email", "name"]` and the agent says `["email", "phone"]`, the intersection is `{"email"}`, union is `{"email", "name", "phone"}`, Jaccard = 1/3, so fields score = 0.4 * 0.33 = 0.13.

### Medium Grader (`_grade_medium`, line 264)

Fully deterministic per-field validation:

1. Parse the agent's `fixed_request` as JSON (fail = 0.0)
2. Check every required field is present and non-null
3. Check every present field has the correct type (using `validators.py`)
4. Check no unknown fields are present
5. Score = `passed_checks / total_checks`

If the original error was `missing_auth_header`, headers are also validated (80% body + 20% headers blend).

### Hard Grader (`_grade_hard`, line 307)

70% deterministic + 30% LLM-judged:

```python
total = 0.7 * fix_score + 0.3 * explain_score
```

The fix portion reuses the medium grader exactly. The explanation goes through `_score_explanation()` which tries the LLM judge first, then falls back to a heuristic.

---

## Reward Shaping and Step Decay

The reward formula:

```
reward = raw_score * max(1.0 - 0.1 * (step - 1), 0.3)
```

| Step | Multiplier | Effect |
|------|-----------|--------|
| 1 | 1.0x | Full reward for first-try solutions |
| 2 | 0.9x | Slight penalty |
| 3 | 0.8x | |
| 4 | 0.7x | |
| 5 | 0.6x | |
| 6 | 0.5x | |
| 7+ | 0.3x | Floor - agent still gets credit for late fixes |

**Why this matters for RL training:**
- The agent is incentivized to solve problems quickly (higher reward on step 1)
- But it's not punished to zero for needing multiple attempts (0.3x floor)
- The multi-turn feedback loop means the agent can learn from structured feedback between steps
- At episode end, `best_reward` is returned - the best score across all attempts

**Reward clamping** (`environment.py:194`):
```python
reward = max(0.001, min(0.999, reward))
```

This was added after Submission #3 failed. The evaluator's score range check requires strictly open interval (0, 1). Without this, a completely wrong answer scored 0.0 and a perfect first-step answer scored 1.0, both of which the evaluator rejected.

---

## LLM-as-Judge: How It Actually Works

### The Judge Call (`_llm_judge_explanation`, line 349)

The judge receives:
- The API name, method, and endpoint
- The **actual ground truth** errors (type + affected fields) - the judge knows the right answer
- The agent's explanation text

The judge scores on three weighted criteria:

| Criterion | Max Score | What it evaluates |
|-----------|----------|-------------------|
| Root cause identification | 0.4 | Did the agent correctly name the error types and affected fields? |
| Fix guidance | 0.3 | Does the explanation describe the correct remediation? |
| Developer clarity | 0.3 | Is the explanation actionable and clear for a real developer? |

The judge returns a single JSON object: `{"score": 0.85}`.

**Key design decisions:**
- **10-second timeout** (`timeout=10`): Prevents blocking `step()` if the LLM is slow. The grader must respond quickly.
- **temperature=0.0**: Deterministic judge output for consistency
- **max_tokens=50**: The judge only needs to return a score, not a long response

### Heuristic Fallback (`_heuristic_score_explanation`, line 398)

If the LLM judge fails (network error, timeout, bad response), we fall back to:

```python
keyword_score = min(keyword_hits / 6.0, 1.0)  # 23 debugging keywords
length_score = based on len(explanation)        # Sweet spot: 50-500 chars
final = 0.5 * keyword_score + 0.5 * length_score
```

Keywords include: "because", "should", "missing", "type", "format", "expected", "invalid", "authorization", "schema", "endpoint", "method", "payload", etc.

This ensures the hard task never gets stuck. Even if the LLM judge is completely unavailable, agents still get meaningful (if less precise) scores for reasonable explanations.

### How multiple valid paths are handled

This is the question Alexa asked. The answer:

- **For the fix portion (70%)**: Grading validates against the **spec**, not against a single golden answer. Any request that has all required fields with correct types and no unknown fields gets full credit. Two completely different valid fixes both score 1.0 on the fix portion.
- **For the explanation (30%)**: The LLM judge evaluates whether the agent **identified the actual injected errors**, not whether it matched specific phrasing. An explanation that says "the email field was missing" and one that says "a required field (email) was not included in the request body" both get credit for root cause identification.

---

## Error Injection System

### The 10 Error Types (`server/error_injectors.py`)

Each injector is a pure function: `(request, headers, spec, rng) -> (broken_request, broken_headers, ground_truth)`

| # | Error Type | What it does | Code location |
|---|-----------|-------------|---------------|
| 1 | `missing_required_field` | Removes a random required field from the request body | Line 38 |
| 2 | `wrong_field_type` | Changes a field's value to the wrong type (int to string, etc.) | Line 62 |
| 3 | `invalid_email_format` | Corrupts an email field (e.g. `user@` or `@domain.com`) | Line 103 |
| 4 | `missing_auth_header` | Removes the Authorization header | Line 131 |
| 5 | `extra_unknown_field` | Adds a field not in the spec (`debug_mode: true`, `_private`, etc.) | Line 159 |
| 6 | `null_value_in_required` | Sets a required field to `null` | Line 185 |
| 7 | `wrong_http_method` | Records that the wrong HTTP method was shown | Line 209 |
| 8 | `malformed_json_value` | Replaces a field's value with broken JSON fragments (`{broken`, `NaN`) | Line 235 |
| 9 | `invalid_enum_value` | Uses a value not in the allowed enum list | Line 270 |
| 10 | `datetime_format_error` | Replaces ISO 8601 datetime with wrong format (`04/01/2026`) | Line 297 |

### Multi-Error Injection (`inject_multiple_errors`, line 370)

For hard tasks, we inject 2-3 errors simultaneously:

```python
chosen_types = rng.sample(ERROR_TYPES, min(count, len(ERROR_TYPES)))
for err_type in chosen_types:
    broken_req, broken_hdrs, gt = injector(broken_req, broken_hdrs, spec, rng)
    all_truths.append(gt)
```

`rng.sample()` picks without replacement, so the agent never sees two of the same error type in one episode. Errors are applied sequentially to the same request, so they can compound (e.g., a field gets removed AND a type gets changed on another field).

### Fallback Handling

Some injectors need specific field types that might not exist in the chosen spec. For example, `inject_invalid_email_format` needs an email field. If the spec has no email fields, it falls back to `inject_missing_required_field` instead. Same for `inject_invalid_enum_value` (falls back to `inject_wrong_field_type`) and `inject_datetime_format_error`.

---

## Infinite Unique Scenarios

The combinatorial space:

- **30 API specs** across 6 domains
- **10 error types** (each with random field selection within the spec)
- **Multiple bad values per error type** (e.g., 5 bad email formats, 5 malformed JSON fragments, 5 bad datetime formats)
- **Random field selection** within each spec (which required field gets removed, which gets type-changed)
- **Hard mode**: 2-3 errors from different types, applied sequentially

Conservative estimate: 30 specs x 10 error types x 3 field choices x 5 bad values = **4,500+ unique easy/medium scenarios**. Hard mode with combinations: significantly more.

**Why this matters for RL**: An agent cannot memorize answers after one training run. It must learn a generalizable debugging strategy. If your environment has fixed scenarios, the agent overfits to those specific cases. With procedural generation, the agent has to learn the underlying skill.

---

## The Bugs That Almost Killed It

### Submission #2: "inference.py raised an unhandled exception"

**What happened**: The evaluator runs `inference.py` in their own runtime. Our script had two unprotected lines in `main()`:

1. `OpenAI(api_key=None)` - When `HF_TOKEN` is not set in the evaluator's environment, `os.getenv("HF_TOKEN")` returns `None`. The OpenAI client constructor crashes immediately with `OpenAIError: The api_key client option must be set` when given `None`. This was confirmed by running it locally.

2. `await APIDebugEnv.from_docker_image(IMAGE_NAME)` - If Docker isn't available or the image doesn't exist, this throws an unhandled exception.

**Fix**:
- Added fallback chain: `API_KEY = HF_TOKEN or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")`
- Wrapped `from_docker_image()` in try/except with fallback to direct URL connection
- Added try/except around each episode so one failure doesn't crash the whole run

**Misdiagnosis**: Another AI agent suggested the bug was a missing `async with` context manager for the WebSocket connection. This was wrong. We verified by reading the OpenEnv SDK source: `EnvClient._ensure_connected()` auto-calls `connect()` when the WebSocket is None. No explicit context manager needed.

### Submission #3: "One or more task scores are out of range"

**What happened**: The evaluator requires scores in the strictly open interval (0, 1). Our graders returned:
- Exactly `0.0` for completely wrong answers (empty action, invalid JSON, etc.)
- Exactly `1.0` for perfect first-step answers (1.0 raw score x 1.0 step multiplier)

Both boundary values were rejected.

**Fix**: Added reward clamping in `environment.py` step() method:
```python
reward = max(0.001, min(0.999, reward))
```

Also added score clamping in `inference.py`:
```python
score = min(max(score, 0.001), 0.999)
```

Updated 7 unit tests that previously asserted `== 0.0` or `== 1.0` to use `<= 0.01` and `>= 0.99`.

---

## Unique Features and Benefits

### 1. Procedural Generation (Not Fixed Datasets)
Most RL environments have a fixed set of scenarios. Our environment generates new scenarios every episode. 30 specs x 10 error types x random field selection = thousands of unique episodes. The agent genuinely learns a skill, not a lookup table.

### 2. Multi-Turn with Structured Feedback
The agent doesn't just get "right" or "wrong". It receives structured feedback like:
```
Validation: 3/5 checks passed.
  email: PRESENT
  name: MISSING
  email type: VALID (email)
  amount type: INVALID (expected integer)
  debug_mode: UNKNOWN FIELD (not in spec)
```
This feedback loop lets the agent iterate and improve within an episode.

### 3. Progressive Difficulty
Three clearly separated difficulty levels:
- Easy: Classification only (identify the error)
- Medium: Fixing (produce correct JSON)
- Hard: Fixing + reasoning (explain like a developer)

Each level builds on the previous, and the hard task genuinely tests whether the agent can reason about what went wrong, not just pattern-match.

### 4. Deterministic + LLM Hybrid Grading
Easy and medium tasks are 100% deterministic. No variance, no API calls, reproducible results. Hard tasks are 70% deterministic (the fix portion) + 30% LLM-judged (explanation quality). This hybrid approach means:
- Most of the score is stable and reproducible
- The subjective part (explanation) uses an actual LLM to judge quality
- If the LLM judge fails, a keyword heuristic ensures the system never blocks

### 5. Reward Shaping for Efficient Problem-Solving
Step decay encourages solving on the first try. The 0.3x floor means late fixes are still rewarded. `best_reward` tracking means the agent's best attempt is what counts.

### 6. Real-World Domain Relevance
API debugging is a real developer pain point. Calendar Gym research showed malformed tool arguments caused >50% of agent failures. This environment directly trains agents to handle that failure mode.

### 7. Concurrent Session Support
`SUPPORTS_CONCURRENT_SESSIONS = True` + `max_concurrent_envs=10` means the environment can train 10 agents in parallel on the same deployment. Each session has its own state.

### 8. 79 Unit Tests (579 lines)
Comprehensive test coverage:
- All three graders (easy, medium, hard)
- Step decay multiplier values at each step
- Reward bounds (never < 0.001, never > 0.999)
- Episode termination conditions
- Seeded reproducibility
- Edge cases (empty actions, invalid JSON, non-dict JSON, extra fields)
- Best reward tracking
- Heuristic explanation scorer

---

## Baseline Results

Scores from running `inference.py` against the live HF Space (3 episodes per task):

| Task | Episodes | Qwen2.5-72B-Instruct | gpt-4o-mini |
|------|----------|----------------------|-------------|
| easy | 3 | 0.999 | 0.667 |
| medium | 3 | 0.999 | 0.999 |
| hard | 3 | 0.780 | 0.760 |
| **overall** | **9** | **0.926** | **0.809** |

**Key takeaway**: Larger models perform better on hard tasks (explanation quality + multi-error fixing), showing meaningful difficulty progression. The environment is not trivially solvable but also not impossibly hard.

---

## Implemented Advancements (Round 2)

All five advancement items from the original roadmap have been implemented:

### 1. GRPO Training Pipeline (IMPLEMENTED)
**What**: Full GRPO training loop in `training/train.py` that trains Qwen 0.5B on the live environment using TRL's GRPOTrainer with vLLM colocate mode.
**How it works**: The model generates JSON debugging attempts, the environment grades them via its deterministic graders, and GRPO updates the policy to prefer higher-scoring responses. The rollout function connects to the live HF Space via WebSocket, runs multi-turn episodes, and returns prompt_ids, completion_ids, logprobs, and env_reward.
**Key config**: `max_completion_length=128`, `gradient_accumulation_steps=16`, `vllm_gpu_memory_utilization=0.3`. Runs on free Colab T4 GPU.

### 2. Expanded API Specs and Domains (IMPLEMENTED)
**What**: Expanded from 30 specs / 6 domains to 45 specs / 9 domains.
**New domains**: Analytics/Monitoring (dashboards, metrics, alerts), DevOps/Infrastructure (deployments, DNS, load balancers), AI/ML APIs (inference, fine-tuning, embeddings).
**Impact**: 50% more scenario diversity for training generalization. Each domain uses realistic field types, headers, and validation rules.

### 3. Chained Multi-Step Error Scenarios (IMPLEMENTED)
**What**: 5 chain patterns where fixing one error reveals the next, simulating real-world API debugging.
**Chain patterns**:
- **auth_gate**: Missing/expired auth blocks body error visibility
- **content_type_gate**: Wrong content type masks type/value errors
- **method_chain**: Wrong HTTP method hides field-level errors
- **rate_limit_chain**: Rate limit headers combined with auth/field errors
- **redirect_chain**: Redirect loops combined with type/format errors

**How it works**: `inject_chained_errors()` picks a random chain pattern, applies the gate error first, then injects body errors from the pattern's pool. The hard task uses chained errors 50% of the time.

### 4. Response Validation Task (IMPLEMENTED)
**What**: 6th task where the agent receives a request-response pair and identifies response issues.
**Issue types**: wrong_status_code, missing_response_field, wrong_response_type, extra_response_field (data leak detection), inconsistent_error_format.
**Grading**: 0.5 x Jaccard(issue_types) + 0.3 x Jaccard(affected_fields) + 0.2 x status_code_match.
**8 response templates** covering: Create, List, Update, Delete, Batch, Authentication, File Upload, Search operations.

### 5. Curriculum Learning (IMPLEMENTED)
**What**: Both training-side and environment-side curriculum learning.
**Training side** (`training/train.py`): 6-level curriculum that auto-promotes through easy -> classify -> medium -> headers -> response -> hard based on rolling average reward exceeding thresholds (0.7, 0.6, 0.6, 0.5, 0.5).
**Environment side** (`task="auto"`): The environment itself tracks per-session reward history and auto-promotes, so any client can benefit from adaptive difficulty without implementing curriculum logic.

### Scope for Further Advancement

- **GraphQL and gRPC protocols**: Add non-REST API specs for cross-protocol debugging
- **OAuth flow simulation**: Multi-step auth flows with token refresh, scope validation
- **Response body fixing**: Agent generates the correct response body, not just identifies issues
- **Multi-agent debugging**: Two agents collaborate on different aspects (headers vs body)
- **Real-world API replay**: Import failed requests from production logs for training data

---

## Practical Applications: Where the Trained LLM Sits

An LLM trained on this environment learns a specific skill: given a broken API request and a spec, diagnose the error, fix the request, and explain what went wrong. Here's how that skill translates to real-world developer tooling:

### 1. IDE Integration (Copilot-Style API Debugger)

```
Developer writes code          Trained LLM              API Server
       |                           |                        |
       |--- makes API call ------->|                        |
       |                           |--- forwards request -->|
       |                           |<-- 400/422 error ------|
       |                           |                        |
       |                     [LLM analyzes request          |
       |                      vs API spec, identifies       |
       |                      error, generates fix]         |
       |                           |                        |
       |<-- "Your 'amount' field   |                        |
       |    is a string but the    |                        |
       |    API expects integer.   |                        |
       |    Here's the fix: ..."   |                        |
```

**Where it sits**: As a VS Code / JetBrains extension plugin. Intercepts failed API calls in the developer's HTTP client (like Postman, Thunder Client, or `fetch`/`requests` in code), compares the request against known API specs, and suggests fixes inline.

**Developer experience**: Developer hits "Send" on an API request, gets a 400 error. Instead of reading the error response and manually debugging, the extension pops up: "Missing required field `email`. The spec requires it as type `email`. Here's the corrected request." One click to apply the fix.

### 2. API Gateway Middleware (Pre-Request Validation Layer)

```
Client App        API Gateway + Trained LLM           Backend API
    |                      |                              |
    |--- POST /v1/users -->|                              |
    |    {bad request}     |                              |
    |                      |-- [LLM validates against     |
    |                      |   API schema before          |
    |                      |   forwarding]                |
    |                      |                              |
    |<-- 422 + fix hint ---|                              |
    |    "Field 'email'    |                              |
    |     is malformed.    |                              |
    |     Expected format: |                              |
    |     user@domain.com" |                              |
```

**Where it sits**: As a middleware layer in an API gateway (Kong, AWS API Gateway, Nginx). Before the request reaches the backend, the LLM validates it against the spec and returns human-readable fix suggestions instead of cryptic validation errors.

**Developer experience**: Instead of getting `{"error": "validation_error", "detail": [{"loc": ["body", "email"], "msg": "value is not a valid email address"}]}`, the developer gets: "The `email` field contains `user@` which is not a valid email. A valid email must have a domain (e.g., `user@example.com`). The `amount` field is a string but should be an integer. Send `2500` instead of `\"2500\"`."

### 3. CI/CD Pipeline Integration (Contract Testing)

```
Developer pushes code
        |
        v
CI Pipeline runs
        |
        v
API Contract Tests (using trained LLM)
        |
        |--- Replays recent API calls against updated spec
        |--- LLM identifies breaking changes
        |--- Generates migration guide
        |
        v
PR Comment: "3 API calls in your test suite
will break with the new spec. Here are the fixes..."
```

**Where it sits**: As a CI step that runs after API spec changes. The trained LLM compares existing API calls (from test suites, logs, or recorded traffic) against the updated spec and flags what will break.

**Developer experience**: Developer updates an API schema (adds a required field). The CI pipeline catches that 15 existing test calls are now invalid and generates the exact fix for each one.

### 4. Production Error Analysis (Log-Based Debugging)

```
Production System           Error Aggregator          Trained LLM
      |                          |                        |
      |--- 400/422 errors ------>|                        |
      |--- request logs -------->|                        |
      |                          |--- batch analysis ---->|
      |                          |                        |
      |                          |<-- "Top 3 error        |
      |                          |    patterns:           |
      |                          |    1. 40% of failures  |
      |                          |       are datetime     |
      |                          |       format errors    |
      |                          |       in /v1/events    |
      |                          |    2. ..."             |
```

**Where it sits**: Connected to error aggregation tools (Sentry, Datadog, PagerDuty). Analyzes batches of 4xx errors, groups them by root cause, and suggests API spec improvements or client-side fixes.

**Developer experience**: Oncall engineer gets a Slack alert: "87 new 422 errors on POST /v1/subscriptions in the last hour. Root cause: mobile clients sending `start_date` as `MM/DD/YYYY` instead of ISO 8601. Suggested fix: add format hint to error response, or accept both formats in the endpoint."

### 5. SDK/Documentation Generator

```
API Spec (OpenAPI/Swagger)
        |
        v
Trained LLM analyzes common error patterns
        |
        v
Auto-generated:
  - "Common mistakes" section per endpoint
  - Request validation examples
  - Error handling code snippets
  - Migration guides between API versions
```

**Where it sits**: As part of the API documentation pipeline. The LLM, having been trained on thousands of debugging scenarios, knows which errors are most common per endpoint type and generates preventive documentation.

### Key Insight

The environment trains a skill that's useful at **every layer of the API stack** - from the developer's IDE to the API gateway to production monitoring. The core capability (understand spec, diagnose broken request, suggest fix) is the same; only the integration point changes. A model trained on this environment could power any of these tools, because the underlying reasoning is identical: compare request against spec, find the mismatch, produce the fix.
