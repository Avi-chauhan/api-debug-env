# RL Architecture: API Debug Environment

## Overview

This environment trains LLM agents to debug malformed API requests. A task that appears constantly in production systems. The agent receives a broken HTTP request and API specification, then must identify the error, fix the request, and explain its reasoning. This maps directly to how developers debug API integration failures every day.

---

## Episode Lifecycle

```
reset(task="easy"|"medium"|"hard")
  → picks 1 of 30 API spec templates at random
  → generates valid request from spec
  → injects 1-3 errors via error injectors
  → returns broken request + spec as observation

for step in 1..max_steps:
    step(action: APIDebugAction)
      → grades agent's response
      → applies step decay multiplier
      → returns feedback + reward

    if raw_score >= 0.95 or step == max_steps:
        return best_reward (highest achieved this episode)
        done = True
```

The agent can iterate - each step gets the previous step's structured feedback, allowing multi-turn refinement.

---

## Reward Design

### Formula

```
reward = raw_score × max(1.0 - 0.1 × (step - 1), 0.3)
```

| Step | Multiplier | Effect |
|------|------------|--------|
| 1    | 1.0x       | Full reward for immediate correct diagnosis |
| 2    | 0.9x       | Small penalty for needing one attempt |
| 5    | 0.6x       | Moderate penalty for slow convergence |
| 8+   | 0.3x floor | Agent still gets credit for late fixes |

At episode end, the **best reward** achieved across all steps is returned. This means:
- An agent that gets it right on step 1 scores highest
- An agent that improves over multiple steps still gets its best score
- Agents are incentivized to be efficient, not just eventually correct

### Per-Task Scoring

**Easy (error identification):**
```
raw_score = 0.6 × type_match + 0.4 × jaccard(agent_fields, gt_fields)
```
- `type_match`: 1.0 if error_type is correct, 0.0 otherwise
- `jaccard`: |intersection| / |union| for partial field credit

**Medium (request fix):**
```
raw_score = per-field validation against spec
```
Per check: required fields present, field types correct, no unknown fields. Equal weight across checks. If a header error was injected: `0.8 × body_score + 0.2 × header_score`.

**Hard (fix + explain):**
```
raw_score = 0.7 × fix_score + 0.3 × explanation_score
```
`fix_score` uses medium grading. `explanation_score` uses an LLM judge (gpt-4o-mini) with ground-truth-aware prompting, with a keyword + length heuristic fallback.

---

## Infinite Episode Generation

The key RL training advantage: every episode is unique.

```
30 API specs × 10 error types = 300 base combinations
Hard task: 2-3 simultaneous errors from 10 types = ~720 combinations
Total: thousands of distinct episodes
```

Contrast with fixed-fixture environments: an agent can memorize the answer after episode 1. Our generator forces the agent to learn a **generalizable debugging strategy**.

API spec domains:
- Payment (Stripe-like): 5 specs
- User Management: 5 specs
- Content (GitHub-like): 5 specs
- Messaging (Twilio-like): 5 specs
- E-Commerce: 5 specs
- Calendar and Auth: 5 specs

Error types: missing_required_field, wrong_field_type, invalid_email_format, missing_auth_header, extra_unknown_field, null_value_in_required, wrong_http_method, malformed_json_value, invalid_enum_value, datetime_format_error.

---

## Grading Philosophy

### Why Hybrid Grading (Deterministic + LLM)

Easy and medium tasks have objectively correct answers: a field is either present or missing, a type is either correct or wrong. Pure deterministic grading is appropriate.

Hard task requires evaluating **explanation quality** - whether the agent communicated the root cause clearly to a developer. This is inherently subjective and benefits from LLM judgment. The LLM judge receives the ground truth (actual error types and affected fields) so it scores the explanation against what was actually wrong, not in isolation.

The 70/30 split (fix vs. explain) reflects production reality: a correct fix without explanation leaves developers unable to prevent future recurrences.

### Structured Feedback

Every step returns machine-readable feedback:
```
Validation: 5/7 checks passed.
  email: PRESENT
  name: PRESENT
  email type: VALID
  amount type: INVALID (expected integer, got string)
  Authorization header: MISSING
```

This lets the agent know exactly which fields are still wrong, enabling targeted multi-turn improvement rather than blind re-guessing.

---

## Reward Range Compliance

All rewards are strictly in [0.0, 1.0]:
- `raw_score` is always in [0.0, 1.0] (ratios and weighted averages)
- `step_multiplier` is always in [0.3, 1.0]
- `reward = raw_score × step_multiplier` is therefore in [0.0, 1.0]
- `best_reward = max(rewards seen)` is also in [0.0, 1.0]

No reward shaping pushes values outside this range.

---

## Concurrency

`SUPPORTS_CONCURRENT_SESSIONS = True`. The server supports up to 10 concurrent environments (`max_concurrent_envs=10`). Each session maintains independent state: spec, broken request, ground truth, step count, and best reward. Sessions are identified by `episode_id`.

---

## Action Space

`APIDebugAction` fields (all optional, submit what you have):

| Field | Type | Task |
|-------|------|------|
| error_type | string | easy, hard |
| affected_fields | list[string] | easy, hard |
| fixed_request | string (JSON) | medium, hard |
| fixed_headers | dict | medium, hard |
| explanation | string | hard |

The agent can submit a partial action (diagnosis only, fix only, or everything). This makes multi-turn interaction natural: the agent can refine each component independently.
