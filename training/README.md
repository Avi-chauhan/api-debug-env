# Training with GRPO on API Debug Environment

Trains a small LLM using **GRPO** (Group Relative Policy Optimization)
on the live API Debug Environment with **curriculum learning**.

## What is GRPO?

For each prompt, GRPO:
1. Generates multiple completions (debug attempts)
2. Scores each with the environment's grader (reward signal)
3. Updates the model to prefer higher-scoring responses

Over thousands of episodes, the LLM learns to debug API requests
purely from reward signals -- no labelled data needed.

## Curriculum Learning

The training auto-promotes through difficulty levels:

| Level | Task | Threshold | Max Turns | Skill |
|-------|------|-----------|-----------|-------|
| 1 | easy | 0.7 avg reward | 3 | Identify single error type + fields |
| 2 | classify | 0.6 avg reward | 4 | Identify ALL error types + fields |
| 3 | medium | 0.6 avg reward | 5 | Fix the broken request body |
| 4 | headers | 0.5 avg reward | 4 | Fix header-level errors |
| 5 | hard | -- | 7 | Fix mixed errors + explain reasoning |

Promotion happens when the rolling average reward (window=10) exceeds
the threshold for the current level.

## Architecture
```
Dataset prompt ("Debug this broken API request.")
     |
GRPOTrainer calls rollout_func()
     |
rollout_func() connects to live HF Space via WebSocket
     |
env.reset(task=current_task) -> broken API request
     |
LLM generates JSON response -> env.step(action) -> reward
     |  (repeat up to max_turns)
Returns: prompt_ids, completion_ids, logprobs, env_reward
     |
reward_from_env() extracts env_reward
     |
GRPO updates model weights
     |
maybe_promote() checks if agent should advance to next task
```

## Run on Google Colab (free T4 GPU)
```python
# Cell 1 -- Install
!pip install trl>=0.26.0 transformers torch datasets openenv-core openai

# Cell 2 -- Clone repo
!git clone https://github.com/Avi-chauhan/api-debug-env.git
%cd api-debug-env

# Cell 3 -- Train
!python training/train.py
```

## Requirements

- GPU: T4 or better (free Colab works)
- RAM: 8GB+
- The live HF Space must be running:
  https://huggingface.co/spaces/avichauhan/api-debug-env
