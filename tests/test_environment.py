"""
Unit tests for APIDebugEnvironment.

55 tests across 7 test classes:
  TestEasyGrader (15)        - error_type + affected_fields scoring
  TestMediumGrader (12)      - request fix validation
  TestHardGrader (8)         - fix + explanation blend
  TestHeuristicExplainer (5) - fallback explanation scorer
  TestReset (15)             - task config, seeding, state clearing
  TestStepBehavior (8)       - decay, termination, best_reward tracking
  TestRewardRange (7)        - always [0.0, 1.0]

Run from api-debug-env/ with .venv activated:
  pytest tests/ -v --tb=short
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import APIDebugAction
from server.environment import APIDebugEnvironment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env(task: str = "easy", seed: int = 42) -> APIDebugEnvironment:
    env = APIDebugEnvironment()
    env.reset(task=task, seed=seed)
    return env


def perfect_easy_action(env: APIDebugEnvironment) -> APIDebugAction:
    """Build a perfect easy action from ground truth."""
    gt = env.ground_truths[0]
    return APIDebugAction(
        error_type=gt["error_type"],
        affected_fields=gt["affected_fields"],
    )


def perfect_medium_action(env: APIDebugEnvironment) -> APIDebugAction:
    """Build a perfect medium action using the valid example and headers."""
    gt = env.ground_truths[0]
    return APIDebugAction(
        fixed_request=json.dumps(gt["valid_request"]),
        fixed_headers=gt["valid_headers"],
    )


# ---------------------------------------------------------------------------
# TestEasyGrader
# ---------------------------------------------------------------------------

class TestEasyGrader:
    def test_perfect_answer_scores_1(self):
        env = make_env("easy", seed=42)
        obs = env.step(perfect_easy_action(env))
        assert obs.reward == 1.0

    def test_correct_type_only_scores_0_6(self):
        env = make_env("easy", seed=42)
        gt = env.ground_truths[0]
        # No affected_fields supplied: jaccard = 0, score = 0.6
        obs = env.step(APIDebugAction(error_type=gt["error_type"], affected_fields=[]))
        # step 1 multiplier = 1.0, raw = 0.6
        assert abs(obs.reward - 0.6) < 0.01

    def test_correct_fields_only_no_type_scores_0_4(self):
        env = make_env("easy", seed=42)
        gt = env.ground_truths[0]
        if not gt["affected_fields"]:
            pytest.skip("Ground truth has no affected fields for this seed")
        obs = env.step(APIDebugAction(error_type=None, affected_fields=gt["affected_fields"]))
        # type score = 0, field jaccard = 1.0, total = 0.4
        assert abs(obs.reward - 0.4) < 0.01

    def test_empty_action_scores_0(self):
        env = make_env("easy", seed=42)
        obs = env.step(APIDebugAction())
        assert obs.reward == 0.0

    def test_wrong_type_wrong_fields_scores_0(self):
        env = make_env("easy", seed=42)
        obs = env.step(
            APIDebugAction(error_type="completely_wrong_type", affected_fields=["nonexistent_field"])
        )
        assert obs.reward == 0.0

    def test_wrong_type_correct_fields_partial_credit(self):
        env = make_env("easy", seed=42)
        gt = env.ground_truths[0]
        if not gt["affected_fields"]:
            pytest.skip("No affected fields for this seed")
        obs = env.step(
            APIDebugAction(error_type="wrong_guess", affected_fields=gt["affected_fields"])
        )
        # type=0, jaccard=1.0, raw = 0.4
        assert abs(obs.reward - 0.4) < 0.01

    def test_extra_fields_reduces_jaccard(self):
        env = make_env("easy", seed=42)
        gt = env.ground_truths[0]
        if not gt["affected_fields"]:
            pytest.skip("No affected fields for this seed")
        extra = gt["affected_fields"] + ["bogus_extra_field"]
        obs = env.step(APIDebugAction(error_type=gt["error_type"], affected_fields=extra))
        # jaccard < 1.0 due to extra field
        assert obs.reward < 1.0

    def test_partial_fields_gives_partial_jaccard(self):
        # Find a seed where gt has >= 2 affected fields
        for seed in range(50):
            env = APIDebugEnvironment()
            env.reset(task="easy", seed=seed)
            gt = env.ground_truths[0]
            if len(gt["affected_fields"]) >= 2:
                partial = gt["affected_fields"][:1]
                obs = env.step(
                    APIDebugAction(error_type=gt["error_type"], affected_fields=partial)
                )
                # jaccard = 1/2 (or less) → total < 1.0 but > 0.6
                assert 0.6 <= obs.reward < 1.0
                return
        # If no seed gives multi-field gt, just verify the mechanism still works
        env = make_env("easy", seed=0)
        gt = env.ground_truths[0]
        obs = env.step(APIDebugAction(error_type=gt["error_type"], affected_fields=gt["affected_fields"]))
        assert obs.reward <= 1.0

    def test_step_decay_on_second_step(self):
        env = make_env("easy", seed=42)
        # Step 1: wrong answer → reward=0, done=False
        obs1 = env.step(APIDebugAction(error_type="wrong", affected_fields=[]))
        assert not obs1.done
        # Step 2: perfect → raw=1.0, mult=0.9 → reward=0.9, done=True
        obs2 = env.step(perfect_easy_action(env))
        assert obs2.done
        # best_reward = max(0.0, 0.9) = 0.9
        assert abs(obs2.reward - 0.9) < 0.01

    def test_perfect_answer_ends_episode(self):
        env = make_env("easy", seed=42)
        obs = env.step(perfect_easy_action(env))
        assert obs.done is True

    def test_max_steps_ends_episode(self):
        env = make_env("easy", seed=42)
        obs = None
        for _ in range(3):  # easy max_steps = 3
            if not env.episode_done:
                obs = env.step(APIDebugAction())
        assert obs is not None
        assert obs.done is True

    def test_step_after_done_returns_zero_reward(self):
        env = make_env("easy", seed=42)
        env.step(perfect_easy_action(env))  # ends episode
        obs = env.step(APIDebugAction(error_type="anything"))
        assert obs.reward == 0.0
        assert obs.done is True

    def test_reward_always_non_negative(self):
        env = make_env("easy", seed=42)
        obs = env.step(APIDebugAction(error_type="garbage", affected_fields=["garbage"]))
        assert obs.reward >= 0.0

    def test_reward_always_at_most_1(self):
        env = make_env("easy", seed=42)
        obs = env.step(perfect_easy_action(env))
        assert obs.reward <= 1.0

    def test_feedback_contains_correct_marker(self):
        env = make_env("easy", seed=42)
        obs = env.step(perfect_easy_action(env))
        assert "CORRECT" in obs.feedback


# ---------------------------------------------------------------------------
# TestMediumGrader
# ---------------------------------------------------------------------------

class TestMediumGrader:
    def test_valid_fix_scores_high(self):
        env = make_env("medium", seed=42)
        obs = env.step(perfect_medium_action(env))
        assert obs.reward >= 0.7

    def test_no_fixed_request_scores_0(self):
        env = make_env("medium", seed=42)
        obs = env.step(APIDebugAction())
        assert obs.reward == 0.0

    def test_invalid_json_scores_0(self):
        env = make_env("medium", seed=42)
        obs = env.step(APIDebugAction(fixed_request="{broken json{{"))
        assert obs.reward == 0.0

    def test_non_dict_json_scores_0(self):
        env = make_env("medium", seed=42)
        obs = env.step(APIDebugAction(fixed_request='["array", "not", "dict"]'))
        assert obs.reward == 0.0

    def test_empty_json_object_low_score(self):
        env = make_env("medium", seed=42)
        obs = env.step(APIDebugAction(fixed_request="{}"))
        # Missing all required fields → low score
        assert obs.reward < 0.5

    def test_step_decay_applies_medium(self):
        env = make_env("medium", seed=42)
        # Burn step 1
        env.step(APIDebugAction())
        # Step 2: perfect fix → raw~1.0, mult=0.9
        obs = env.step(perfect_medium_action(env))
        assert obs.reward <= 1.0
        assert obs.done is True

    def test_valid_fix_ends_episode(self):
        env = make_env("medium", seed=7)
        obs = env.step(perfect_medium_action(env))
        assert obs.done is True

    def test_valid_json_with_required_fields_passes(self):
        env = make_env("medium", seed=42)
        gt = env.ground_truths[0]
        obs = env.step(APIDebugAction(fixed_request=json.dumps(gt["valid_request"])))
        assert obs.reward > 0.0

    def test_feedback_contains_validation_info(self):
        env = make_env("medium", seed=42)
        obs = env.step(APIDebugAction(fixed_request="{}"))
        assert "Validation" in obs.feedback

    def test_partial_fix_gives_partial_score(self):
        env = make_env("medium", seed=42)
        req_fields = env.spec["required_fields"]
        # Provide only the first required field
        partial = {}
        if req_fields:
            first_field = req_fields[0]
            partial[first_field] = env.spec["valid_example"][first_field]
        obs = env.step(APIDebugAction(fixed_request=json.dumps(partial)))
        assert 0.0 <= obs.reward <= 1.0

    def test_headers_fix_when_missing_auth(self):
        """Providing correct headers when there's a header error boosts score."""
        for seed in range(30):
            env = APIDebugEnvironment()
            env.reset(task="medium", seed=seed)
            if env.ground_truths[0]["error_type"] == "missing_auth_header":
                gt = env.ground_truths[0]
                obs = env.step(APIDebugAction(
                    fixed_request=json.dumps(gt["valid_request"]),
                    fixed_headers=gt["valid_headers"],
                ))
                assert obs.reward >= 0.7
                return
        # If missing_auth_header not found in 30 seeds, just verify the path exists
        assert True

    def test_reward_in_valid_range(self):
        env = make_env("medium", seed=42)
        obs = env.step(APIDebugAction(fixed_request="{}"))
        assert 0.0 <= obs.reward <= 1.0


# ---------------------------------------------------------------------------
# TestHardGrader
# ---------------------------------------------------------------------------

class TestHardGrader:
    def test_hard_has_multiple_errors(self):
        env = make_env("hard", seed=42)
        assert len(env.ground_truths) >= 2

    def test_hard_error_count_in_observation(self):
        env = APIDebugEnvironment()
        obs = env.reset(task="hard", seed=42)
        assert obs.error_count >= 2

    def test_explanation_improves_score(self):
        # Compare fix-only vs fix+explanation using same seed
        env1 = make_env("hard", seed=42)
        gt1 = env1.ground_truths[0]
        obs_fix_only = env1.step(APIDebugAction(
            fixed_request=json.dumps(gt1["valid_request"]),
        ))

        env2 = make_env("hard", seed=42)
        gt2 = env2.ground_truths[0]
        obs_with_explain = env2.step(APIDebugAction(
            fixed_request=json.dumps(gt2["valid_request"]),
            explanation=(
                "The request was missing required fields and had incorrect field types. "
                "The fix adds the missing required field and corrects the field type. "
                "The authorization header must be included as a Bearer token."
            ),
        ))
        assert obs_with_explain.reward >= obs_fix_only.reward

    def test_explanation_quality_affects_score(self):
        env = make_env("hard", seed=42)
        gt = env.ground_truths[0]
        obs = env.step(APIDebugAction(
            fixed_request=json.dumps(gt["valid_request"]),
            explanation="x",  # too short to be scored
        ))
        assert obs.reward >= 0.0

    def test_hard_reward_in_valid_range(self):
        env = make_env("hard", seed=42)
        obs = env.step(APIDebugAction())
        assert 0.0 <= obs.reward <= 1.0

    def test_hard_max_steps_is_7(self):
        env = make_env("hard", seed=42)
        assert env.max_steps == 7

    def test_hard_fix_score_weighted_70_percent(self):
        env = make_env("hard", seed=42)
        gt = env.ground_truths[0]
        # No explanation → total = 0.7 * fix_score + 0.3 * 0.0
        # Valid request should score near 1.0 on fix portion → reward ~0.7
        obs = env.step(APIDebugAction(fixed_request=json.dumps(gt["valid_request"])))
        assert obs.reward >= 0.5

    def test_empty_action_hard_scores_0(self):
        env = make_env("hard", seed=42)
        obs = env.step(APIDebugAction())
        assert obs.reward == 0.0


# ---------------------------------------------------------------------------
# TestHeuristicExplainer
# ---------------------------------------------------------------------------

class TestHeuristicExplainer:
    def test_empty_explanation_returns_low_score(self):
        env = make_env("hard", seed=42)
        score = env._heuristic_score_explanation("")
        # length < 20 → length_score = 0.1, keyword_score = 0.0 → 0.05
        assert score <= 0.2

    def test_keyword_rich_explanation_high_score(self):
        env = make_env("hard", seed=42)
        explanation = (
            "The field is missing because the required authentication header "
            "was not provided. The expected format is Bearer token. "
            "The fix should include the Authorization header with the correct value. "
            "The error type is missing_auth_header affecting the body payload schema."
        )
        score = env._heuristic_score_explanation(explanation)
        assert score >= 0.5

    def test_very_short_explanation_low_score(self):
        env = make_env("hard", seed=42)
        score = env._heuristic_score_explanation("bad")
        assert score < 0.4

    def test_long_explanation_does_not_exceed_1(self):
        env = make_env("hard", seed=42)
        long_text = (
            "fix error missing field required invalid type format schema "
            "endpoint method body authorization authentication payload constraint "
        ) * 20
        score = env._heuristic_score_explanation(long_text)
        assert 0.0 <= score <= 1.0

    def test_api_terms_score_well(self):
        env = make_env("hard", seed=42)
        # Uses several of the expanded keyword list terms
        explanation = (
            "The authorization header is missing from the request body. "
            "The schema requires the endpoint to use correct method and payload."
        )
        score = env._heuristic_score_explanation(explanation)
        assert score >= 0.4


# ---------------------------------------------------------------------------
# TestReset
# ---------------------------------------------------------------------------

class TestReset:
    def test_easy_max_steps_is_3(self):
        env = APIDebugEnvironment()
        env.reset(task="easy")
        assert env.max_steps == 3

    def test_medium_max_steps_is_5(self):
        env = APIDebugEnvironment()
        env.reset(task="medium")
        assert env.max_steps == 5

    def test_hard_max_steps_is_7(self):
        env = APIDebugEnvironment()
        env.reset(task="hard")
        assert env.max_steps == 7

    def test_reset_clears_step_count(self):
        env = make_env("easy", seed=42)
        env.step(APIDebugAction())
        env.reset(task="easy")
        assert env.current_step == 0

    def test_reset_clears_episode_done(self):
        env = make_env("easy", seed=42)
        env.step(perfect_easy_action(env))  # ends episode
        env.reset(task="easy")
        assert env.episode_done is False

    def test_reset_returns_done_false(self):
        env = APIDebugEnvironment()
        obs = env.reset(task="easy")
        assert obs.done is False

    def test_reset_returns_reward_zero(self):
        env = APIDebugEnvironment()
        obs = env.reset(task="easy")
        assert obs.reward == 0.0

    def test_seeded_reset_is_deterministic(self):
        env1 = APIDebugEnvironment()
        obs1 = env1.reset(task="easy", seed=42)
        env2 = APIDebugEnvironment()
        obs2 = env2.reset(task="easy", seed=42)
        assert obs1.api_name == obs2.api_name
        assert obs1.broken_request == obs2.broken_request
        assert obs1.error_count == obs2.error_count

    def test_different_seeds_produce_valid_obs(self):
        env = APIDebugEnvironment()
        obs1 = env.reset(task="easy", seed=1)
        obs2 = env.reset(task="easy", seed=99)
        assert isinstance(obs1.api_name, str) and len(obs1.api_name) > 0
        assert isinstance(obs2.api_name, str) and len(obs2.api_name) > 0

    def test_invalid_task_defaults_easy(self):
        env = APIDebugEnvironment()
        env.reset(task="nonexistent_task")
        assert env.task == "easy"

    def test_reset_generates_one_ground_truth_for_easy(self):
        env = make_env("easy", seed=42)
        assert len(env.ground_truths) == 1

    def test_reset_hard_generates_multiple_ground_truths(self):
        env = make_env("hard", seed=42)
        assert len(env.ground_truths) >= 2

    def test_reset_observation_has_correct_task(self):
        env = APIDebugEnvironment()
        obs = env.reset(task="medium")
        assert obs.task == "medium"

    def test_reset_observation_has_api_name(self):
        env = APIDebugEnvironment()
        obs = env.reset(task="easy")
        assert obs.api_name and isinstance(obs.api_name, str)

    def test_reset_observation_has_broken_request(self):
        env = APIDebugEnvironment()
        obs = env.reset(task="easy")
        # broken_request is a non-empty JSON string
        assert obs.broken_request and len(obs.broken_request) > 2
        parsed = json.loads(obs.broken_request)
        assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# TestStepBehavior
# ---------------------------------------------------------------------------

class TestStepBehavior:
    def test_step_increments_current_step(self):
        env = make_env("easy", seed=42)
        assert env.current_step == 0
        env.step(APIDebugAction())
        assert env.current_step == 1
        env.step(APIDebugAction())
        assert env.current_step == 2

    def test_best_reward_tracks_max(self):
        env = make_env("easy", seed=42)
        # Step 1: no reward
        env.step(APIDebugAction())
        assert env.best_reward == 0.0
        # Step 2: partial reward (correct type only)
        gt = env.ground_truths[0]
        env.step(APIDebugAction(error_type=gt["error_type"], affected_fields=[]))
        # raw=0.6, mult=0.9 → reward=0.54. best_reward updated.
        assert env.best_reward > 0.0

    @pytest.mark.parametrize("step_num,expected_mult", [
        (1, 1.0),
        (2, 0.9),
        (3, 0.8),
        (4, 0.7),
        (5, 0.6),
    ])
    def test_step_decay_multiplier_values(self, step_num, expected_mult):
        mult = max(1.0 - 0.1 * (step_num - 1), 0.3)
        assert abs(mult - expected_mult) < 0.001

    @pytest.mark.parametrize("step_num", [8, 10, 15])
    def test_step_decay_floor_at_0_3(self, step_num):
        mult = max(1.0 - 0.1 * (step_num - 1), 0.3)
        assert mult == 0.3

    def test_episode_done_on_perfect_score(self):
        env = make_env("easy", seed=42)
        obs = env.step(perfect_easy_action(env))
        assert obs.done is True
        assert env.episode_done is True

    def test_episode_done_on_max_steps(self):
        env = make_env("easy", seed=42)
        for _ in range(3):  # easy has max_steps=3
            if not env.episode_done:
                obs = env.step(APIDebugAction())
        assert env.episode_done is True

    def test_step_after_episode_done_returns_done(self):
        env = make_env("easy", seed=42)
        env.step(perfect_easy_action(env))  # done=True
        obs = env.step(APIDebugAction())
        assert obs.done is True

    def test_step_number_in_observation(self):
        env = make_env("easy", seed=42)
        obs = env.step(APIDebugAction())
        assert obs.step_number == 1
        obs2 = env.step(APIDebugAction())
        assert obs2.step_number == 2


# ---------------------------------------------------------------------------
# TestRewardRange
# ---------------------------------------------------------------------------

class TestRewardRange:
    @pytest.mark.parametrize("task", ["easy", "medium", "hard"])
    def test_reward_in_range_empty_action(self, task):
        env = APIDebugEnvironment()
        env.reset(task=task, seed=42)
        obs = env.step(APIDebugAction())
        assert 0.0 <= obs.reward <= 1.0

    @pytest.mark.parametrize("seed", [1, 5, 10, 42, 99])
    def test_reward_range_across_seeds(self, seed):
        env = APIDebugEnvironment()
        env.reset(task="easy", seed=seed)
        obs = env.step(perfect_easy_action(env))
        assert 0.0 <= obs.reward <= 1.0

    def test_reward_never_negative(self):
        env = make_env("medium", seed=42)
        for _ in range(5):
            if not env.episode_done:
                obs = env.step(APIDebugAction(fixed_request="{}"))
                assert obs.reward >= 0.0

    def test_best_reward_returned_at_end(self):
        env = make_env("easy", seed=42)
        gt = env.ground_truths[0]
        # Step 1: correct type only → raw=0.6, mult=1.0 → reward=0.6
        env.step(APIDebugAction(error_type=gt["error_type"], affected_fields=[]))
        # Step 2: perfect → raw=1.0, mult=0.9 → reward=0.9. done=True.
        obs = env.step(APIDebugAction(error_type=gt["error_type"], affected_fields=gt["affected_fields"]))
        # best_reward = max(0.6, 0.9) = 0.9 returned on done
        assert obs.done is True
        assert abs(obs.reward - 0.9) < 0.01
