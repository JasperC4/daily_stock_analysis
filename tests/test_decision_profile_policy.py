from __future__ import annotations

from src.services.decision_profile_policy import DecisionSignalCandidate, apply_decision_profile_policy


def test_policy_keeps_valid_snapshot_action_without_profile_upgrade() -> None:
    result = apply_decision_profile_policy(
        DecisionSignalCandidate(
            action="hold",
            score=52,
            confidence=0.6,
            horizon=None,
            market_phase=None,
        ),
        decision_profile="aggressive",
        data_quality_level="medium",
    )

    assert result.candidate.action == "hold"
    assert result.candidate.horizon == "3d"
    assert result.guardrail_result.passed is True
    assert result.guardrail_result.adjusted is False


def test_policy_safely_downgrades_buy_with_missing_confidence() -> None:
    result = apply_decision_profile_policy(
        DecisionSignalCandidate(
            action="buy",
            score=70,
            confidence=None,
            horizon="3d",
            stop_loss=10,
            target_price=15,
        ),
        decision_profile="balanced",
        data_quality_level="medium",
    )

    assert result.guardrail_result.raw_action == "buy"
    assert result.guardrail_result.final_action == "watch"
    assert result.candidate.action == "watch"
    assert result.guardrail_result.passed is True
    assert result.guardrail_result.adjusted is True
    assert "missing_confidence" in result.guardrail_result.violations
    assert result.guardrail_result.adjustments
    assert result.blocked_reason is None
    assert {warning["code"] for warning in result.warnings} == {"action_adjusted_by_guardrail"}
    assert all(warning.get("message") for warning in result.warnings)


def test_policy_requires_explicit_invalidation_for_aggressive_buy() -> None:
    result = apply_decision_profile_policy(
        DecisionSignalCandidate(
            action="buy",
            score=70,
            confidence=0.7,
            horizon="3d",
            stop_loss=10,
            target_price=15,
        ),
        decision_profile="aggressive",
        data_quality_level="medium",
    )

    assert result.candidate.action == "watch"
    assert "aggressive_missing_explicit_invalidation" in result.guardrail_result.violations


def test_policy_blocks_aggressive_buy_with_long_horizon_without_silent_cap() -> None:
    result = apply_decision_profile_policy(
        DecisionSignalCandidate(
            action="buy",
            confidence=0.7,
            horizon="long",
            invalidation="跌破趋势线",
            stop_loss=10,
            target_price=15,
        ),
        decision_profile="aggressive",
        data_quality_level="medium",
    )

    assert result.candidate.action == "watch"
    assert result.candidate.horizon == "long"
    assert "aggressive_horizon_long_not_allowed" in result.guardrail_result.violations


def test_policy_records_price_relationship_violations() -> None:
    result = apply_decision_profile_policy(
        DecisionSignalCandidate(
            action="add",
            confidence=0.7,
            horizon="3d",
            invalidation="跌破趋势线",
            entry_low=20,
            entry_high=18,
            stop_loss=19,
            target_price=17,
        ),
        decision_profile="balanced",
        data_quality_level="medium",
    )

    assert result.guardrail_result.final_action == "alert"
    assert result.guardrail_result.adjusted is True
    assert result.guardrail_result.passed is False
    assert "entry_range_invalid" in result.guardrail_result.violations
    assert "stop_loss_not_below_target_price" in result.guardrail_result.violations
    assert result.blocked_reason
    assert {warning["code"] for warning in result.warnings} == {
        "action_adjusted_by_guardrail",
        "action_blocked_by_guardrail",
    }
    assert all(warning.get("message") for warning in result.warnings)


def _plan_candidate(**overrides) -> DecisionSignalCandidate:
    base = {
        "action": "buy",
        "score": 70,
        "confidence": 0.8,
        "horizon": "3d",
        "entry_low": 98.0,
        "entry_high": 100.0,
        "stop_loss": 95.0,
        "target_price": 112.0,
    }
    base.update(overrides)
    return DecisionSignalCandidate(**base)


def _apply(candidate: DecisionSignalCandidate, **kwargs):
    return apply_decision_profile_policy(
        candidate,
        decision_profile="balanced",
        data_quality_level="medium",
        **kwargs,
    )


def test_min_reward_risk_is_disabled_by_default() -> None:
    # Reward 12 / risk 5 = 2.4, but a thin plan must still pass untouched.
    thin = _plan_candidate(target_price=101.0)
    for kwargs in ({}, {"min_reward_risk": None}, {"min_reward_risk": 0}):
        result = _apply(thin, **kwargs)
        assert result.guardrail_result.final_action == "buy"
        assert result.guardrail_result.violations == []
        assert "min_reward_risk" not in result.scoring_breakdown


def test_min_reward_risk_downgrades_thin_plan_to_watch() -> None:
    # Reward 6 / risk 5 = 1.2 against a 2.0 minimum.
    result = _apply(_plan_candidate(target_price=106.0), min_reward_risk=2.0)

    assert result.guardrail_result.violations == ["reward_risk_below_minimum"]
    assert result.guardrail_result.final_action == "watch"
    assert result.guardrail_result.passed is True
    assert result.guardrail_result.adjusted is True
    assert result.blocked_reason is None
    assert result.scoring_breakdown["min_reward_risk"] == 2.0
    assert {warning["code"] for warning in result.warnings} == {"action_adjusted_by_guardrail"}


def test_min_reward_risk_accepts_ratio_at_the_minimum_and_above() -> None:
    # Reward 10 / risk 5 is exactly 2.0; 12 / 5 is 2.4.
    for target in (110.0, 112.0):
        result = _apply(_plan_candidate(target_price=target), min_reward_risk=2.0)
        assert result.guardrail_result.final_action == "buy"
        assert result.guardrail_result.violations == []
        assert result.scoring_breakdown["min_reward_risk"] == 2.0


def test_min_reward_risk_measures_from_the_top_of_the_entry_range() -> None:
    # From entry_low 90 the ratio would be (112-90)/(90-95) -> not meaningful;
    # from entry_high 100 it is 2.4. Widening the range downward cannot help.
    result = _apply(_plan_candidate(entry_low=90.0), min_reward_risk=2.0)
    assert result.guardrail_result.final_action == "buy"


def test_min_reward_risk_requires_price_levels_when_enabled() -> None:
    for missing in ("entry_high", "stop_loss", "target_price"):
        result = _apply(
            _plan_candidate(**{missing: None, "invalidation": "close below 95"}),
            min_reward_risk=2.0,
        )
        assert "reward_risk_inputs_missing" in result.guardrail_result.violations
        assert result.guardrail_result.final_action == "watch"
        assert result.guardrail_result.passed is True


def test_min_reward_risk_does_not_double_report_a_stop_above_entry() -> None:
    result = _apply(_plan_candidate(stop_loss=101.0), min_reward_risk=2.0)

    assert "stop_loss_not_below_entry_high" in result.guardrail_result.violations
    assert "reward_risk_below_minimum" not in result.guardrail_result.violations
    assert "reward_risk_inputs_missing" not in result.guardrail_result.violations
    # Contradictory prices keep their existing blocked semantics.
    assert result.guardrail_result.final_action == "alert"
    assert result.guardrail_result.passed is False


def test_min_reward_risk_only_applies_to_buy_and_add() -> None:
    thin = {"target_price": 101.0}
    for action in ("hold", "watch", "reduce", "sell"):
        result = _apply(_plan_candidate(action=action, **thin), min_reward_risk=2.0)
        assert result.guardrail_result.final_action == action
        assert result.guardrail_result.violations == []
    assert _apply(_plan_candidate(action="add", **thin), min_reward_risk=2.0).candidate.action == "watch"


def test_decision_min_reward_risk_env_parsing() -> None:
    import os
    from unittest.mock import patch

    from src.config import Config

    def load(env: dict[str, str]) -> float:
        with patch("src.config.setup_env"), patch.object(Config, "_parse_litellm_yaml", return_value=[]):
            with patch.dict(os.environ, env, clear=True):
                return Config._load_from_env().decision_min_reward_risk

    assert load({}) == 0.0
    assert load({"DECISION_MIN_REWARD_RISK": "2.5"}) == 2.5
    assert load({"DECISION_MIN_REWARD_RISK": " 2 "}) == 2.0
    # Non-numeric values fall back to disabled; out-of-range values are clamped
    # to 0-100 (the repository's parse_env_float convention) without failing startup.
    assert load({"DECISION_MIN_REWARD_RISK": "abc"}) == 0.0
    assert load({"DECISION_MIN_REWARD_RISK": "-1"}) == 0.0
    assert load({"DECISION_MIN_REWARD_RISK": "1000"}) == 100.0
