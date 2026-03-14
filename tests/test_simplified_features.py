"""Tests for compute_new_features() Phase-1 feature engineering."""

import sys
from pathlib import Path

import pandas as pd
import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.simplified_features import (
    ROUND_STAGE_DEFAULT,
    ROUND_STAGE_MAP,
    compute_new_features,
)


def _base_df(n: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "round": ["Round 1"] * n,
            "match_date": pd.to_datetime(["2024-06-01"] * n),
            "winner_id": [f"P{i}A" for i in range(n)],
            "loser_id": [f"P{i}B" for i in range(n)],
            "winner_streak": [3] * n,
            "loser_streak": [2] * n,
            "total_player_matches": [100] * n,
            "winner_form_5": [0.6] * n,
            "loser_form_5": [0.4] * n,
            "winner_form_10": [0.55] * n,
            "loser_form_10": [0.45] * n,
            "h2h_win_rate": [0.6] * n,
            "h2h_matches": [5] * n,
            "log_rank_diff": [1.0] * n,
            "category_flag": [0] * n,
            "winner_home": [1] * n,
            "form_diff_10": [0.1] * n,
        }
    )


def test_round_stage_mapping() -> None:
    """Known round names map to correct stage integers; unknown names fall back to default."""
    rows = pd.DataFrame(
        {
            "round": ["Q-Round 1", "Quarter-final", "Final", "Unknown Round"],
            "match_date": pd.to_datetime(["2024-01-01"] * 4),
            "winner_id": ["W1", "W2", "W3", "W4"],
            "loser_id": ["L1", "L2", "L3", "L4"],
            "winner_streak": [0] * 4,
            "loser_streak": [0] * 4,
            "total_player_matches": [50] * 4,
            "winner_form_5": [0.5] * 4,
            "loser_form_5": [0.5] * 4,
            "h2h_win_rate": [0.5] * 4,
            "h2h_matches": [0] * 4,
            "log_rank_diff": [0.0] * 4,
            "category_flag": [0] * 4,
            "winner_home": [0] * 4,
            "form_diff_10": [0.0] * 4,
        }
    )
    out = compute_new_features(rows)
    stages = out["round_stage"].tolist()
    assert stages[0] == ROUND_STAGE_MAP["Q-Round 1"], "Q-Round 1 should map to 0"
    assert stages[1] == ROUND_STAGE_MAP["Quarter-final"], "Quarter-final should map to 6"
    assert stages[2] == ROUND_STAGE_MAP["Final"], "Final should map to 8"
    assert stages[3] == ROUND_STAGE_DEFAULT, "Unknown round should fall back to default (4)"


def test_streak_capped() -> None:
    """Streaks beyond ±5 are capped at ±5 while preserving sign."""
    df = _base_df()
    df["winner_streak"] = 10
    df["loser_streak"] = -8
    out = compute_new_features(df)
    assert out["streak_capped_w"].iloc[0] == 5, "Positive streak 10 should be capped to 5"
    assert out["streak_capped_l"].iloc[0] == -5, "Negative streak -8 should be capped to -5"


def test_career_stage_u_curve() -> None:
    """Career stage follows the U-curve: 75 matches -> 2.0, 600 matches -> 0.0."""
    df_75 = _base_df()
    df_75["total_player_matches"] = 75
    df_600 = _base_df()
    df_600["total_player_matches"] = 600

    out_75 = compute_new_features(df_75)
    out_600 = compute_new_features(df_600)

    assert out_75["career_stage"].iloc[0] == pytest.approx(2.0), (
        "75 matches should give career_stage=2.0"
    )
    assert out_600["career_stage"].iloc[0] == pytest.approx(0.0), (
        "600 matches should give career_stage=0.0"
    )


def test_h2h_bayes_smoothing() -> None:
    """Bayesian H2H smoothing: 1 win from 1 match -> (1+2.5)/(1+5) ≈ 0.5833."""
    df = _base_df()
    df["h2h_win_rate"] = 1.0  # 1 win from 1 match
    df["h2h_matches"] = 1
    out = compute_new_features(df)
    expected = (1.0 + 5 * 0.5) / (1 + 5)  # (1 + 2.5) / (1 + 5) = 3.5 / 6
    assert out["h2h_win_rate_bayes"].iloc[0] == pytest.approx(expected, rel=1e-4), (
        f"Expected Bayesian H2H ≈ {expected:.4f}"
    )


def test_rank_closeness_equal_ranks() -> None:
    """When log_rank_diff=0, rank_closeness must equal exactly 1.0."""
    df = _base_df()
    df["log_rank_diff"] = 0.0
    out = compute_new_features(df)
    assert out["rank_closeness"].iloc[0] == pytest.approx(1.0), (
        "Equal ranks (log_rank_diff=0) should produce rank_closeness=1.0"
    )


def test_match_month() -> None:
    """match_month is extracted correctly from match_date."""
    df = _base_df()
    df["match_date"] = pd.to_datetime(["2024-06-01"])
    out = compute_new_features(df)
    assert out["match_month"].iloc[0] == 6, "June date should yield month=6"


def test_momentum_diff() -> None:
    """form_momentum_w = winner_form_5 - winner_form_10; momentum_diff = w - l."""
    df = _base_df()
    # winner: form5=0.6, form10=0.55 -> momentum_w = 0.05
    # loser:  form5=0.4, form10=0.45 -> momentum_l = -0.05
    # momentum_diff = 0.05 - (-0.05) = 0.10
    out = compute_new_features(df)
    assert out["form_momentum_w"].iloc[0] == pytest.approx(0.05, abs=1e-9), (
        "form_momentum_w should be 0.6 - 0.55 = 0.05"
    )
    assert out["momentum_diff"].iloc[0] == pytest.approx(0.10, abs=1e-9), (
        "momentum_diff should be 0.05 - (-0.05) = 0.10"
    )
