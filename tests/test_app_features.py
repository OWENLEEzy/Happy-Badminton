"""Tests for build_general_features() and predict-general endpoint."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_build_general_features_new_params():
    from frontend.app import build_general_features

    p1 = {
        "ranking": 5,
        "nationality": "CHN",
        "form5_wins": 4,
        "form10_wins": 8,
        "form20_wins": 15,
        "streak": 3,
        "career_matches": 200,
    }
    p2 = {
        "ranking": 20,
        "nationality": "KOR",
        "form5_wins": 3,
        "form10_wins": 7,
        "form20_wins": 12,
        "streak": -1,
        "career_matches": 150,
    }
    h2h = {"p1_wins": 3, "total": 7}

    feats = build_general_features(
        match_type="MS",
        tournament_level="S1000",
        round_stage=6,
        match_month=3,
        host_country="CHN",
        p1=p1,
        p2=p2,
        h2h=h2h,
    )
    assert "round_stage" in feats
    assert feats["round_stage"] == 6
    assert "match_month" in feats
    assert feats["match_month"] == 3
    assert "h2h_win_rate_bayes" in feats
    assert "loser_home" in feats
    assert "rank_closeness" in feats
    assert "momentum_diff" in feats
    assert "streak_capped_w" in feats
    assert "career_stage" in feats


def test_build_general_features_h2h_bayes():
    from frontend.app import build_general_features

    feats = build_general_features(
        match_type="MS",
        tournament_level="S300",
        round_stage=4,
        match_month=6,
        host_country="",
        p1={
            "ranking": 10,
            "form5_wins": 3,
            "form10_wins": 5,
            "form20_wins": 10,
            "streak": 1,
            "career_matches": 100,
        },
        p2={
            "ranking": 20,
            "form5_wins": 3,
            "form10_wins": 5,
            "form20_wins": 10,
            "streak": 1,
            "career_matches": 100,
        },
        h2h={"p1_wins": 1, "total": 1},
    )
    # (1 + 2.5) / (1 + 5) = 0.583
    assert abs(feats["h2h_win_rate_bayes"] - 0.583) < 0.01
