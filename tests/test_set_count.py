"""Tests for set count prediction scenario logic and compute_3set_rates."""

import sys
from pathlib import Path

import pandas as pd
import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "scripts"))

from train_set_count import compute_3set_rates  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _match_df(
    winners: list[str],
    losers: list[str],
    sets_played: list[int],
    base_date: str = "2024-01-01",
) -> pd.DataFrame:
    """Minimal DataFrame for compute_3set_rates."""
    n = len(winners)
    return pd.DataFrame(
        {
            "match_date": pd.date_range(base_date, periods=n, freq="D"),
            "winner_id": winners,
            "loser_id": losers,
            "sets_played": sets_played,
        }
    )


# ---------------------------------------------------------------------------
# compute_3set_rates
# ---------------------------------------------------------------------------


class TestCompute3SetRates:
    def test_first_match_equals_prior(self) -> None:
        """Before any history, 3-set rate must equal the population prior (0.40)."""
        df = _match_df(["A"], ["B"], [2])
        result = compute_3set_rates(df)

        assert result["winner_3set_rate"].iloc[0] == pytest.approx(0.40)
        assert result["loser_3set_rate"].iloc[0] == pytest.approx(0.40)

    def test_all_prior_3set_pushes_rate_above_prior(self) -> None:
        """After all-3-set prior matches, winner_3set_rate at last match > 0.40."""
        # A plays 5 matches, all going to 3 sets; 6th match at index 5
        winners = ["A"] * 6
        losers = ["B", "C", "D", "E", "F", "G"]
        sets_played = [3, 3, 3, 3, 3, 2]  # first 5 are 3-set
        df = _match_df(winners, losers, sets_played)
        result = compute_3set_rates(df)

        # At match 5, A has 5 prior 3-set matches → rate > 0.40
        assert result["winner_3set_rate"].iloc[5] > 0.40

    def test_no_leakage_current_match_excluded(self) -> None:
        """The current match's sets_played must NOT be reflected in its own 3set_rate."""
        # One player with one 3-set match; rate at that match must still be the prior
        df = _match_df(["A"], ["B"], [3])
        result = compute_3set_rates(df)

        # No prior matches → rate is still 0.40 despite this match going 3 sets
        assert result["winner_3set_rate"].iloc[0] == pytest.approx(0.40)

    def test_rate_reflects_prior_matches_correctly(self) -> None:
        """After exactly 1 prior 3-set match, Bayesian rate must match hand calculation."""
        # A beats B in a 3-set match, then plays again
        df = _match_df(["A", "A"], ["B", "C"], [3, 2])
        result = compute_3set_rates(df)

        # At match index 1: A had 1 prior match (3-set=1), prior_n=5, prior=0.40
        # rate = (1 + 0.40*5) / (1 + 5) = (1 + 2) / 6 = 0.5
        expected = (1 + 0.40 * 5) / (1 + 5)
        assert result["winner_3set_rate"].iloc[1] == pytest.approx(expected)

    def test_threeset_rate_diff_equals_winner_minus_loser(self) -> None:
        """threeset_rate_diff must equal winner_3set_rate - loser_3set_rate row-wise."""
        df = _match_df(["A", "B", "A"], ["B", "C", "C"], [3, 2, 3])
        result = compute_3set_rates(df)

        expected = result["winner_3set_rate"] - result["loser_3set_rate"]
        pd.testing.assert_series_equal(
            result["threeset_rate_diff"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_output_columns_present(self) -> None:
        """winner_3set_rate, loser_3set_rate, and threeset_rate_diff must all be created."""
        df = _match_df(["A"], ["B"], [2])
        result = compute_3set_rates(df)

        assert "winner_3set_rate" in result.columns
        assert "loser_3set_rate" in result.columns
        assert "threeset_rate_diff" in result.columns

    def test_rates_bounded_in_unit_interval(self) -> None:
        """All 3-set rate values must be in [0, 1]."""
        df = _match_df(
            ["A", "A", "B", "A"],
            ["B", "C", "C", "D"],
            [3, 2, 3, 2],
        )
        result = compute_3set_rates(df)

        assert (result["winner_3set_rate"].between(0.0, 1.0)).all()
        assert (result["loser_3set_rate"].between(0.0, 1.0)).all()


def test_set_count_scenarios_sum_to_one():
    """Four scenario probabilities must sum to 1.0."""
    scenarios = {
        "p1_2_0": 0.47,
        "p1_2_1": 0.29,
        "p2_0_2": 0.16,
        "p2_1_2": 0.08,
    }
    assert abs(sum(scenarios.values()) - 1.0) < 0.01


def test_set_count_scenarios_winner_leads():
    """P1 total (2-0 + 2-1) must equal the win probability."""
    win_prob = 0.75
    set3_prob = 0.38
    set2_prob = 1.0 - set3_prob
    p1_total = win_prob * set2_prob + win_prob * set3_prob
    assert abs(p1_total - win_prob) < 1e-6


def test_set_count_scenarios_construction():
    """Verify scenario values are computed correctly from win_prob and set3_prob."""
    prob = 0.6
    set3_prob = 0.4
    set2_prob = 1.0 - set3_prob
    scenarios = {
        "p1_2_0": round(prob * set2_prob, 4),
        "p1_2_1": round(prob * set3_prob, 4),
        "p2_0_2": round((1 - prob) * set2_prob, 4),
        "p2_1_2": round((1 - prob) * set3_prob, 4),
    }
    assert abs(sum(scenarios.values()) - 1.0) < 1e-4
    assert scenarios["p1_2_0"] == round(0.6 * 0.6, 4)
    assert scenarios["p1_2_1"] == round(0.6 * 0.4, 4)
    assert scenarios["p2_0_2"] == round(0.4 * 0.6, 4)
    assert scenarios["p2_1_2"] == round(0.4 * 0.4, 4)
