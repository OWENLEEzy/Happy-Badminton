"""Tests for MomentumFeatures and FatigueFeatures in src/data/advanced_features.py."""

import sys
from pathlib import Path

import pandas as pd
import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.advanced_features import FatigueFeatures, MomentumFeatures


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _match_df(pairs: list[tuple[str, str]], base_date: str = "2024-01-01") -> pd.DataFrame:
    """Build a DataFrame with winner_id/loser_id pairs on consecutive days."""
    n = len(pairs)
    return pd.DataFrame(
        {
            "match_date": pd.date_range(base_date, periods=n, freq="D"),
            "winner_id": [p[0] for p in pairs],
            "loser_id": [p[1] for p in pairs],
            "duration": [50.0] * n,
            "sets_played": [2] * n,
        }
    )


# ---------------------------------------------------------------------------
# MomentumFeatures.add_form_features
# ---------------------------------------------------------------------------


class TestAddFormFeatures:
    def test_form_first_match_is_prior(self) -> None:
        """First match for a player must give form = 0.5 (prior, no history)."""
        df = _match_df([("A", "B"), ("C", "D")])
        mf = MomentumFeatures(df)
        result = mf.add_form_features(windows=[5])

        # Both players in row 0 appear for the first time -> form = 0.5
        assert result["winner_form_5"].iloc[0] == pytest.approx(0.5)
        assert result["loser_form_5"].iloc[0] == pytest.approx(0.5)

    def test_form_below_min_periods_stays_at_prior(self) -> None:
        """Form stays 0.5 until a player has at least 3 prior appearances (min_periods=3)."""
        # A beats B twice, then A vs C. A has 2 prior results at 3rd match.
        df = _match_df([("A", "B"), ("A", "B"), ("A", "C")])
        mf = MomentumFeatures(df)
        result = mf.add_form_features(windows=[5])

        # At row 2: A has 2 prior appearances -> rolling(5, min_periods=3) still NaN -> 0.5
        assert result["winner_form_5"].iloc[2] == pytest.approx(0.5)

    def test_form_with_enough_history(self) -> None:
        """After 3+ prior wins, winner_form_5 must be > 0.5."""
        # A beats different opponents 6 times, giving A 5+ prior wins before match 6.
        pairs = [("A", "B"), ("A", "C"), ("A", "D"), ("A", "E"), ("A", "F"), ("A", "G")]
        df = _match_df(pairs)
        mf = MomentumFeatures(df)
        result = mf.add_form_features(windows=[5])

        # At row 5: A has 5 prior wins -> form_5 should reflect those 5 wins = 1.0
        assert result["winner_form_5"].iloc[5] > 0.5

    def test_form_diff_equals_winner_minus_loser(self) -> None:
        """form_diff_5 must equal winner_form_5 - loser_form_5 for every row."""
        pairs = [("A", "B")] * 4 + [("B", "A")] * 4
        df = _match_df(pairs)
        mf = MomentumFeatures(df)
        result = mf.add_form_features(windows=[5])

        expected = result["winner_form_5"] - result["loser_form_5"]
        pd.testing.assert_series_equal(
            result["form_diff_5"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_form_columns_all_present(self) -> None:
        """All nine form/diff columns must exist after add_form_features()."""
        df = _match_df([("A", "B")] * 5)
        mf = MomentumFeatures(df)
        result = mf.add_form_features(windows=[5, 10, 20])

        expected_cols = [
            "winner_form_5",
            "loser_form_5",
            "form_diff_5",
            "winner_form_10",
            "loser_form_10",
            "form_diff_10",
            "winner_form_20",
            "loser_form_20",
            "form_diff_20",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# MomentumFeatures.add_streak_features
# ---------------------------------------------------------------------------


class TestAddStreakFeatures:
    def test_streak_first_match_zero(self) -> None:
        """On the first match, winner_streak must be 0 (no prior wins)."""
        df = _match_df([("A", "B"), ("A", "C")])
        mf = MomentumFeatures(df)
        result = mf.add_streak_features()

        assert result["winner_streak"].iloc[0] == 0

    def test_streak_accumulates(self) -> None:
        """After winning 3 consecutive matches, winner_streak at match 4 should be 3."""
        pairs = [("A", "B"), ("A", "C"), ("A", "D"), ("A", "E")]
        df = _match_df(pairs)
        mf = MomentumFeatures(df)
        result = mf.add_streak_features()

        # At match index 3, A has won 3 prior matches in a row
        assert result["winner_streak"].iloc[3] == 3

    def test_loser_streak_negative(self) -> None:
        """After B loses 2 consecutive matches, loser_streak should be -2 on the 3rd loss."""
        # B loses to different opponents 3 times in a row
        pairs = [("A", "B"), ("C", "B"), ("D", "B")]
        df = _match_df(pairs)
        mf = MomentumFeatures(df)
        result = mf.add_streak_features()

        # At match index 2: B has lost 2 prior matches -> loser_streak = -2
        assert result["loser_streak"].iloc[2] == -2

    def test_streak_resets_on_role_change(self) -> None:
        """After A wins 3 then loses, A's win streak is 3 at the loss match; then resets."""
        # A wins 3, then loses in match 4
        pairs = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "A")]
        df = _match_df(pairs)
        mf = MomentumFeatures(df)
        result = mf.add_streak_features()

        # At match index 3: A (now loser) had 3 consecutive wins -> recorded as winner_streak of B = 0
        # But A as loser: -loss_streaks[A] at this point is 0 (A hadn't lost yet)
        # The winner (B) has 0 wins prior, so winner_streak for match 3 = 0
        # A's loser_streak at match 3 = -loss_streaks["A"] before update = 0
        assert result["winner_streak"].iloc[3] == 0  # B's prior win streak = 0
        assert result["loser_streak"].iloc[3] == 0  # A had no prior loss streak

    def test_streak_diff_column(self) -> None:
        """streak_diff must equal winner_streak - loser_streak row-wise."""
        pairs = [("A", "B"), ("A", "B"), ("B", "A"), ("A", "B")]
        df = _match_df(pairs)
        mf = MomentumFeatures(df)
        result = mf.add_streak_features()

        expected = result["winner_streak"] - result["loser_streak"]
        pd.testing.assert_series_equal(
            result["streak_diff"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_streak_columns_present(self) -> None:
        """winner_streak, loser_streak, streak_diff must all be present."""
        df = _match_df([("A", "B")] * 3)
        mf = MomentumFeatures(df)
        result = mf.add_streak_features()

        assert "winner_streak" in result.columns
        assert "loser_streak" in result.columns
        assert "streak_diff" in result.columns


# ---------------------------------------------------------------------------
# FatigueFeatures.compute_fatigue_features
# ---------------------------------------------------------------------------


class TestComputeFatigueFeatures:
    def test_fatigue_first_match_zero(self) -> None:
        """First match for both players must record 0 fatigue."""
        df = _match_df([("A", "B")])
        ff = FatigueFeatures(df)
        result = ff.compute_fatigue_features()

        assert result["winner_fatigue"].iloc[0] == pytest.approx(0.0)
        assert result["loser_fatigue"].iloc[0] == pytest.approx(0.0)

    def test_fatigue_accumulates_same_day(self) -> None:
        """Two matches on the same date: second match fatigue > 0 for returning players."""
        # A and B play twice on the same date (no decay between same-day matches)
        df = pd.DataFrame(
            {
                "match_date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
                "winner_id": ["A", "A"],
                "loser_id": ["B", "B"],
                "duration": [50.0, 50.0],
                "sets_played": [2, 2],
            }
        )
        ff = FatigueFeatures(df)
        result = ff.compute_fatigue_features()

        # First match: fatigue = 0; increment = 50*2/60 = 1.667
        # Second match (same day, no decay): fatigue should equal increment from first match
        expected_increment = 50.0 * 2 / 60.0
        assert result["winner_fatigue"].iloc[1] == pytest.approx(expected_increment)
        assert result["loser_fatigue"].iloc[1] == pytest.approx(expected_increment)

    def test_fatigue_decays_across_days(self) -> None:
        """Fatigue at the second match (next day) must be lower than the raw increment."""
        df = pd.DataFrame(
            {
                "match_date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
                "winner_id": ["A", "A"],
                "loser_id": ["B", "B"],
                "duration": [60.0, 60.0],
                "sets_played": [2, 2],
            }
        )
        ff = FatigueFeatures(df)
        result = ff.compute_fatigue_features()

        # Raw increment from match 1: 60*2/60 = 2.0
        # After 1 day decay (weight_factor=0.7): fatigue_at_match2 = 2.0 * 0.7^1 = 1.4
        raw_increment = 60.0 * 2 / 60.0
        expected_fatigue = raw_increment * (0.7**1)
        assert result["winner_fatigue"].iloc[1] == pytest.approx(expected_fatigue)

    def test_fatigue_diff_column(self) -> None:
        """fatigue_diff must equal winner_fatigue - loser_fatigue row-wise."""
        # Use different players so they accumulate different fatigue histories
        pairs = [("A", "B"), ("C", "B"), ("A", "D")]
        df = _match_df(pairs)
        ff = FatigueFeatures(df)
        result = ff.compute_fatigue_features()

        expected = result["winner_fatigue"] - result["loser_fatigue"]
        pd.testing.assert_series_equal(
            result["fatigue_diff"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_fatigue_columns_present(self) -> None:
        """winner_fatigue, loser_fatigue, and fatigue_diff must all be present."""
        df = _match_df([("A", "B")] * 3)
        ff = FatigueFeatures(df)
        result = ff.compute_fatigue_features()

        assert "winner_fatigue" in result.columns
        assert "loser_fatigue" in result.columns
        assert "fatigue_diff" in result.columns
