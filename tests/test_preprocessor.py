"""Tests for src/data/preprocessor.py DataPreprocessor class."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocessor import DataPreprocessor


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _base_df(n: int = 5) -> pd.DataFrame:
    """Minimal match DataFrame satisfying preprocessor column requirements."""
    return pd.DataFrame(
        {
            "match_date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "score": ["21-11 / 21-18"] * n,
            "duration": [50.0] * n,
            "winner_rank": [5.0] * n,
            "loser_rank": [10.0] * n,
            "type": ["MS"] * n,
            "winner_points": [21] * n,
            "loser_points": [15] * n,
            "sets_played": [2] * n,
            "seconds_per_point": [30.0] * n,
        }
    )


# ---------------------------------------------------------------------------
# filter_future_dates
# ---------------------------------------------------------------------------


class TestFilterFutureDates:
    def test_filter_future_dates_removes_future_rows(self) -> None:
        """Rows with match_date >= cutoff_date must be removed."""
        df = pd.DataFrame(
            {
                "match_date": [
                    pd.Timestamp("2024-01-01"),
                    pd.Timestamp("2024-01-02"),
                    pd.Timestamp("2030-06-01"),  # future
                ],
                "score": ["21-11 / 21-18"] * 3,
                "duration": [50.0] * 3,
                "winner_rank": [5.0] * 3,
                "loser_rank": [10.0] * 3,
                "type": ["MS"] * 3,
                "winner_points": [21] * 3,
                "loser_points": [15] * 3,
                "sets_played": [2] * 3,
                "seconds_per_point": [30.0] * 3,
            }
        )
        cutoff = pd.Timestamp("2025-01-01")
        preprocessor = DataPreprocessor(df)
        preprocessor.filter_future_dates(cutoff_date=cutoff)

        assert len(preprocessor.df) == 2

    def test_filter_future_dates_all_past_unchanged(self) -> None:
        """When all rows are before cutoff, nothing is removed."""
        df = _base_df(4)
        cutoff = pd.Timestamp("2099-01-01")
        preprocessor = DataPreprocessor(df)
        preprocessor.filter_future_dates(cutoff_date=cutoff)

        assert len(preprocessor.df) == 4


# ---------------------------------------------------------------------------
# identify_and_filter_retirements
# ---------------------------------------------------------------------------


class TestIdentifyAndFilterRetirements:
    def test_retirements_flagged_and_removed(self) -> None:
        """Row with 'Ret.' in score is removed; stats count equals 1."""
        scores = ["21-11 / 21-18", "21-14 / 21-10", "22-20 / 21-18", "21-14 Ret."]
        df = _base_df(4)
        df["score"] = scores

        preprocessor = DataPreprocessor(df)
        preprocessor.identify_and_filter_retirements()

        assert len(preprocessor.df) == 3
        assert preprocessor.stats["retirement_matches"] == 1

    def test_retirements_walkover_detected(self) -> None:
        """Score 'W.O.' is treated as a retirement and removed."""
        df = _base_df(3)
        df.loc[1, "score"] = "W.O."

        preprocessor = DataPreprocessor(df)
        preprocessor.identify_and_filter_retirements()

        assert preprocessor.stats["retirement_matches"] == 1
        assert len(preprocessor.df) == 2

    def test_no_retirements_all_remain(self) -> None:
        """When all scores are normal, nothing is removed."""
        df = _base_df(5)  # default score is "21-11 / 21-18"

        preprocessor = DataPreprocessor(df)
        preprocessor.identify_and_filter_retirements()

        assert preprocessor.stats["retirement_matches"] == 0
        assert len(preprocessor.df) == 5


# ---------------------------------------------------------------------------
# handle_duration_outliers
# ---------------------------------------------------------------------------


class TestHandleDurationOutliers:
    def test_duration_extreme_capped_at_max(self) -> None:
        """Duration > 150 is capped to 150."""
        df = _base_df(3)
        df.loc[1, "duration"] = 200.0

        preprocessor = DataPreprocessor(df)
        preprocessor.handle_duration_outliers()

        assert preprocessor.df.loc[1, "duration"] == pytest.approx(150.0)

    def test_duration_zero_becomes_nan(self) -> None:
        """Duration == 0 is converted to NaN."""
        df = _base_df(3)
        df.loc[0, "duration"] = 0.0

        preprocessor = DataPreprocessor(df)
        preprocessor.handle_duration_outliers()

        assert pd.isna(preprocessor.df.loc[0, "duration"])

    def test_duration_normal_unchanged(self) -> None:
        """Duration within valid range is not modified."""
        df = _base_df(3)
        # All durations are 50.0 by default

        preprocessor = DataPreprocessor(df)
        preprocessor.handle_duration_outliers()

        assert preprocessor.df["duration"].eq(50.0).all()


# ---------------------------------------------------------------------------
# parse_scores
# ---------------------------------------------------------------------------


class TestParseScores:
    def test_parse_scores_total_points(self) -> None:
        """Score '21-11 / 21-18' should give total_points = 71."""
        df = _base_df(1)
        df.loc[0, "score"] = "21-11 / 21-18"

        preprocessor = DataPreprocessor(df)
        preprocessor.parse_scores()

        assert preprocessor.df.loc[0, "total_points"] == 71

    def test_parse_scores_sets_played_two(self) -> None:
        """Score '21-11 / 21-18' should give sets_played = 2."""
        df = _base_df(1)
        df.loc[0, "score"] = "21-11 / 21-18"

        preprocessor = DataPreprocessor(df)
        preprocessor.parse_scores()

        assert preprocessor.df.loc[0, "sets_played"] == 2

    def test_parse_scores_sets_played_three(self) -> None:
        """Score '21-6 / 20-22 / 21-13' should give sets_played = 3."""
        df = _base_df(1)
        df.loc[0, "score"] = "21-6 / 20-22 / 21-13"

        preprocessor = DataPreprocessor(df)
        preprocessor.parse_scores()

        assert preprocessor.df.loc[0, "sets_played"] == 3

    def test_parse_scores_seconds_per_point(self) -> None:
        """With duration=71 min and total_points=71, seconds_per_point = 60.0."""
        df = _base_df(1)
        # 21+11+21+18 = 71 total points
        df.loc[0, "score"] = "21-11 / 21-18"
        df.loc[0, "duration"] = 71.0

        preprocessor = DataPreprocessor(df)
        preprocessor.parse_scores()

        # seconds_per_point = duration * 60 / total_points = 71 * 60 / 71 = 60.0
        assert preprocessor.df.loc[0, "seconds_per_point"] == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# handle_missing_values
# ---------------------------------------------------------------------------


class TestHandleMissingValues:
    def test_missing_rank_filled_with_999(self) -> None:
        """NaN winner_rank and loser_rank are both filled with 999."""
        df = _base_df(3)
        df.loc[1, "winner_rank"] = np.nan
        df.loc[2, "loser_rank"] = np.nan

        preprocessor = DataPreprocessor(df)
        preprocessor.handle_missing_values()

        assert preprocessor.df.loc[1, "winner_rank"] == pytest.approx(999.0)
        assert preprocessor.df.loc[2, "loser_rank"] == pytest.approx(999.0)

    def test_missing_duration_filled_with_type_median(self) -> None:
        """NaN MS duration is replaced with the median of non-NaN MS rows."""
        df = pd.DataFrame(
            {
                "match_date": pd.date_range("2024-01-01", periods=5, freq="D"),
                "score": ["21-11 / 21-18"] * 5,
                "duration": [60.0, 80.0, np.nan, 70.0, 40.0],
                "winner_rank": [5.0] * 5,
                "loser_rank": [10.0] * 5,
                "type": ["MS"] * 5,
                "winner_points": [21] * 5,
                "loser_points": [15] * 5,
                "sets_played": [2] * 5,
                "seconds_per_point": [30.0] * 5,
            }
        )
        # Non-NaN MS durations: 60, 80, 70, 40 -> median = (60+70)/2 = 65.0
        expected_median = float(np.median([60.0, 80.0, 70.0, 40.0]))

        preprocessor = DataPreprocessor(df)
        preprocessor.handle_missing_values()

        assert preprocessor.df.loc[2, "duration"] == pytest.approx(expected_median)


# ---------------------------------------------------------------------------
# sort_by_date
# ---------------------------------------------------------------------------


class TestSortByDate:
    def test_sort_by_date_ascending(self) -> None:
        """After sort_by_date(), match_date must be monotonically non-decreasing."""
        dates = pd.to_datetime(["2024-03-01", "2024-01-01", "2024-02-01"])
        df = pd.DataFrame(
            {
                "match_date": dates,
                "score": ["21-11 / 21-18"] * 3,
                "duration": [50.0] * 3,
                "winner_rank": [5.0] * 3,
                "loser_rank": [10.0] * 3,
                "type": ["MS"] * 3,
                "winner_points": [21] * 3,
                "loser_points": [15] * 3,
                "sets_played": [2] * 3,
                "seconds_per_point": [30.0] * 3,
            }
        )

        preprocessor = DataPreprocessor(df)
        preprocessor.sort_by_date()

        dates_after = preprocessor.df["match_date"].tolist()
        assert dates_after == sorted(dates_after)


# ---------------------------------------------------------------------------
# add_target_variable
# ---------------------------------------------------------------------------


class TestAddTargetVariable:
    def test_target_is_always_one(self) -> None:
        """All rows must have target == 1 after add_target_variable()."""
        df = _base_df(6)

        preprocessor = DataPreprocessor(df)
        preprocessor.add_target_variable()

        assert (preprocessor.df["target"] == 1).all()


# ---------------------------------------------------------------------------
# get_summary
# ---------------------------------------------------------------------------


class TestGetSummary:
    def test_get_summary_shape(self) -> None:
        """get_summary() must return a dict containing 'original_shape' and 'final_shape'."""
        df = _base_df(4)

        preprocessor = DataPreprocessor(df)
        summary = preprocessor.get_summary()

        assert "original_shape" in summary
        assert "final_shape" in summary
        assert summary["original_shape"] == df.shape

    def test_get_summary_contains_stats(self) -> None:
        """get_summary() must include a 'stats' key."""
        df = _base_df(3)

        preprocessor = DataPreprocessor(df)
        summary = preprocessor.get_summary()

        assert "stats" in summary
        assert isinstance(summary["stats"], dict)


# ---------------------------------------------------------------------------
# filter_future_dates — default cutoff (line 58)
# ---------------------------------------------------------------------------


class TestFilterFutureDatesDefaultCutoff:
    def test_no_argument_removes_far_future_rows(self) -> None:
        """Calling filter_future_dates() with no argument must remove dates far in the future."""
        df = pd.DataFrame(
            {
                "match_date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2030-01-01")],
                "score": ["21-11 / 21-18"] * 2,
                "duration": [50.0] * 2,
                "winner_rank": [5.0] * 2,
                "loser_rank": [10.0] * 2,
                "type": ["MS"] * 2,
                "winner_points": [21] * 2,
                "loser_points": [15] * 2,
                "sets_played": [2] * 2,
                "seconds_per_point": [30.0] * 2,
            }
        )
        preprocessor = DataPreprocessor(df)
        preprocessor.filter_future_dates()  # uses Timestamp.now() as default

        # 2020-01-01 is in the past; 2030-01-01 should be removed
        assert len(preprocessor.df) == 1
        assert preprocessor.df["match_date"].iloc[0] == pd.Timestamp("2020-01-01")


# ---------------------------------------------------------------------------
# preprocess_pipeline (orchestration)
# ---------------------------------------------------------------------------


class TestPreprocessPipeline:
    def test_returns_dataframe(self) -> None:
        """preprocess_pipeline() must return a pandas DataFrame."""
        from src.data.preprocessor import preprocess_pipeline

        df = _base_df(5)
        result = preprocess_pipeline(df)

        assert isinstance(result, pd.DataFrame)

    def test_adds_target_column(self) -> None:
        """Output must contain a 'target' column (added by add_target_variable)."""
        from src.data.preprocessor import preprocess_pipeline

        df = _base_df(5)
        result = preprocess_pipeline(df)

        assert "target" in result.columns
        assert (result["target"] == 1).all()

    def test_removes_future_rows_via_cutoff(self) -> None:
        """Rows whose match_date >= cutoff_date must be absent from the output."""
        from src.data.preprocessor import preprocess_pipeline

        df = _base_df(3)
        df.loc[2, "match_date"] = pd.Timestamp("2030-06-01")  # future row
        cutoff = pd.Timestamp("2025-01-01")
        result = preprocess_pipeline(df, cutoff_date=cutoff)

        assert len(result) == 2

    def test_retirement_rows_removed(self) -> None:
        """Rows with retirement scores must not appear in pipeline output."""
        from src.data.preprocessor import preprocess_pipeline

        df = _base_df(4)
        df.loc[3, "score"] = "21-14 Ret."
        result = preprocess_pipeline(df)

        assert len(result) == 3
