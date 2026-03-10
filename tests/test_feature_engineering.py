"""Tests for src/data/feature_engineering.py FeatureEngineer class."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.feature_engineering import FeatureEngineer


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _base_df(n: int = 5) -> pd.DataFrame:
    """Minimal DataFrame with all columns required by FeatureEngineer."""
    return pd.DataFrame(
        {
            "match_date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "winner_id": ["A"] * n,
            "loser_id": ["B"] * n,
            "winner_elo": [1700.0] * n,
            "loser_elo": [1600.0] * n,
            "winner_rank": [5.0] * n,
            "loser_rank": [10.0] * n,
            "winner_assoc": ["China"] * n,
            "loser_assoc": ["Japan"] * n,
            "country": ["China"] * n,
            "type": ["MS"] * n,
            "level": ["IS"] * n,
            "seconds_per_point": [30.0] * n,
            "duration": [50.0] * n,
            "sets_played": [2] * n,
        }
    )


# ---------------------------------------------------------------------------
# add_basic_features
# ---------------------------------------------------------------------------


class TestAddBasicFeatures:
    def test_elo_diff_correct(self) -> None:
        """elo_diff must equal winner_elo - loser_elo."""
        df = _base_df(3)
        eng = FeatureEngineer(df)
        eng.add_basic_features()

        assert eng.df["elo_diff"].iloc[0] == pytest.approx(100.0)

    def test_log_rank_diff_sign(self) -> None:
        """Winner ranked 5 vs loser ranked 10 → log_rank_diff < 0 (lower number = better)."""
        df = _base_df(3)
        eng = FeatureEngineer(df)
        eng.add_basic_features()

        # log1p(5) < log1p(10), so winner − loser < 0
        assert eng.df["log_rank_diff"].iloc[0] < 0.0

    def test_log_rank_diff_equal_ranks_zero(self) -> None:
        """Equal ranks must give log_rank_diff == 0."""
        df = _base_df(1)
        df.loc[0, "winner_rank"] = 7.0
        df.loc[0, "loser_rank"] = 7.0
        eng = FeatureEngineer(df)
        eng.add_basic_features()

        assert eng.df["log_rank_diff"].iloc[0] == pytest.approx(0.0)

    def test_category_flag_ms_is_zero(self) -> None:
        """MS match type must give category_flag = 0."""
        df = _base_df(2)
        eng = FeatureEngineer(df)
        eng.add_basic_features()

        assert (eng.df["category_flag"] == 0).all()

    def test_category_flag_ws_is_one(self) -> None:
        """WS match type must give category_flag = 1."""
        df = _base_df(2)
        df["type"] = "WS"
        eng = FeatureEngineer(df)
        eng.add_basic_features()

        assert (eng.df["category_flag"] == 1).all()

    def test_winner_home_when_same_country(self) -> None:
        """winner_home must be 1 when winner_assoc matches host country."""
        df = _base_df(2)  # winner_assoc='China', country='China'
        eng = FeatureEngineer(df)
        eng.add_basic_features()

        assert (eng.df["winner_home"] == 1).all()

    def test_loser_home_when_different_country(self) -> None:
        """loser_home must be 0 when loser_assoc differs from host country."""
        df = _base_df(2)  # loser_assoc='Japan', country='China'
        eng = FeatureEngineer(df)
        eng.add_basic_features()

        assert (eng.df["loser_home"] == 0).all()

    def test_level_numeric_international_series(self) -> None:
        """'IS' level must map to 2 (from LEVEL_MAP)."""
        df = _base_df(2)
        eng = FeatureEngineer(df)
        eng.add_basic_features()

        assert (eng.df["level_numeric"] == 2).all()

    def test_level_numeric_unknown_level_is_zero(self) -> None:
        """Unknown level strings must map to 0 via fillna."""
        df = _base_df(1)
        df.loc[0, "level"] = "UNKNOWN_LEVEL"
        eng = FeatureEngineer(df)
        eng.add_basic_features()

        assert eng.df["level_numeric"].iloc[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# add_rolling_features
# ---------------------------------------------------------------------------


class TestAddRollingFeatures:
    def test_first_appearance_winner_match_count_zero(self) -> None:
        """On the first match, winner_match_count must be 0 (no prior wins)."""
        df = _base_df(3)
        eng = FeatureEngineer(df)
        eng.add_rolling_features()

        assert eng.df["winner_match_count"].iloc[0] == 0

    def test_winner_match_count_accumulates(self) -> None:
        """Third appearance of the same winner must give match_count == 2."""
        df = _base_df(4)  # all winner_id = 'A'
        eng = FeatureEngineer(df)
        eng.add_rolling_features()

        assert eng.df["winner_match_count"].iloc[3] == 3

    def test_total_player_matches_column_exists(self) -> None:
        """total_player_matches column must be created."""
        df = _base_df(3)
        eng = FeatureEngineer(df)
        eng.add_rolling_features()

        assert "total_player_matches" in eng.df.columns

    def test_is_sparse_player_for_few_matches(self) -> None:
        """Rows with total_player_matches <= 5 must have is_sparse_player == 1."""
        df = _base_df(3)  # first match: total_player_matches = 0 + 0 = 0 → sparse
        eng = FeatureEngineer(df)
        eng.add_rolling_features()

        # First row: winner_match_count=0, loser_match_count=0 → sparse
        assert eng.df["is_sparse_player"].iloc[0] == 1


# ---------------------------------------------------------------------------
# add_fatigue_features
# ---------------------------------------------------------------------------


class TestAddFatigueFeatures:
    def test_implied_fatigue_formula(self) -> None:
        """implied_fatigue must equal duration * sets_played."""
        df = _base_df(3)  # duration=50, sets_played=2 → 100
        eng = FeatureEngineer(df)
        eng.add_fatigue_features()

        assert eng.df["implied_fatigue"].iloc[0] == pytest.approx(100.0)

    def test_implied_fatigue_three_sets(self) -> None:
        """3-set match must give higher implied_fatigue than 2-set match."""
        df = _base_df(2)
        df.loc[0, "sets_played"] = 2
        df.loc[1, "sets_played"] = 3
        eng = FeatureEngineer(df)
        eng.add_fatigue_features()

        assert eng.df["implied_fatigue"].iloc[1] > eng.df["implied_fatigue"].iloc[0]

    def test_implied_fatigue_column_exists(self) -> None:
        """implied_fatigue column must be present after add_fatigue_features()."""
        df = _base_df(2)
        eng = FeatureEngineer(df)
        eng.add_fatigue_features()

        assert "implied_fatigue" in eng.df.columns


# ---------------------------------------------------------------------------
# fit_scalers + apply_standardization
# ---------------------------------------------------------------------------


class TestScalerPipeline:
    def _df_with_varied_pace(self) -> pd.DataFrame:
        """DataFrame with five distinct seconds_per_point values for MS."""
        seconds = [20.0, 30.0, 40.0, 50.0, 60.0]
        durations = [45.0, 50.0, 55.0, 60.0, 65.0]
        n = len(seconds)
        df = pd.DataFrame(
            {
                "match_date": pd.date_range("2024-01-01", periods=n, freq="D"),
                "winner_id": ["A"] * n,
                "loser_id": ["B"] * n,
                "winner_elo": [1700.0] * n,
                "loser_elo": [1600.0] * n,
                "winner_rank": [5.0] * n,
                "loser_rank": [10.0] * n,
                "winner_assoc": ["China"] * n,
                "loser_assoc": ["Japan"] * n,
                "country": ["China"] * n,
                "type": ["MS"] * n,
                "level": ["IS"] * n,
                "seconds_per_point": seconds,
                "duration": durations,
                "sets_played": [2] * n,
            }
        )
        return df

    def test_fit_scalers_requires_no_prior_state(self) -> None:
        """fit_scalers must populate the scalers dict with 'MS' and/or 'WS' keys."""
        df = self._df_with_varied_pace()
        eng = FeatureEngineer(df)
        all_true = pd.Series([True] * len(df), index=df.index)
        eng.fit_scalers(all_true)

        assert "MS" in eng.scalers

    def test_fit_scalers_ms_mean_correct(self) -> None:
        """Stored MS mean for seconds_per_point must match empirical mean."""
        df = self._df_with_varied_pace()
        eng = FeatureEngineer(df)
        all_true = pd.Series([True] * len(df), index=df.index)
        eng.fit_scalers(all_true)

        expected_mean = float(np.mean([20.0, 30.0, 40.0, 50.0, 60.0]))
        assert eng.scalers["MS"]["seconds_per_point_mean"] == pytest.approx(expected_mean)

    def test_apply_standardization_raises_without_fit(self) -> None:
        """Calling apply_standardization() before fit_scalers() must raise ValueError."""
        df = self._df_with_varied_pace()
        eng = FeatureEngineer(df)

        with pytest.raises(ValueError, match="fit_scalers"):
            eng.apply_standardization()

    def test_apply_standardization_mean_row_pace_z_near_zero(self) -> None:
        """Row whose seconds_per_point equals the mean must have pace_z ≈ 0."""
        df = self._df_with_varied_pace()
        eng = FeatureEngineer(df)
        all_true = pd.Series([True] * len(df), index=df.index)
        eng.fit_scalers(all_true)
        eng.apply_standardization()

        # Row index 2 has seconds_per_point=40.0 which is the mean of [20,30,40,50,60]
        assert eng.df["pace_z"].iloc[2] == pytest.approx(0.0, abs=1e-9)

    def test_apply_standardization_above_mean_positive(self) -> None:
        """Row with seconds_per_point above mean must have pace_z > 0."""
        df = self._df_with_varied_pace()
        eng = FeatureEngineer(df)
        all_true = pd.Series([True] * len(df), index=df.index)
        eng.fit_scalers(all_true)
        eng.apply_standardization()

        # Row index 4: seconds_per_point=60.0 > mean=40.0 → pace_z > 0
        assert eng.df["pace_z"].iloc[4] > 0.0

    def test_apply_standardization_pace_z_and_duration_z_exist(self) -> None:
        """Both pace_z and duration_z columns must be present after standardization."""
        df = self._df_with_varied_pace()
        eng = FeatureEngineer(df)
        all_true = pd.Series([True] * len(df), index=df.index)
        eng.fit_scalers(all_true)
        eng.apply_standardization()

        assert "pace_z" in eng.df.columns
        assert "duration_z" in eng.df.columns


# ---------------------------------------------------------------------------
# get_feature_columns
# ---------------------------------------------------------------------------


class TestGetFeatureColumns:
    def test_non_post_match_features_included(self) -> None:
        """elo_diff and log_rank_diff (not post-match) must appear in the list."""
        df = _base_df(5)
        df["target"] = 1
        eng = FeatureEngineer(df)
        eng.add_basic_features()
        cols = eng.get_feature_columns()

        assert "elo_diff" in cols
        assert "log_rank_diff" in cols

    def test_post_match_features_excluded(self) -> None:
        """implied_fatigue, pace_z, duration_z must be excluded (POST_MATCH_FEATURES)."""
        df = _base_df(5)
        df["target"] = 1
        eng = FeatureEngineer(df)
        eng.add_basic_features()
        eng.add_fatigue_features()
        all_true = pd.Series([True] * len(df), index=df.index)
        eng.fit_scalers(all_true)
        eng.apply_standardization()
        cols = eng.get_feature_columns()

        assert "implied_fatigue" not in cols
        assert "pace_z" not in cols
        assert "duration_z" not in cols

    def test_result_stored_in_feature_cols_attribute(self) -> None:
        """get_feature_columns() must cache its result in self.feature_cols."""
        df = _base_df(5)
        df["target"] = 1
        eng = FeatureEngineer(df)
        eng.add_basic_features()
        cols = eng.get_feature_columns()

        assert eng.feature_cols == cols


# ---------------------------------------------------------------------------
# get_features_and_target
# ---------------------------------------------------------------------------


class TestGetFeaturesAndTarget:
    def test_returns_x_and_y(self) -> None:
        """Must return a (DataFrame, Series) tuple."""
        df = _base_df(5)
        df["target"] = 1
        eng = FeatureEngineer(df)
        eng.add_basic_features()
        cols = eng.get_feature_columns()
        X, y = eng.get_features_and_target(cols)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_x_shape_matches_feature_cols(self) -> None:
        """X must have exactly len(feature_cols) columns and n rows."""
        df = _base_df(5)
        df["target"] = 1
        eng = FeatureEngineer(df)
        eng.add_basic_features()
        cols = eng.get_feature_columns()
        X, _ = eng.get_features_and_target(cols)

        assert X.shape == (5, len(cols))

    def test_x_no_nans_after_nan_input(self) -> None:
        """NaN values in the feature matrix must be filled with 0."""
        df = _base_df(5)
        df["target"] = 1
        df.loc[2, "winner_rank"] = float("nan")  # causes NaN in log_rank_diff
        eng = FeatureEngineer(df)
        eng.add_basic_features()
        cols = eng.get_feature_columns()
        X, _ = eng.get_features_and_target(cols)

        assert not X.isna().any().any()

    def test_y_all_ones(self) -> None:
        """y must equal the target column (all 1s from add_target_variable)."""
        df = _base_df(5)
        df["target"] = 1
        eng = FeatureEngineer(df)
        eng.add_basic_features()
        cols = eng.get_feature_columns()
        _, y = eng.get_features_and_target(cols)

        assert (y == 1).all()


# ---------------------------------------------------------------------------
# build_features (pipeline)
# ---------------------------------------------------------------------------


class TestBuildFeaturesPipeline:
    def _df(self, n: int = 10) -> pd.DataFrame:
        df = _base_df(n)
        df["target"] = 1
        return df

    def test_returns_four_tuple(self) -> None:
        """build_features() must return (X, y, train_mask, engineer)."""
        from src.data.feature_engineering import build_features

        X, y, train_mask, engineer = build_features(self._df(10))

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(train_mask, pd.Series)
        assert isinstance(engineer, FeatureEngineer)

    def test_train_mask_70_percent(self) -> None:
        """Default train_ratio=0.7 must mark the first 70% of rows as True."""
        from src.data.feature_engineering import build_features

        _, _, train_mask, _ = build_features(self._df(10), train_ratio=0.7)

        assert train_mask.sum() == 7  # int(10 * 0.7)

    def test_x_columns_match_engineer_feature_cols(self) -> None:
        """X columns must match engineer.feature_cols (post-match features excluded)."""
        from src.data.feature_engineering import build_features

        X, _, _, engineer = build_features(self._df(10))

        assert list(X.columns) == engineer.feature_cols

    def test_x_no_nans(self) -> None:
        """X returned by build_features() must have no NaN values."""
        from src.data.feature_engineering import build_features

        X, _, _, _ = build_features(self._df(10))

        assert not X.isna().any().any()
