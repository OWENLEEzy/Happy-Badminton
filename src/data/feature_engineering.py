"""
Feature engineering module.

Functionality:
1. Basic features (Elo diff, rank diff, etc.)
2. Rolling state features (shift to prevent leakage)
3. Group-based standardization (statistics computed on training set only)
4. Interaction features (level x home advantage, etc.)
5. Fatigue accumulation features
"""

import pandas as pd
import numpy as np
from typing import Any
import loguru

from src.utils.constants import LEVEL_MAP, POST_MATCH_FEATURES

logger = loguru.logger


class FeatureEngineer:
    """Feature engineering processor."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the feature engineer.

        Args:
            df: Preprocessed data (already sorted by time)
        """
        self.df = df.copy()
        self.scalers = {}  # Store standardization statistics (to prevent leakage)
        self.feature_cols = []

    def add_basic_features(self) -> "FeatureEngineer":
        """
        Add basic features.

        - Elo_Diff
        - Log_Rank_Diff
        - Category_Flag
        - Home_Advantage
        """
        logger.info("Adding basic features...")

        # Elo difference
        self.df["elo_diff"] = self.df["winner_elo"] - self.df["loser_elo"]

        # Log rank difference
        self.df["log_winner_rank"] = np.log1p(self.df["winner_rank"])
        self.df["log_loser_rank"] = np.log1p(self.df["loser_rank"])
        self.df["log_rank_diff"] = self.df["log_winner_rank"] - self.df["log_loser_rank"]

        # Category flag
        self.df["category_flag"] = (self.df["type"] == "WS").astype(int)

        # Home advantage
        self.df["winner_home"] = (self.df["winner_assoc"] == self.df["country"]).astype(int)
        self.df["loser_home"] = (self.df["loser_assoc"] == self.df["country"]).astype(int)
        self.df["home_advantage"] = self.df["winner_home"]

        # Tournament level numeric encoding — single source of truth is LEVEL_MAP in constants.py.
        self.df["level_numeric"] = self.df["level"].map(LEVEL_MAP).fillna(0)  # type: ignore[arg-type]

        # Level x home advantage interaction
        self.df["level_x_home"] = self.df["level_numeric"] * self.df["home_advantage"]

        logger.info("  Basic features complete.")
        return self

    def fit_scalers(self, train_mask: pd.Series) -> "FeatureEngineer":
        """
        Compute and store standardization statistics on the training set.

        Important: Must be computed on the training set only to avoid data leakage.

        Args:
            train_mask: Boolean mask for the training set
        """
        logger.info("Computing standardization statistics (training set only)...")

        train_df = self.df[train_mask].copy()

        # Compute statistics grouped by MS/WS category
        for category in ["MS", "WS"]:
            cat_data = train_df[train_df["type"] == category]

            self.scalers[category] = {
                "seconds_per_point_mean": cat_data["seconds_per_point"].mean(),
                "seconds_per_point_std": cat_data["seconds_per_point"].std(),
                "duration_mean": cat_data["duration"].mean(),
                "duration_std": cat_data["duration"].std(),
            }

            logger.info(
                f"  {category}: Pace μ={self.scalers[category]['seconds_per_point_mean']:.1f}, σ={self.scalers[category]['seconds_per_point_std']:.1f}"
            )

        return self

    def apply_standardization(self) -> "FeatureEngineer":
        """
        Apply standardization using training set statistics.

        Must call fit_scalers() first.
        """
        if not self.scalers:
            raise ValueError("Please call fit_scalers() first!")

        logger.info("Applying standardization...")

        for category in ["MS", "WS"]:
            mask = self.df["type"] == category
            s = self.scalers[category]

            # Z-score standardization
            self.df.loc[mask, "pace_z"] = (
                self.df.loc[mask, "seconds_per_point"] - s["seconds_per_point_mean"]
            ) / s["seconds_per_point_std"]

            self.df.loc[mask, "duration_z"] = (
                self.df.loc[mask, "duration"] - s["duration_mean"]
            ) / s["duration_std"]

        # Fill NaN values if any
        self.df["pace_z"] = self.df["pace_z"].fillna(0)
        self.df["duration_z"] = self.df["duration_z"].fillna(0)

        logger.info("  Standardization complete.")
        return self

    def add_rolling_features(self, window: int = 5) -> "FeatureEngineer":
        """Add rolling career-count features for career stage computation.

        Uses cumcount() (0-based cumulative row count per player up to but not
        including the current row) so that no future match information leaks
        into earlier rows. Data must be sorted by match_date before this call.

        Args:
            window: Unused; kept for API compatibility.
        """
        logger.info("Adding rolling state features...")

        # cumcount() is 0-based: for the k-th appearance of a player, value = k.
        # This equals the number of matches they have played in the given role
        # BEFORE the current row, which is time-safe.
        self.df["winner_match_count"] = self.df.groupby("winner_id").cumcount()
        self.df["loser_match_count"] = self.df.groupby("loser_id").cumcount()

        # Approximate total career matches as winner's prior wins + loser's prior losses.
        # Intentionally kept consistent with the original two-stream design so that
        # downstream career_stage computation is not broken.
        self.df["total_player_matches"] = (
            self.df["winner_match_count"] + self.df["loser_match_count"]
        )

        self.df["is_sparse_player"] = (self.df["total_player_matches"] <= 5).astype(int)

        logger.info(f"  Sparse-player rows: {(self.df['is_sparse_player'] == 1).sum()}")

        return self

    def add_fatigue_features(self) -> "FeatureEngineer":
        """Add a simplified fatigue proxy feature.

        Computes implied_fatigue = duration * sets_played as a rough measure of
        match load. This is a post-match feature (requires actual duration and
        sets played) and should only be used in pipelines that accept post-match
        inputs. It does NOT implement exponential decay — for time-decayed fatigue
        accumulation see FatigueFeatures in advanced_features.py.
        """
        logger.info("Adding fatigue features...")

        # Simplified version: use match duration as fatigue indicator.
        # Ideally, this should compute cumulative duration over a rolling window.

        self.df["implied_fatigue"] = self.df["duration"] * self.df["sets_played"]

        logger.info("  Fatigue features complete.")
        return self

    def get_feature_columns(self) -> list:
        """
        Get feature column names.

        Returns:
            List of feature column names
        """
        # Define feature columns
        feature_cols = [
            # Basic features
            "elo_diff",
            "log_rank_diff",
            "category_flag",
            "home_advantage",
            "level_numeric",
            "level_x_home",
            # Score features
            "total_points",
            "sets_played",
            "point_diff_set1",
            "seconds_per_point",
            "implied_fatigue",
            # Standardized features
            "pace_z",
            "duration_z",
            # Player features
            "total_player_matches",
            "is_sparse_player",
        ]

        # Keep only columns that exist and are not post-match features (to prevent leakage)
        _post_match_set = set(POST_MATCH_FEATURES)
        feature_cols = [
            c for c in feature_cols if c in self.df.columns and c not in _post_match_set
        ]

        self.feature_cols = feature_cols
        return feature_cols

    def get_features_and_target(
        self, feature_cols: list | None = None
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Get feature matrix and target variable.

        Args:
            feature_cols: List of feature column names

        Returns:
            (X, y) feature matrix and target variable
        """
        if feature_cols is None:
            feature_cols = self.get_feature_columns()

        X = self.df[feature_cols].copy()
        y = self.df["target"].copy()

        # Handle NaN values
        X = X.fillna(0)

        return X, y


def build_features(
    df: pd.DataFrame, train_ratio: float = 0.7
) -> tuple[pd.DataFrame, pd.Series, pd.Series, Any]:
    """
    Complete feature engineering pipeline.

    Args:
        df: Preprocessed data
        train_ratio: Training set ratio

    Returns:
        (X, y, train_mask, engineer) feature matrix, target variable, training set mask, fitted FeatureEngineer
    """
    engineer = FeatureEngineer(df)

    # 1. Add basic features
    engineer.add_basic_features()

    # 2. Define training set (time-based split)
    n = len(df)
    train_size = int(n * train_ratio)
    train_mask = pd.Series([False] * n, index=df.index)
    train_mask.iloc[:train_size] = True

    # 3. Compute standardization statistics on training set
    engineer.fit_scalers(train_mask)

    # 4. Apply standardization
    engineer.apply_standardization()

    # 5. Add rolling features
    engineer.add_rolling_features()

    # 6. Add fatigue features
    engineer.add_fatigue_features()

    # 7. Get features and target
    feature_cols = engineer.get_feature_columns()
    X, y = engineer.get_features_and_target(feature_cols)

    logger.info(f"Feature engineering complete: {X.shape}")
    logger.info(f"Feature columns: {len(feature_cols)}")

    return X, y, train_mask, engineer


if __name__ == "__main__":
    # Test
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.utils.logger import setup_logger

    setup_logger()

    # Load data
    df = pd.read_parquet("data/processed/matches_clean.parquet")

    # Feature engineering
    X, y, train_mask, engineer = build_features(df)

    logger.info(f"Feature matrix: {X.shape}")
    logger.info(f"Feature columns: {X.columns.tolist()}")
    logger.info(f"Feature preview:\n{X.head()}")
