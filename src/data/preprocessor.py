"""
Data preprocessing module.

Functionality:
1. Filter future dates
2. Identify and remove retired/incomplete matches
3. Handle outliers (duration)
4. Parse scores
5. Save processed data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import loguru

from src.utils.helpers import (
    is_retirement,
    extract_total_points,
    count_sets,
    calculate_point_diff_set1,
)

logger = loguru.logger


class DataPreprocessor:
    """Data preprocessing handler."""

    # Duration outlier thresholds
    MAX_DURATION_MINUTES = 150

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the preprocessor.

        Args:
            df: Fully merged raw data
        """
        self.df = df.copy()
        self.original_shape = df.shape
        self.stats = {}

    def filter_future_dates(self, cutoff_date: pd.Timestamp = None) -> "DataPreprocessor":
        """
        Filter out rows with future dates.

        Args:
            cutoff_date: Cutoff date; defaults to current date/time

        Returns:
            self
        """
        if cutoff_date is None:
            cutoff_date = pd.Timestamp.now()

        before_count = len(self.df)
        self.df = self.df[self.df["match_date"] < cutoff_date].copy()
        after_count = len(self.df)

        self.stats["future_dates_filtered"] = before_count - after_count
        logger.info(
            f"Filter future dates: {before_count} -> {after_count} (removed {self.stats['future_dates_filtered']} rows)"
        )

        return self

    def identify_and_filter_retirements(self) -> "DataPreprocessor":
        """
        Identify and flag retired matches, then remove them.

        Returns:
            self
        """
        # Flag retirement matches
        self.df["is_retirement"] = self.df["score"].apply(is_retirement)

        retirement_count = self.df["is_retirement"].sum()
        before_count = len(self.df)

        # Remove retirement matches
        self.df = self.df[~self.df["is_retirement"]].copy()
        after_count = len(self.df)

        self.stats["retirement_matches"] = retirement_count
        self.stats["retirement_rate"] = retirement_count / before_count * 100

        logger.info(
            f"Retirement/incomplete matches: {retirement_count} ({self.stats['retirement_rate']:.2f}%)"
        )
        logger.info(f"After removing retirements: {before_count} -> {after_count}")

        return self

    def handle_duration_outliers(self) -> "DataPreprocessor":
        """
        Handle duration outliers.

        Rules:
        - 0 minutes: interpolate if points exist; otherwise mark as missing
        - > MAX_DURATION_MINUTES: cap at MAX_DURATION_MINUTES

        Returns:
            self
        """
        before_stats = self.df["duration"].describe()

        # Count outliers
        zero_duration = (self.df["duration"] == 0).sum()
        max_duration = self.df["duration"].max()
        extreme_duration = (self.df["duration"] > self.MAX_DURATION_MINUTES).sum()

        self.stats["zero_duration"] = zero_duration
        self.stats["extreme_duration"] = extreme_duration
        self.stats["max_duration_original"] = max_duration

        logger.info("Duration outlier stats:")
        logger.info(f"  0 min: {zero_duration} matches")
        logger.info(f"  > {self.MAX_DURATION_MINUTES} min: {extreme_duration} matches")
        logger.info(f"  Max value: {max_duration} min")

        # Cap extreme values
        self.df.loc[self.df["duration"] > self.MAX_DURATION_MINUTES, "duration"] = (
            self.MAX_DURATION_MINUTES
        )

        # Zero-duration: mark as NaN for later interpolation
        zero_mask = self.df["duration"] == 0
        self.df.loc[zero_mask, "duration"] = np.nan

        after_stats = self.df["duration"].describe()
        logger.info("Duration stats after handling:")
        logger.info(f"  Mean: {before_stats['mean']:.1f} -> {after_stats['mean']:.1f} min")
        logger.info(f"  Median: {before_stats['50%']:.1f} -> {after_stats['50%']:.1f} min")

        return self

    def parse_scores(self) -> "DataPreprocessor":
        """
        Parse score strings and extract features.

        Returns:
            self
        """
        logger.info("Parsing scores...")

        self.df["total_points"] = self.df["score"].apply(extract_total_points)
        self.df["sets_played"] = self.df["score"].apply(count_sets)
        self.df["point_diff_set1"] = self.df["score"].apply(calculate_point_diff_set1)

        # Compute seconds per point
        self.df["seconds_per_point"] = self.df["duration"] * 60 / self.df["total_points"]

        logger.info("Score parsing complete:")
        logger.info(f"  total_points: mean={self.df['total_points'].mean():.1f}")
        logger.info(
            f"  sets_played: 2 sets={(self.df['sets_played'] == 2).sum()}, 3 sets={(self.df['sets_played'] == 3).sum()}"
        )
        logger.info(f"  seconds_per_point: mean={self.df['seconds_per_point'].mean():.1f}")

        return self

    def handle_missing_values(self) -> "DataPreprocessor":
        """
        Handle missing values.

        - winner_rank / loser_rank: fill with 999
        - duration: interpolate by match type median

        Returns:
            self
        """
        # Fill missing rank values
        self.df["winner_rank"] = self.df["winner_rank"].fillna(999)
        self.df["loser_rank"] = self.df["loser_rank"].fillna(999)

        # Interpolate missing duration using per-type median
        if self.df["duration"].isna().any():
            for match_type in ["MS", "WS"]:
                type_median = self.df[
                    (self.df["type"] == match_type) & (self.df["duration"].notna())
                ]["duration"].median()

                mask = (self.df["type"] == match_type) & (self.df["duration"].isna())
                self.df.loc[mask, "duration"] = type_median
                logger.info(
                    f"  {match_type} duration missing -> filled with median: {type_median:.1f} min"
                )

        return self

    def sort_by_date(self) -> "DataPreprocessor":
        """
        Sort by date (critical for preventing data leakage).

        Returns:
            self
        """
        self.df = self.df.sort_values("match_date").reset_index(drop=True)
        logger.info("Data sorted by match_date")
        return self

    def add_target_variable(self) -> "DataPreprocessor":
        """
        Add the target variable.

        target = 1 indicates winner_id won (always 1 since data is from the winner's perspective).
        The actual prediction target (player A vs B) is constructed during feature engineering.

        Returns:
            self
        """
        # Simple label: winner = 1
        self.df["target"] = 1
        return self

    def save_processed(self, output_path: str) -> None:
        """
        Save processed data to disk.

        Args:
            output_path: Output file path (.parquet or .csv)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure score column is string type (required for parquet)
        if "score" in self.df.columns:
            self.df["score"] = self.df["score"].astype(str)

        # Ensure round column is string type
        if "round" in self.df.columns:
            self.df["round"] = self.df["round"].astype(str)

        # Ensure type column is string type
        if "type" in self.df.columns:
            self.df["type"] = self.df["type"].astype(str)

        # Ensure categorical columns are string type
        for col in ["level", "country", "winner_assoc", "loser_assoc"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str)

        if output_path.suffix == ".parquet":
            self.df.to_parquet(output_path, index=False)
        elif output_path.suffix == ".csv":
            self.df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported file format: {output_path.suffix}")

        logger.info(f"Processed data saved: {output_path}")

    def get_summary(self) -> dict:
        """
        Get preprocessing statistics summary.

        Returns:
            Summary statistics dictionary
        """
        return {
            "original_shape": self.original_shape,
            "final_shape": self.df.shape,
            "stats": self.stats,
        }


def preprocess_pipeline(input_df: pd.DataFrame, cutoff_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Complete preprocessing pipeline.

    Args:
        input_df: Fully merged raw data
        cutoff_date: Cutoff date

    Returns:
        Processed DataFrame
    """
    preprocessor = DataPreprocessor(input_df)

    preprocessor.filter_future_dates(
        cutoff_date
    ).identify_and_filter_retirements().handle_duration_outliers().parse_scores().handle_missing_values().sort_by_date().add_target_variable()

    logger.info("Preprocessing pipeline complete")
    logger.info(f"Final data shape: {preprocessor.df.shape}")

    return preprocessor.df


if __name__ == "__main__":
    # Test
    from src.data.loader import load_and_merge

    # Load data
    df = load_and_merge("data/raw/Tournament Results.xlsx")

    # Preprocess
    processed_df = preprocess_pipeline(df)

    # Save
    preprocessor = DataPreprocessor(df)
    preprocessor.filter_future_dates().identify_and_filter_retirements().handle_duration_outliers().parse_scores().handle_missing_values().sort_by_date().add_target_variable()

    preprocessor.save_processed("data/processed/matches_clean.parquet")

    logger.info(f"Processed data preview:\n{preprocessor.df.head()}")
    logger.info(f"Data shape: {preprocessor.df.shape}")
