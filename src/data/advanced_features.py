"""
Advanced feature engineering module.

SOTA improvements:
1. MOV (Margin of Victory) weighted Elo rating
2. Momentum features (recent form trend)
3. Head-to-Head (H2H) historical record features
4. Enhanced fatigue accumulation features

References:
- [Tennis Elo comparison 2024](https://www.researchgate.net/publication/375589629_A_comparative_evaluation_of_Elo_ratings)
- [Badminton dynamic Elo 2024](https://www.researchgate.net/publication/381185834_Badminton_player_ranking_model_based_on_dynamic_ELO_model)
"""

import pandas as pd
import numpy as np
import yaml
from collections import defaultdict
from pathlib import Path
import loguru

logger = loguru.logger


def _load_k_factor() -> float:
    """Load elo_k_factor from config.yaml, falling back to MOVEloRating.K_FACTOR_BASE."""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    try:
        with config_path.open() as f:
            cfg = yaml.safe_load(f)
        return float(cfg["features"]["elo_k_factor"])
    except Exception:
        return 32.0


class MOVEloRating:
    """
    Margin of Victory (MOV) weighted Elo rating system.

    Based on the paper "A Comparative Evaluation of Elo Ratings" (2024).
    The MOV method adjusts Elo changes by score margin, more accurately reflecting player strength.

    Key improvements:
    - Dominant win (e.g. 3-0) -> larger Elo change
    - Narrow win (e.g. 2-1) -> smaller Elo change
    - Tournament level weights applied
    """

    # Default Elo parameters
    INITIAL_ELO = 1500
    K_FACTOR_BASE = 32

    # Tournament level weights
    LEVEL_WEIGHTS = {
        "OG": 1.5,  # Olympics - highest weight
        "WC": 1.4,  # World Championships
        "WTF": 1.3,  # World Tour Finals
        "S1000": 1.25,
        "S750": 1.2,
        "S500": 1.15,
        "S300": 1.1,
        "S100": 1.05,
        "IS": 1.0,  # International Series
        "IC": 1.0,  # International Challenge
    }

    def __init__(self, k_factor: float | None = None, mov_exponent: float = 1.0) -> None:
        """
        Initialize the MOV Elo system.

        Args:
            k_factor: K-factor (default 32)
            mov_exponent: MOV influence exponent (1.0=linear, <1=compressed, >1=amplified)
        """
        self.k_factor = k_factor if k_factor is not None else self.K_FACTOR_BASE
        self.mov_exponent = mov_exponent
        self.elo_ratings: dict[str, float] = defaultdict(lambda: self.INITIAL_ELO)

    def calculate_mov_multiplier(
        self, winner_score: int, loser_score: int, sets_played: int
    ) -> float:
        """
        Calculate a multiplier based on score margin.

        A 3-0 dominant win receives a higher bonus; a 2-1 narrow win receives a lower bonus.

        Args:
            winner_score: Winner's total points
            loser_score: Loser's total points
            sets_played: Number of sets played

        Returns:
            MOV multiplier (1.0 ~ 2.0)
        """
        total_points = winner_score + loser_score
        if total_points == 0:
            return 1.0

        point_diff = winner_score - loser_score
        point_ratio = point_diff / total_points

        # Set bonus: 3-set matches carry higher weight
        set_bonus = 1.2 if sets_played == 3 else 1.0

        # MOV multiplier = base + point ratio * set bonus
        mov_multiplier = 1.0 + (point_ratio * set_bonus)

        # Apply exponent adjustment
        mov_multiplier = mov_multiplier**self.mov_exponent

        return min(mov_multiplier, 2.5)  # Cap the maximum value

    def calculate_expected_score(self, elo_a: float, elo_b: float) -> float:
        """
        Calculate expected score using the standard Elo formula.

        Args:
            elo_a: Player A's Elo rating
            elo_b: Player B's Elo rating

        Returns:
            Player A's expected score (0~1)
        """
        return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))

    def update_ratings(
        self,
        winner_id: str,
        loser_id: str,
        winner_score: int,
        loser_score: int,
        sets_played: int,
        level: str = "IS",
    ) -> tuple[float, float]:
        """
        Update Elo ratings for both players.

        Args:
            winner_id: Winner's player ID
            loser_id: Loser's player ID
            winner_score: Winner's total points
            loser_score: Loser's total points
            sets_played: Number of sets played
            level: Tournament level

        Returns:
            (winner_new_elo, loser_new_elo) updated ratings
        """
        # Get current ratings
        winner_elo = self.elo_ratings[winner_id]
        loser_elo = self.elo_ratings[loser_id]

        # Calculate expected scores
        expected_winner = self.calculate_expected_score(winner_elo, loser_elo)
        expected_loser = 1.0 - expected_winner

        # Calculate MOV multiplier
        mov_multiplier = self.calculate_mov_multiplier(winner_score, loser_score, sets_played)

        # Tournament level weight
        level_weight = self.LEVEL_WEIGHTS.get(level, 1.0)

        # Effective K-factor
        effective_k = self.k_factor * mov_multiplier * level_weight

        # Update ratings
        winner_new_elo = winner_elo + effective_k * (1.0 - expected_winner)
        loser_new_elo = loser_elo + effective_k * (0.0 - expected_loser)

        # Save new ratings
        self.elo_ratings[winner_id] = winner_new_elo
        self.elo_ratings[loser_id] = loser_new_elo

        return winner_new_elo, loser_new_elo

    def get_rating(self, player_id: str) -> float:
        """Get a player's current Elo rating."""
        return self.elo_ratings.get(player_id, self.INITIAL_ELO)

    def compute_mov_elo_history(
        self,
        df: pd.DataFrame,
        winner_col: str = "winner_id",
        loser_col: str = "loser_id",
        score_col: str = "score",
        level_col: str = "level",
    ) -> pd.DataFrame:
        """
        Compute historical MOV Elo ratings.

        Warning: Computed in chronological order to avoid data leakage.

        Args:
            df: Match data (must be sorted by date)
            winner_col: Column name for winner ID
            loser_col: Column name for loser ID
            score_col: Column name for score
            level_col: Column name for tournament level

        Returns:
            DataFrame with MOV Elo columns added
        """
        logger.info("Computing MOV Elo ratings...")

        df = df.copy().reset_index(drop=True)

        # Initialize columns
        df["winner_mov_elo"] = float(self.INITIAL_ELO)
        df["loser_mov_elo"] = float(self.INITIAL_ELO)

        # Update in chronological order
        for idx, row in df.iterrows():
            winner_id = str(row[winner_col])
            loser_id = str(row[loser_col])

            # Record pre-match ratings
            df.at[idx, "winner_mov_elo"] = self.get_rating(winner_id)  # type: ignore[arg-type]
            df.at[idx, "loser_mov_elo"] = self.get_rating(loser_id)  # type: ignore[arg-type]

            # Parse scores — pd.Series.get() stubs return Unknown|None; defaults ensure int/str safety
            winner_score = int(row.get("winner_points", 21))  # type: ignore[arg-type]
            loser_score = int(row.get("loser_points", 15))  # type: ignore[arg-type]
            sets_played = int(row.get("sets_played", 2))  # type: ignore[arg-type]
            level = str(row.get(level_col, "IS"))

            # Update ratings
            self.update_ratings(winner_id, loser_id, winner_score, loser_score, sets_played, level)  # type: ignore[arg-type]

        # Compute Elo difference
        df["mov_elo_diff"] = df["winner_mov_elo"] - df["loser_mov_elo"]

        logger.info("MOV Elo computation complete.")
        logger.info(
            f"  Winner Elo range: {df['winner_mov_elo'].min():.1f} ~ {df['winner_mov_elo'].max():.1f}"
        )
        logger.info(
            f"  Loser Elo range: {df['loser_mov_elo'].min():.1f} ~ {df['loser_mov_elo'].max():.1f}"
        )

        return df


class MomentumFeatures:
    """
    Momentum feature engineering.

    Reference: "Momentum Tracking and Analysis of Tennis Matches" (ACM 2024)
    Captures recent form trends and momentum shifts for each player.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize the momentum feature calculator.

        Args:
            df: Match data (must be sorted by date)
        """
        self.df = df.copy().reset_index(drop=True)

    def add_form_features(self, windows: list[int] | None = None) -> pd.DataFrame:
        """Add recent form features using time-safe rolling win rates.

        Builds a unified per-player appearance timeline, computes rolling win
        rate with shift(1) to prevent leakage, then maps back to df rows via
        the original row index and role (winner/loser).

        Args:
            windows: Rolling window sizes.

        Returns:
            DataFrame with winner_form_*, loser_form_*, form_diff_* columns.
        """
        if windows is None:
            windows = [5, 10, 20]
        logger.info(f"Adding form features (windows={windows})...")

        # Build unified timeline preserving df index and role so we can map back.
        winner_records = pd.DataFrame(
            {
                "player_id": self.df["winner_id"].values,
                "date": self.df["match_date"].values,
                "won": 1,
                "df_idx": self.df.index,
                "role": "winner",
            }
        )
        loser_records = pd.DataFrame(
            {
                "player_id": self.df["loser_id"].values,
                "date": self.df["match_date"].values,
                "won": 0,
                "df_idx": self.df.index,
                "role": "loser",
            }
        )
        all_matches = pd.concat([winner_records, loser_records], ignore_index=True)
        # Sort chronologically; df_idx breaks ties to preserve original match order.
        all_matches = all_matches.sort_values(["date", "df_idx"]).reset_index(drop=True)

        for window in windows:
            # Rolling win rate per player with shift(1) — no leakage of current match result.
            all_matches[f"form_{window}"] = (
                all_matches.groupby("player_id")["won"].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=3).mean()
                )
            ).fillna(0.5)

            # Map computed form back to original df rows via df_idx and role.
            winner_form = all_matches[all_matches["role"] == "winner"].set_index("df_idx")[
                f"form_{window}"
            ]
            loser_form = all_matches[all_matches["role"] == "loser"].set_index("df_idx")[
                f"form_{window}"
            ]

            self.df[f"winner_form_{window}"] = winner_form.reindex(self.df.index).fillna(0.5).values
            self.df[f"loser_form_{window}"] = loser_form.reindex(self.df.index).fillna(0.5).values
            self.df[f"form_diff_{window}"] = (
                self.df[f"winner_form_{window}"] - self.df[f"loser_form_{window}"]
            )

        logger.info("  Form features added.")
        return self.df

    def add_streak_features(self) -> pd.DataFrame:
        """Add win/loss streak features.

        Tracks win streaks and loss streaks separately so that a player on a
        losing streak gets a negative loser_streak value (not 0 as in the
        previous implementation that only tracked win streaks and reset on loss).

        winner_streak: consecutive wins the winner had BEFORE this match (>= 0).
        loser_streak:  negative of consecutive losses the loser had before this
                       match (0 = no prior losing streak, -3 = lost 3 in a row).

        Returns:
            DataFrame with winner_streak, loser_streak, streak_diff columns.
        """
        logger.info("Adding streak features...")

        win_streaks: dict = defaultdict(int)  # player -> current consecutive wins
        loss_streaks: dict = defaultdict(int)  # player -> current consecutive losses

        winner_streak_list: list = []
        loser_streak_list: list = []

        for _, row in self.df.iterrows():
            winner_id = row["winner_id"]
            loser_id = row["loser_id"]

            # Record pre-match streaks (before updating with this result).
            winner_streak_list.append(win_streaks[winner_id])
            loser_streak_list.append(-loss_streaks[loser_id])

            # Update streaks based on match outcome.
            win_streaks[winner_id] += 1
            loss_streaks[winner_id] = 0  # winner resets loss streak
            loss_streaks[loser_id] += 1
            win_streaks[loser_id] = 0  # loser resets win streak

        self.df["winner_streak"] = winner_streak_list
        self.df["loser_streak"] = loser_streak_list
        self.df["streak_diff"] = self.df["winner_streak"] - self.df["loser_streak"]

        logger.info("  Streak features added.")
        return self.df


class HeadToHeadFeatures:
    """
    Head-to-Head (H2H) feature engineering.

    Captures historical head-to-head advantage between players.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize H2H feature calculator.

        Args:
            df: Match data sorted by date.
        """
        self.df = df.copy().reset_index(drop=True)
        # pair_total: unordered pair -> total matches played (no double-counting)
        self.pair_total: dict = defaultdict(int)
        # directed_wins: (winner_id, loser_id) -> times winner beat loser
        self.directed_wins: dict = defaultdict(int)

    def compute_h2h_features(self) -> pd.DataFrame:
        """Compute head-to-head win rate features.

        Uses pre-match H2H records (updated after each row) so there is no
        leakage of the current match result. pair_total and directed_wins are
        stored separately to avoid double-counting when the winner_id happens
        to be alphabetically first (which would make pair_key == (winner, loser)).

        Returns:
            DataFrame with h2h_win_rate, h2h_matches, h2h_recent_form columns.
        """
        logger.info("Computing H2H features...")

        self.df["h2h_win_rate"] = 0.5
        self.df["h2h_matches"] = 0
        self.df["h2h_recent_form"] = 0.5

        for idx, row in self.df.iterrows():
            winner_id = row["winner_id"]
            loser_id = row["loser_id"]
            pair_key = tuple(sorted([winner_id, loser_id]))

            total = self.pair_total[pair_key]
            if total > 0:
                directed = self.directed_wins[(winner_id, loser_id)]
                self.df.at[idx, "h2h_win_rate"] = directed / total
                self.df.at[idx, "h2h_matches"] = total

            # Update AFTER reading to avoid leaking current match result.
            self.directed_wins[(winner_id, loser_id)] += 1
            self.pair_total[pair_key] += 1

        logger.info("  H2H features computed.")
        return self.df


class FatigueFeatures:
    """
    Fatigue accumulation features.

    Considers:
    - Recent number of matches played
    - Cumulative match duration
    - Travel fatigue (cross-country matches)
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize the fatigue feature calculator.

        Args:
            df: Match data (must be sorted by date)
        """
        self.df = df.copy().reset_index(drop=True)
        self.player_fatigue = defaultdict(float)

    def compute_fatigue_features(
        self, decay_days: int = 7, weight_factor: float = 0.7
    ) -> pd.DataFrame:
        """
        Compute fatigue accumulation features.

        Args:
            decay_days: Number of days for fatigue decay
            weight_factor: Per-day decay coefficient

        Returns:
            DataFrame with fatigue features added
        """
        logger.info("Computing fatigue accumulation features...")

        self.df["winner_fatigue"] = 0.0
        self.df["loser_fatigue"] = 0.0
        self.df["fatigue_diff"] = 0.0

        current_date = None
        fatigue_accumulator = defaultdict(float)

        for idx, row in self.df.iterrows():
            match_date = row["match_date"]
            winner_id = row["winner_id"]
            loser_id = row["loser_id"]
            duration = row.get("duration", 30)
            sets_played = row.get("sets_played", 2)

            # Apply date-based decay
            if current_date is not None:
                days_diff = (match_date - current_date).days  # type: ignore[union-attr]
                if days_diff > 0:
                    decay = weight_factor**days_diff
                    for player in fatigue_accumulator:
                        fatigue_accumulator[player] *= decay

            current_date = match_date

            # Record current fatigue values
            self.df.at[idx, "winner_fatigue"] = fatigue_accumulator[winner_id]
            self.df.at[idx, "loser_fatigue"] = fatigue_accumulator[loser_id]

            # Compute fatigue increment for this match
            fatigue_increment = duration * sets_played / 60.0  # Normalized to hours

            # Update fatigue accumulator
            fatigue_accumulator[winner_id] += fatigue_increment
            fatigue_accumulator[loser_id] += fatigue_increment

        # Fatigue difference
        self.df["fatigue_diff"] = self.df["winner_fatigue"] - self.df["loser_fatigue"]

        logger.info("  Fatigue features computed.")
        return self.df


class NationalityFeatures:
    """Nationality-based and continent-based features.

    Computes:
    - same_nationality: binary flag for same-country derby matches
    - nat_matchup_win_rate_w: Bayesian rolling win rate of winner's nation vs loser's nation
    - nat_matchup_win_diff: deviation from 0.5 (positive = winner's nation historically dominates)
    - winner_continent_home: 1 if winner's continent matches host continent
    - loser_continent_home: 1 if loser's continent matches host continent
    - continent_advantage_diff: winner_continent_home - loser_continent_home
    """

    NAT_PRIOR = 10  # stronger prior than player H2H (less granular data)

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy().sort_values("match_date").reset_index(drop=True)

    def add_nationality_features(self) -> pd.DataFrame:
        """Compute all nationality and continent features. Shift-safe (no leakage)."""
        from src.utils.constants import CONTINENT_MAP

        df = self.df.copy()

        # 1. Same-nationality derby flag
        df["same_nationality"] = (df["winner_assoc"] == df["loser_assoc"]).astype(int)

        # 2. Nationality matchup win rate (Bayesian rolling, shift-safe)
        # Canonical pair key: alphabetically sorted so (A,B) == (B,A)
        df["_nat_pair"] = df.apply(
            lambda r: f"{min(r['winner_assoc'], r['loser_assoc'])}|{max(r['winner_assoc'], r['loser_assoc'])}",
            axis=1,
        )
        df["_winner_is_first"] = (df["winner_assoc"] <= df["loser_assoc"]).astype(int)

        # Cumulative count of previous matches in this nat pair
        df["_nat_cum_total"] = df.groupby("_nat_pair").cumcount()

        # Cumulative wins for the "first" (alphabetically) nation in pair (shift-safe)
        df["_nat_cum_wins"] = (
            df.groupby("_nat_pair")["_winner_is_first"]
            .transform(lambda x: x.shift(1).expanding(min_periods=1).sum())
            .fillna(0)
        )

        df["_nat_first_win_rate"] = (df["_nat_cum_wins"] + self.NAT_PRIOR * 0.5) / (
            df["_nat_cum_total"] + self.NAT_PRIOR
        )

        # Convert to winner's perspective
        df["nat_matchup_win_rate_w"] = np.where(
            df["_winner_is_first"] == 1,
            df["_nat_first_win_rate"],
            1.0 - df["_nat_first_win_rate"],
        )
        df["nat_matchup_win_diff"] = df["nat_matchup_win_rate_w"] - 0.5

        # 3. Continent features
        df["_winner_continent"] = df["winner_assoc"].map(CONTINENT_MAP).fillna("Unknown")  # type: ignore[arg-type]
        df["_loser_continent"] = df["loser_assoc"].map(CONTINENT_MAP).fillna("Unknown")  # type: ignore[arg-type]
        df["_host_continent"] = df["country"].map(CONTINENT_MAP).fillna("Unknown")  # type: ignore[arg-type]

        df["winner_continent_home"] = (
            (df["_winner_continent"] == df["_host_continent"])
            & (df["_winner_continent"] != "Unknown")
        ).astype(int)
        df["loser_continent_home"] = (
            (df["_loser_continent"] == df["_host_continent"])
            & (df["_loser_continent"] != "Unknown")
        ).astype(int)
        df["continent_advantage_diff"] = df["winner_continent_home"] - df["loser_continent_home"]

        # Drop temp columns
        df = df.drop(
            columns=[
                "_nat_pair",
                "_winner_is_first",
                "_nat_cum_total",
                "_nat_cum_wins",
                "_nat_first_win_rate",
                "_winner_continent",
                "_loser_continent",
                "_host_continent",
            ]
        )

        logger.info("  Nationality features computed.")
        return df


def build_nat_pair_lookup(
    df: pd.DataFrame, train_end: int | None = None, nat_prior: int = 10
) -> dict[str, float]:
    """Build final nat pair win rate lookup from fully-featured dataframe.

    Returns dict mapping canonical 'NationA|NationB' -> win_rate_of_NationA (0–1).
    Call this after NationalityFeatures.add_nationality_features() has been run.

    Args:
        df: Match data with nationality features.
        train_end: Optional index boundary. If provided, only use df[:train_end] to
                   build the lookup (prevents test-set leakage).
        nat_prior: Prior count for Bayesian smoothing. Default from config.yaml.

    Returns:
        Dict mapping 'NationA|NationB' to Bayesian-smoothed win rate of NationA.
    """
    # Use only training data if train_end is specified
    df_build = df[:train_end] if train_end is not None else df.copy()
    df_build = df_build.copy()
    df_build["_nat_pair"] = df_build.apply(
        lambda r: f"{min(r['winner_assoc'], r['loser_assoc'])}|{max(r['winner_assoc'], r['loser_assoc'])}",
        axis=1,
    )
    df_build["_winner_is_first"] = (df_build["winner_assoc"] <= df_build["loser_assoc"]).astype(int)

    lookup: dict[str, float] = {}
    for pair, group in df_build.groupby("_nat_pair"):
        total = len(group)
        wins_first = group["_winner_is_first"].sum()
        lookup[str(pair)] = float((wins_first + nat_prior * 0.5) / (total + nat_prior))
    return lookup


def build_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all advanced features.

    Args:
        df: Preprocessed data (must be sorted by date)

    Returns:
        DataFrame with all advanced features added
    """
    df = df.copy().sort_values("match_date").reset_index(drop=True)
    original_col_count = len(df.columns)

    logger.info("=" * 50)
    logger.info("Building advanced SOTA features")
    logger.info("=" * 50)

    # 1. MOV Elo — k_factor sourced from config.yaml (features.elo_k_factor)
    mov_elo = MOVEloRating(k_factor=_load_k_factor(), mov_exponent=1.0)
    df = mov_elo.compute_mov_elo_history(df)

    # 2. Momentum features
    momentum = MomentumFeatures(df)
    df = momentum.add_form_features(windows=[5, 10, 20])
    df = momentum.add_streak_features()

    # 3. H2H features
    h2h = HeadToHeadFeatures(df)
    df = h2h.compute_h2h_features()

    # 4. Fatigue features
    fatigue = FatigueFeatures(df)
    df = fatigue.compute_fatigue_features()

    # 5. Nationality and continent features
    nat = NationalityFeatures(df)
    df = nat.add_nationality_features()

    logger.info("=" * 50)
    logger.info(f"Advanced feature build complete: {df.shape}")
    logger.info(f"New feature columns added: {len(df.columns) - original_col_count}")
    logger.info("=" * 50)

    return df


def get_advanced_feature_columns() -> list:
    """Return all feature columns produced by build_advanced_features().

    Note: downstream callers should filter this list against df.columns and
    SOTA_EXCLUDE so that stale or excluded columns are dropped automatically.
    """
    return [
        # MOV Elo (pre-match computed; mov_elo_diff is the training target — exclude from features)
        "winner_mov_elo",
        "loser_mov_elo",
        "mov_elo_diff",
        # Momentum / form
        "winner_form_5",
        "loser_form_5",
        "winner_form_10",
        "loser_form_10",
        "winner_form_20",
        "loser_form_20",
        "form_diff_5",
        "form_diff_10",
        "form_diff_20",
        "winner_streak",
        "loser_streak",
        "streak_diff",
        # H2H
        "h2h_win_rate",
        "h2h_matches",
        # Note: h2h_recent_form is intentionally omitted — it is initialized to 0.5
        # in compute_h2h_features() but never updated (placeholder only). It does not
        # appear in SIMPLIFIED_FEATURES and provides no signal.
        # Fatigue
        "winner_fatigue",
        "loser_fatigue",
        "fatigue_diff",
        # Nationality
        "same_nationality",
        "nat_matchup_win_diff",
        "winner_continent_home",
        "loser_continent_home",
        "continent_advantage_diff",
    ]


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.utils.logger import setup_logger

    setup_logger()

    # Load data
    df = pd.read_parquet("data/processed/matches_clean.parquet")

    # Build advanced features
    df_advanced = build_advanced_features(df)

    advanced_cols = get_advanced_feature_columns()
    logger.info(f"Advanced feature preview:\n{df_advanced[advanced_cols].head(10)}")
    logger.info(f"Feature statistics:\n{df_advanced[advanced_cols].describe()}")
