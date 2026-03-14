"""Shared Phase-1 non-linear and interaction features for training scripts.

This module contains feature engineering logic shared between train_simplified.py
and train_set_count.py, including round stage mapping and non-linear transformations.
"""

import numpy as np
import pandas as pd


ROUND_STAGE_MAP: dict[str, int] = {
    "Group A": 0,
    "Group B": 0,
    "Group P": 0,
    "Q-Round 1": 0,
    "Q-Round 2": 1,
    "Q-Round 3": 2,
    "Q-Round 4": 2,
    "Q-Quarter-final": 3,
    "Round 1": 3,
    "Round 2": 4,
    "Round 3": 5,
    "Round 4": 5,
    "Quarter-final": 6,
    "Semi-final": 7,
    "Final": 8,
}
ROUND_STAGE_DEFAULT = 4


def compute_new_features(df: pd.DataFrame, h2h_prior: int = 5) -> pd.DataFrame:
    """Compute Phase-1 non-linear and interaction features (pre-match only).

    Called after build_advanced_features() and FeatureEngineer pipeline so
    base columns (winner_form_5, loser_form_5, winner_streak, etc.) already exist.

    Args:
        df: DataFrame with base feature columns already computed.

    Returns:
        DataFrame with Phase-1 features added.
    """
    df = df.copy()

    # Context features
    df["round_stage"] = (
        df["round"].map(lambda x: ROUND_STAGE_MAP.get(x, ROUND_STAGE_DEFAULT)).astype(int)
    )
    df["match_month"] = pd.to_datetime(df["match_date"]).dt.month

    # Non-linear streak (cap at 5, preserving sign)
    df["streak_capped_w"] = np.sign(df["winner_streak"]) * np.minimum(
        np.abs(df["winner_streak"]), 5
    )
    df["streak_capped_l"] = np.sign(df["loser_streak"]) * np.minimum(np.abs(df["loser_streak"]), 5)
    df["streak_capped_diff"] = df["streak_capped_w"] - df["streak_capped_l"]

    # Career stage U-curve (50-100 matches = highest upset potential = 2.0)
    def _career_stage(n: float) -> float:
        if n <= 20:
            return 0.0
        elif n <= 50:
            return 1.0
        elif n <= 100:
            return 2.0
        elif n <= 200:
            return 1.5
        elif n <= 500:
            return 0.5
        else:
            return 0.0

    df["career_stage"] = df["total_player_matches"].apply(_career_stage)

    # Compute loser's total career matches (wins + losses in all prior rows).
    # Time-safe: record before updating so no future info leaks into earlier rows.
    # df must already be sorted by match_date (enforced by prepare_data caller).
    _career_count: dict[str, int] = {}
    _loser_total = np.empty(len(df), dtype=np.int64)
    for _i, (_w, _l) in enumerate(zip(df["winner_id"].tolist(), df["loser_id"].tolist())):
        _loser_total[_i] = _career_count.get(_l, 0)
        _career_count[_w] = _career_count.get(_w, 0) + 1
        _career_count[_l] = _career_count.get(_l, 0) + 1
    df["career_stage_l"] = [_career_stage(int(n)) for n in _loser_total]

    # Form momentum (recent trend: form5 - form10)
    w10 = df["winner_form_10"] if "winner_form_10" in df.columns else df["winner_form_5"]
    l10 = df["loser_form_10"] if "loser_form_10" in df.columns else df["loser_form_5"]
    df["form_momentum_w"] = df["winner_form_5"] - w10
    df["form_momentum_l"] = df["loser_form_5"] - l10
    df["momentum_diff"] = df["form_momentum_w"] - df["form_momentum_l"]

    # Rank closeness (inverse of absolute log rank diff)
    rank_abs = np.abs(df["log_rank_diff"])
    df["rank_closeness"] = 1.0 / (1.0 + rank_abs)

    # Interaction: rank difference x form difference
    form_diff_10 = df["form_diff_10"] if "form_diff_10" in df.columns else 0.0
    df["rank_x_form_diff"] = df["log_rank_diff"] * form_diff_10

    # H2H Bayesian smoothing (prior from config.yaml)
    h2h_total = df["h2h_matches"].clip(lower=0)
    h2h_wins = df["h2h_win_rate"] * h2h_total
    df["h2h_win_rate_bayes"] = (h2h_wins + h2h_prior * 0.5) / (h2h_total + h2h_prior)

    # Interactions with closeness and H2H
    df["rank_closeness_x_h2h"] = df["rank_closeness"] * (df["h2h_win_rate_bayes"] - 0.5)
    df["gender_x_rank"] = df["category_flag"] * df["log_rank_diff"]
    df["home_x_closeness"] = df["winner_home"] * df["rank_closeness"]

    return df
