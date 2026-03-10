"""Simplified ensemble model training script (Phase-1 feature set, 26 features).

Uses only pre-match, user-inputtable features. No player database required.

Feature groups (26 total):
  Ranking    : log_rank_diff, rank_closeness
  Context    : category_flag, level_numeric, round_stage, match_month
  Home       : winner_home, loser_home, level_x_home, home_x_closeness
  Form       : winner_form_5, loser_form_5, form_diff_5, form_diff_10, form_diff_20
  Momentum   : form_momentum_w, form_momentum_l, momentum_diff
  Streak     : streak_capped_w, streak_capped_l, streak_capped_diff
  H2H        : h2h_win_rate_bayes
  Experience : career_stage
  Interactions: rank_x_form_diff, rank_closeness_x_h2h, gender_x_rank

Run:
    uv run python scripts/train_simplified.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import json
from datetime import datetime
import joblib

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from src.data.loader import load_and_merge
from src.data.preprocessor import preprocess_pipeline
from src.data.feature_engineering import FeatureEngineer
from src.data.advanced_features import build_advanced_features, build_nat_pair_lookup
from src.models.ensemble_models import StackingEnsemble
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss, accuracy_score

logger = setup_logger()

# Pre-match features that users can manually input before a match (Phase-1 non-linear + interaction)
SIMPLIFIED_FEATURES = [
    # Ranking
    "log_rank_diff",
    "rank_closeness",
    # Context
    "category_flag",
    "level_numeric",
    "round_stage",
    "match_month",
    # Home
    "winner_home",
    "loser_home",
    "level_x_home",
    "home_x_closeness",
    # Form (windows)
    "winner_form_5",
    "loser_form_5",
    "form_diff_5",
    "form_diff_10",
    "form_diff_20",
    # Form momentum
    "form_momentum_w",
    "form_momentum_l",
    "momentum_diff",
    # Streak (non-linear)
    "streak_capped_w",
    "streak_capped_l",
    "streak_capped_diff",
    # H2H (Bayesian smoothed)
    "h2h_win_rate_bayes",
    # Career stage
    "career_stage",
    "career_stage_l",
    # Interactions
    "rank_x_form_diff",
    "rank_closeness_x_h2h",
    "gender_x_rank",
    # Nationality & continent
    "same_nationality",
    "nat_matchup_win_diff",
    "winner_continent_home",
    "loser_continent_home",
    "continent_advantage_diff",
    # ELO (pre-match, from raw BWF data - user-inputtable from BWF website)
    "winner_elo",
    "loser_elo",
    "elo_diff",
]

# Neutral (baseline) values for driving factor calculation
NEUTRAL_VALUES = {
    "log_rank_diff": 0.0,
    "rank_closeness": 0.5,
    "category_flag": 0.0,
    "level_numeric": 5.0,
    "round_stage": 4.0,
    "match_month": 6.0,
    "winner_home": 0.0,
    "loser_home": 0.0,
    "level_x_home": 0.0,
    "home_x_closeness": 0.0,
    "winner_form_5": 0.5,
    "loser_form_5": 0.5,
    "form_diff_5": 0.0,
    "form_diff_10": 0.0,
    "form_diff_20": 0.0,
    "form_momentum_w": 0.0,
    "form_momentum_l": 0.0,
    "momentum_diff": 0.0,
    "streak_capped_w": 0.0,
    "streak_capped_l": 0.0,
    "streak_capped_diff": 0.0,
    "h2h_win_rate_bayes": 0.5,
    "career_stage": 1.0,
    "career_stage_l": 1.0,
    "rank_x_form_diff": 0.0,
    "rank_closeness_x_h2h": 0.0,
    "gender_x_rank": 0.0,
    # Nationality & continent (neutral = no advantage either way)
    "same_nationality": 0.0,
    "nat_matchup_win_diff": 0.0,
    "winner_continent_home": 0.0,
    "loser_continent_home": 0.0,
    "continent_advantage_diff": 0.0,
    # ELO (neutral = median ELO, equal for both players)
    "winner_elo": 1616.0,
    "loser_elo": 1616.0,
    "elo_diff": 0.0,
}


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
ROUND_STAGE_DEFAULT = 4  # mid-tournament default for unknown values


def compute_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Phase-1 non-linear and interaction features (pre-match only).

    Called after build_advanced_features() and FeatureEngineer pipeline so
    base columns (winner_form_5, loser_form_5, winner_streak, etc.) already exist.
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

    # Rank closeness (1 = equal ranks, 0 = far apart)
    rank_abs = np.abs(df["log_rank_diff"])
    df["rank_closeness"] = 1.0 / (1.0 + rank_abs)

    # Interaction features
    form_diff_10 = df["form_diff_10"] if "form_diff_10" in df.columns else 0.0
    df["rank_x_form_diff"] = df["log_rank_diff"] * form_diff_10

    # Bayesian H2H smoothing (replaces raw h2h_win_rate + h2h_matches)
    prior = 5
    h2h_total = df["h2h_matches"].clip(lower=0)
    h2h_wins = df["h2h_win_rate"] * h2h_total
    df["h2h_win_rate_bayes"] = (h2h_wins + prior * 0.5) / (h2h_total + prior)

    df["rank_closeness_x_h2h"] = df["rank_closeness"] * (df["h2h_win_rate_bayes"] - 0.5)
    df["gender_x_rank"] = df["category_flag"] * df["log_rank_diff"]
    df["home_x_closeness"] = df["winner_home"] * df["rank_closeness"]

    return df


def load_config(config_path: str | Path = "config.yaml") -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        # Try relative to project root
        project_root = Path(__file__).parent.parent
        config_path = project_root / config_path.name
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")
    with config_path.open() as f:
        return yaml.safe_load(f)


def prepare_data(config: dict) -> tuple:
    """Prepare training data using SOTA pipeline, selecting only simplified features."""
    logger.info("=" * 60)
    logger.info("Preparing simplified model training data")
    logger.info("=" * 60)

    df = load_and_merge(config["data"]["raw_path"])
    df_clean = preprocess_pipeline(df)
    df_advanced = build_advanced_features(df_clean)

    engineer = FeatureEngineer(df_advanced)
    engineer.add_basic_features()
    engineer.add_fatigue_features()

    n = len(engineer.df)
    train_size = int(n * 0.7)
    train_mask = pd.Series([False] * n, index=engineer.df.index)
    train_mask.iloc[:train_size] = True

    engineer.fit_scalers(train_mask)
    engineer.apply_standardization()
    engineer.add_rolling_features()

    df_final = engineer.df.copy().sort_values("match_date").reset_index(drop=True)
    df_final = compute_new_features(df_final)  # Phase-1 non-linear and interaction features
    y = (df_final["mov_elo_diff"] > 0).astype(int).values

    available = [f for f in SIMPLIFIED_FEATURES if f in df_final.columns]
    missing = [f for f in SIMPLIFIED_FEATURES if f not in df_final.columns]
    if missing:
        logger.warning(f"Missing features (skipped): {missing}")

    X = df_final[available].copy().fillna(0)

    n_total = len(df_final)
    train_end = int(n_total * 0.70)
    val_end = int(n_total * 0.85)

    X_train = X.iloc[:train_end]
    X_val = X.iloc[train_end:val_end]
    X_test = X.iloc[val_end:]
    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_test = y[val_end:]

    logger.info(f"Feature count: {len(available)}")
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test, available, df_final


def main():
    logger.info("")
    logger.info("╔════════════════════════════════════════╗")
    logger.info("║   Simplified Model Training            ║")
    logger.info("╚════════════════════════════════════════╝")
    logger.info("")

    config = load_config()

    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, df_final = prepare_data(config)

    logger.info("Training StackingEnsemble...")
    ensemble = StackingEnsemble(random_seed=config["model"]["random_seed"])
    ensemble.fit(X_train, y_train, X_val, y_val)

    logger.info("Applying Temperature Scaling calibration...")
    ensemble.calibrate(X_train, y_train, X_val, y_val, method="temperature")

    # Evaluate on test set
    y_pred = ensemble.predict_proba_calibrated(X_test)
    ll = log_loss(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    bs = brier_score_loss(y_test, y_pred)
    acc = accuracy_score(y_test, (y_pred > 0.5).astype(int))

    logger.info("")
    logger.info("╔════════════════════════════════════════╗")
    logger.info("║   Simplified Model Performance         ║")
    logger.info("╠════════════════════════════════════════╣")
    logger.info(f"║  LogLoss:     {ll:.4f}                  ║")
    logger.info(f"║  AUC:         {auc:.4f}                  ║")
    logger.info(f"║  Brier Score: {bs:.4f}                  ║")
    logger.info(f"║  Accuracy:    {acc:.4f}                  ║")
    logger.info("╚════════════════════════════════════════╝")
    logger.info("")

    models_dir = project_root / "models"
    nat_pair_path = models_dir / "nat_pair_win_rates.json"
    model_path = models_dir / "simplified_ensemble.pkl"
    fi_path = models_dir / "simplified_feature_importance.json"
    results_path = models_dir / "simplified_results.json"

    # Save nat pair win rates lookup for production inference
    nat_lookup = build_nat_pair_lookup(df_final)
    with open(nat_pair_path, "w") as f:
        json.dump(nat_lookup, f, indent=2)
    logger.info(
        f"Nationality pair win rate lookup saved: {nat_pair_path} ({len(nat_lookup)} pairs)"
    )

    # Save model
    joblib.dump(ensemble, model_path)
    logger.info(f"Model saved: {model_path}")

    # Extract LGBM feature importances (normalized to sum = 1)
    feature_importance = {}
    lgbm_model = ensemble.base_models.get("lightgbm")
    if lgbm_model is not None and hasattr(lgbm_model, "feature_importances_"):
        importances = lgbm_model.feature_importances_.astype(float)
        total = importances.sum()
        for feat, imp in zip(feature_cols, importances):
            feature_importance[feat] = float(imp / total) if total > 0 else 0.0
        logger.info("Feature importance (Top 5):")
        for feat, imp in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.info(f"  {feat:<30} {imp:.3f}")

    with open(fi_path, "w") as f:
        json.dump(feature_importance, f, indent=2)
    logger.info(f"Feature importance saved: {fi_path}")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_type": "SimplifiedEnsemble",
        "n_features": len(feature_cols),
        "features": feature_cols,
        "neutral_values": NEUTRAL_VALUES,
        "metrics": {
            "log_loss": ll,
            "auc": auc,
            "brier_score": bs,
            "accuracy": acc,
        },
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved: {results_path}")
    logger.info("")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
