"""Model validation script for SimplifiedEnsemble.

Usage:
    uv run python scripts/validate_model.py
    uv run python scripts/validate_model.py --model simplified
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.advanced_features import build_advanced_features
from src.data.feature_engineering import FeatureEngineer
from src.data.loader import load_and_merge
from src.data.preprocessor import preprocess_pipeline
from src.utils.logger import setup_logger

logger = setup_logger()

# Mirrors SIMPLIFIED_FEATURES and compute_new_features() in train_simplified.py.
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


def load_config(config_path: str = "config.yaml") -> dict:
    p = Path(config_path)
    if not p.exists():
        # Fall back to path relative to this script's project root
        p = Path(__file__).parent.parent / Path(config_path).name
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")
    with p.open() as f:
        return yaml.safe_load(f)


def compute_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute Phase-1 features — must stay in sync with train_simplified.py."""
    df = df.copy()

    df["round_stage"] = (
        df["round"].map(lambda x: ROUND_STAGE_MAP.get(x, ROUND_STAGE_DEFAULT)).astype(int)
    )
    df["match_month"] = pd.to_datetime(df["match_date"]).dt.month

    df["streak_capped_w"] = np.sign(df["winner_streak"]) * np.minimum(
        np.abs(df["winner_streak"]), 5
    )
    df["streak_capped_l"] = np.sign(df["loser_streak"]) * np.minimum(np.abs(df["loser_streak"]), 5)
    df["streak_capped_diff"] = df["streak_capped_w"] - df["streak_capped_l"]

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

    # Compute loser's total career matches using time-safe per-row accumulation.
    # df must already be sorted by match_date (enforced by prepare_simplified_data caller).
    _career_count: dict[str, int] = {}
    _loser_total = np.empty(len(df), dtype=np.int64)
    for _i, (_w, _l) in enumerate(zip(df["winner_id"].tolist(), df["loser_id"].tolist())):
        _loser_total[_i] = _career_count.get(_l, 0)
        _career_count[_w] = _career_count.get(_w, 0) + 1
        _career_count[_l] = _career_count.get(_l, 0) + 1
    df["career_stage_l"] = [_career_stage(int(n)) for n in _loser_total]

    w10 = df["winner_form_10"] if "winner_form_10" in df.columns else df["winner_form_5"]
    l10 = df["loser_form_10"] if "loser_form_10" in df.columns else df["loser_form_5"]
    df["form_momentum_w"] = df["winner_form_5"] - w10
    df["form_momentum_l"] = df["loser_form_5"] - l10
    df["momentum_diff"] = df["form_momentum_w"] - df["form_momentum_l"]

    rank_abs = np.abs(df["log_rank_diff"])
    df["rank_closeness"] = 1.0 / (1.0 + rank_abs)

    form_diff_10 = df["form_diff_10"] if "form_diff_10" in df.columns else 0.0
    df["rank_x_form_diff"] = df["log_rank_diff"] * form_diff_10

    prior = 5
    h2h_total = df["h2h_matches"].clip(lower=0)
    h2h_wins = df["h2h_win_rate"] * h2h_total
    df["h2h_win_rate_bayes"] = (h2h_wins + prior * 0.5) / (h2h_total + prior)

    df["rank_closeness_x_h2h"] = df["rank_closeness"] * (df["h2h_win_rate_bayes"] - 0.5)
    df["gender_x_rank"] = df["category_flag"] * df["log_rank_diff"]
    df["home_x_closeness"] = df["winner_home"] * df["rank_closeness"]

    return df


def prepare_simplified_data(config: dict, feature_cols: list[str]) -> tuple:
    """Rebuild the test split using the same pipeline as train_simplified.py."""
    logger.info("Preparing SimplifiedEnsemble test data...")

    df = load_and_merge(config["data"]["raw_path"])
    df_clean = preprocess_pipeline(df)
    df_advanced = build_advanced_features(df_clean)

    engineer = FeatureEngineer(df_advanced)
    engineer.add_basic_features()
    engineer.add_fatigue_features()

    n = len(engineer.df)
    test_size: float = config["model"]["test_size"]
    train_size = int(n * (1.0 - 2 * test_size))
    train_mask = pd.Series([False] * n, index=engineer.df.index)
    train_mask.iloc[:train_size] = True

    engineer.fit_scalers(train_mask)
    engineer.apply_standardization()
    engineer.add_rolling_features()

    df_final = engineer.df.copy().sort_values("match_date").reset_index(drop=True)
    df_final = compute_new_features(df_final)
    y = (df_final["mov_elo_diff"] > 0).astype(int).values

    available = [f for f in feature_cols if f in df_final.columns]
    X = df_final[available].copy().fillna(0)

    n_total = len(df_final)
    val_end = int(n_total * (1.0 - test_size))

    X_test = X.iloc[val_end:]
    y_test = y[val_end:]

    logger.info(f"Test set: {X_test.shape}")
    return X_test, y_test


def validate_model(model, X_test: pd.DataFrame, y_test: np.ndarray, model_name: str) -> dict:
    """Run full evaluation and print report."""
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"{model_name} Validation Report")
    logger.info("=" * 60)

    if hasattr(model, "predict_proba_calibrated"):
        y_pred_prob = model.predict_proba_calibrated(X_test)
    elif hasattr(model, "predict_proba"):
        raw = model.predict_proba(X_test)
        y_pred_prob = raw[:, 1] if raw.ndim > 1 else raw
    else:
        raise ValueError("Model must have predict_proba or predict_proba_calibrated")

    y_pred = (y_pred_prob > 0.5).astype(int)

    metrics = {
        "log_loss": log_loss(y_test, y_pred_prob),
        "roc_auc": roc_auc_score(y_test, y_pred_prob),
        "brier_score": brier_score_loss(y_test, y_pred_prob),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0.0),  # type: ignore[call-overload]
        "recall": recall_score(y_test, y_pred, zero_division=0.0),  # type: ignore[call-overload]
        "f1": f1_score(y_test, y_pred, zero_division=0.0),  # type: ignore[call-overload]
    }

    logger.info("")
    logger.info("[Probability metrics]")
    logger.info(f"  LogLoss:     {metrics['log_loss']:.4f}")
    logger.info(f"  Brier Score: {metrics['brier_score']:.4f}")
    logger.info(f"  ROC AUC:     {metrics['roc_auc']:.4f}")

    logger.info("")
    logger.info("[Classification metrics] (threshold=0.5)")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1']:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    logger.info("")
    logger.info("[Confusion matrix]")
    logger.info(f"  TN={tn:4d}  FP={fp:4d}")
    logger.info(f"  FN={fn:4d}  TP={tp:4d}")
    logger.info(f"  Winner predicted correctly: {tp}/{tp + fn} = {tp / (tp + fn) * 100:.1f}%")
    logger.info(f"  Loser predicted correctly:  {tn}/{tn + fp} = {tn / (tn + fp) * 100:.1f}%")
    logger.info("")
    logger.info("=" * 60)

    return metrics


def save_results(
    metrics: dict, model_name: str, output_path: str = "models/validation_results.json"
) -> None:
    try:
        with open(output_path) as f:
            all_results = json.load(f)
    except FileNotFoundError:
        all_results = {}

    all_results[model_name] = {
        "metrics": {k: float(v) for k, v in metrics.items()},
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Results saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate trained SimplifiedEnsemble")
    parser.add_argument(
        "--model",
        type=str,
        choices=["simplified"],
        default="simplified",
        help="Model to validate (default: simplified)",
    )
    _ = parser.parse_args()

    config = load_config()

    model_path = "models/simplified_ensemble.pkl"
    results_path = "models/simplified_results.json"

    logger.info("")
    logger.info("╔════════════════════════════════════════╗")
    logger.info("║   SimplifiedEnsemble Validation        ║")
    logger.info("╚════════════════════════════════════════╝")

    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded: {model_path}")
    except FileNotFoundError:
        logger.error(f"Model not found: {model_path}")
        logger.error("Run: uv run python scripts/train_simplified.py")
        sys.exit(1)

    try:
        with open(results_path) as f:
            results = json.load(f)
        feature_cols = results["features"]
        logger.info(f"Features: {len(feature_cols)}")
    except FileNotFoundError:
        logger.error(f"Results file not found: {results_path}")
        sys.exit(1)

    X_test, y_test = prepare_simplified_data(config, feature_cols)
    metrics = validate_model(model, X_test, y_test, "SimplifiedEnsemble")
    save_results(metrics, "simplified")


if __name__ == "__main__":
    main()
