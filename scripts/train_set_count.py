"""Set count prediction model training script.

Predicts whether a match goes to 2 sets (2-0) or 3 sets (2-1).
Uses the same 26 pre-match features as SimplifiedEnsemble.

Target: sets_played == 3  (1 = three sets played, 0 = two sets played)

Run:
    uv run python scripts/train_set_count.py
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
from src.data.advanced_features import build_advanced_features
from src.models.ensemble_models import StackingEnsemble
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss, accuracy_score

logger = setup_logger()

# Same 26 features as SimplifiedEnsemble
SET_COUNT_FEATURES = [
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
    # Interactions
    "rank_x_form_diff",
    "rank_closeness_x_h2h",
    "gender_x_rank",
    # Historical 3-set rate (pre-match, shift-safe)
    "winner_3set_rate",
    "loser_3set_rate",
    "threeset_rate_diff",
]

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


def compute_3set_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-player historical 3-set match rate using a shift-safe Bayesian average.

    Uses shift(1) cumulative sum so that the current match result is never included
    in the feature seen at prediction time.  First-match prior = 0.40 (population average).
    """
    df = df.copy()
    PRIOR_3SET: float = 0.40  # ~40% of matches go to 3 sets historically
    PRIOR_N: int = 5

    # Build long format: each match generates one row per player (winner OR loser)
    winner_rows = df[["match_date", "winner_id", "sets_played"]].copy()
    winner_rows = winner_rows.rename(columns={"winner_id": "player_id"})
    winner_rows["role"] = "winner"
    winner_rows["df_idx"] = df.index
    winner_rows["is_3set"] = (winner_rows["sets_played"] == 3).astype(int)

    loser_rows = df[["match_date", "loser_id", "sets_played"]].copy()
    loser_rows = loser_rows.rename(columns={"loser_id": "player_id"})
    loser_rows["role"] = "loser"
    loser_rows["df_idx"] = df.index
    loser_rows["is_3set"] = (loser_rows["sets_played"] == 3).astype(int)

    combined = (
        pd.concat(
            [
                winner_rows[["match_date", "player_id", "is_3set", "role", "df_idx"]],
                loser_rows[["match_date", "player_id", "is_3set", "role", "df_idx"]],
            ],
            ignore_index=True,
        )
        .sort_values(["player_id", "match_date", "df_idx"])
        .reset_index(drop=True)
    )

    # cumcount gives 0-based ordinal within each player group (= number of prior matches)
    # shift(1).cumsum() gives the count of 3-set matches BEFORE the current match
    combined["cum_3set"] = combined.groupby("player_id")["is_3set"].transform(
        lambda x: x.shift(1).cumsum().fillna(0)
    )
    combined["cum_n"] = combined.groupby("player_id").cumcount()

    combined["threeset_rate"] = (combined["cum_3set"] + PRIOR_3SET * PRIOR_N) / (
        combined["cum_n"] + PRIOR_N
    )

    winner_3set = combined[combined["role"] == "winner"].set_index("df_idx")["threeset_rate"]
    loser_3set = combined[combined["role"] == "loser"].set_index("df_idx")["threeset_rate"]

    df["winner_3set_rate"] = winner_3set.reindex(df.index).values
    df["loser_3set_rate"] = loser_3set.reindex(df.index).values
    df["threeset_rate_diff"] = df["winner_3set_rate"] - df["loser_3set_rate"]

    return df


def compute_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Phase-1 non-linear and interaction features (pre-match only)."""
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


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_data(config: dict) -> tuple:
    """Prepare training data, using sets_played as target."""
    logger.info("=" * 60)
    logger.info("准备局数预测模型训练数据")
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
    df_final = compute_new_features(df_final)
    df_final = compute_3set_rates(df_final)

    # Target: 1 = three sets played (2-1), 0 = two sets played (2-0)
    y = (df_final["sets_played"] == 3).astype(int).values

    available = [f for f in SET_COUNT_FEATURES if f in df_final.columns]
    missing = [f for f in SET_COUNT_FEATURES if f not in df_final.columns]
    if missing:
        logger.warning(f"缺少特征（已跳过）: {missing}")

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

    logger.info(f"特征数: {len(available)}")
    logger.info(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
    logger.info(f"训练集局数分布 — 2局: {(y_train == 0).sum()}, 3局: {(y_train == 1).sum()}")

    return X_train, X_val, X_test, y_train, y_val, y_test, available


def main() -> None:
    logger.info("")
    logger.info("╔════════════════════════════════════════╗")
    logger.info("║   局数预测模型训练 (Set Count)         ║")
    logger.info("╚════════════════════════════════════════╝")
    logger.info("")

    config = load_config()

    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = prepare_data(config)

    logger.info("训练 StackingEnsemble (局数预测)...")
    ensemble = StackingEnsemble(random_seed=config["model"]["random_seed"])
    ensemble.fit(X_train, y_train, X_val, y_val)

    logger.info("Applying Temperature Scaling calibration...")
    ensemble.calibrate(X_train, y_train, X_val, y_val, method="temperature")

    y_pred = ensemble.predict_proba_calibrated(X_test)
    ll = log_loss(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    bs = brier_score_loss(y_test, y_pred)
    acc = accuracy_score(y_test, (y_pred > 0.5).astype(int))

    logger.info("")
    logger.info("╔════════════════════════════════════════╗")
    logger.info("║   局数预测模型性能                     ║")
    logger.info("╠════════════════════════════════════════╣")
    logger.info(f"║  LogLoss:     {ll:.4f}                  ║")
    logger.info(f"║  AUC:         {auc:.4f}                  ║")
    logger.info(f"║  Brier Score: {bs:.4f}                  ║")
    logger.info(f"║  Accuracy:    {acc:.4f}                  ║")
    logger.info("╚════════════════════════════════════════╝")
    logger.info("")

    model_path = "models/set_count_model.pkl"
    joblib.dump(ensemble, model_path)
    logger.info(f"模型已保存: {model_path}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "model_type": "SetCountEnsemble",
        "target": "sets_played == 3",
        "n_features": len(feature_cols),
        "features": feature_cols,
        "metrics": {
            "log_loss": ll,
            "auc": auc,
            "brier_score": bs,
            "accuracy": acc,
        },
    }
    results_path = "models/set_count_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"结果已保存: {results_path}")
    logger.info("")
    logger.info("局数预测模型训练完成!")


if __name__ == "__main__":
    main()
