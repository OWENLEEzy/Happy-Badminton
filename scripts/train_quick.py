"""Quick Mode ensemble model training script (21 features).

Uses ONLY features that Quick Mode can collect from user input.
No form/streak/career data required.

Feature groups (21 total):
  Ranking    : log_rank_diff, rank_closeness
  Context    : category_flag, level_numeric, round_stage, match_month
  Home       : winner_home, loser_home, level_x_home, home_x_closeness
  H2H        : h2h_win_rate_bayes
  Interactions: rank_closeness_x_h2h, gender_x_rank
  Nationality : same_nationality, nat_matchup_win_diff
  Continent  : winner_continent_home, loser_continent_home, continent_advantage_diff
  ELO        : winner_elo, loser_elo, elo_diff

Expected performance (based on 2026-03-08 testing):
  AUC ~0.85, Accuracy ~68%

Run:
    uv run python scripts/train_quick.py
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
from src.data.simplified_features import compute_new_features
from src.models.ensemble_models import StackingEnsemble
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss, accuracy_score

logger = setup_logger()

# Quick Mode ONLY features (21 features - no form/streak/career)
QUICK_FEATURES = [
    # Ranking
    "log_rank_diff",
    "rank_closeness",
    # Context
    "category_flag",
    "level_numeric",
    "round_stage",
    "match_month",
    # Home advantage
    "winner_home",
    "loser_home",
    "level_x_home",
    "home_x_closeness",
    # H2H (Bayesian smoothed)
    "h2h_win_rate_bayes",
    # Interactions
    "rank_closeness_x_h2h",
    "gender_x_rank",
    # Nationality
    "same_nationality",
    "nat_matchup_win_diff",
    # Continent home advantage
    "winner_continent_home",
    "loser_continent_home",
    "continent_advantage_diff",
    # ELO (pre-match)
    "winner_elo",
    "loser_elo",
    "elo_diff",
]


def main() -> None:
    logger.info("============================================================")
    logger.info("Quick Mode Model Training")
    logger.info(
        "Features: {} (Quick Mode only - no form/streak/career)".format(len(QUICK_FEATURES))
    )
    logger.info("============================================================")

    # Load config
    with open(project_root / "config.yaml") as f:
        config = yaml.safe_load(f)

    # Load and preprocess data
    logger.info("Loading data from %s", config["data"]["raw_path"])
    df = load_and_merge(config["data"]["raw_path"])
    logger.info("Loaded %d matches", len(df))

    logger.info("Preprocessing data...")
    df_clean = preprocess_pipeline(df)
    logger.info("After preprocessing: %d matches", len(df_clean))

    # Build advanced features
    logger.info("Building advanced features...")
    df_adv = build_advanced_features(df_clean)
    logger.info("After advanced features: %d matches", len(df_adv))

    # Feature engineering
    logger.info("Engineering features...")
    engineer = FeatureEngineer(df_adv)
    engineer.add_basic_features()
    engineer.add_fatigue_features()

    # Time-based split for fitting scalers (70/15/15 split)
    n = len(engineer.df)
    train_size = int(n * 0.70)
    train_mask = pd.Series([False] * n, index=engineer.df.index)
    train_mask.iloc[:train_size] = True

    logger.info("Fitting scalers on training data (first %d matches)...", train_size)
    engineer.fit_scalers(train_mask)
    engineer.apply_standardization()
    engineer.add_rolling_features()

    df_final = engineer.df.copy().sort_values("match_date").reset_index(drop=True)
    logger.info("After feature engineering: %d matches", len(df_final))

    # Compute simplified features
    logger.info("Computing simplified features...")
    df_final = compute_new_features(df_final)

    # Build nationality pair lookup (training data only)
    logger.info("Building nationality pair lookup...")
    train_end = int(len(df_final) * 0.70)
    nat_lookup = build_nat_pair_lookup(df_final, train_end=train_end, nat_prior=10)
    logger.info("Built lookup with %d nationality pairs", len(nat_lookup))

    # Save nationality lookup
    nat_lookup_path = project_root / "models" / "quick_nat_pair_win_rates.json"
    nat_lookup_path.parent.mkdir(exist_ok=True, parents=True)
    with open(nat_lookup_path, "w") as f:
        json.dump(nat_lookup, f, indent=2)
    logger.info("Saved nationality lookup to %s", nat_lookup_path)

    # Prepare training data
    logger.info("Preparing training data...")
    X = df_final[QUICK_FEATURES].copy()
    y = (df_final["mov_elo_diff"] > 0).astype(int).values

    # Handle missing values
    X.fillna(0, inplace=True)

    # Time-based split (70/15/15)
    n_total = len(X)
    train_end = int(n_total * 0.70)
    val_end = int(n_total * 0.85)

    X_train = X.iloc[:train_end]
    X_val = X.iloc[train_end:val_end]
    X_test = X.iloc[val_end:]

    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_test = y[val_end:]

    logger.info("Train: %d, Val: %d, Test: %d", len(X_train), len(X_val), len(X_test))

    # Train model
    logger.info("Training Quick Mode ensemble...")
    model = StackingEnsemble(random_seed=config["model"]["random_seed"])

    logger.info("Fitting base models...")
    model.train_base_models(X_train, y_train, X_val, y_val)

    logger.info("Training meta model...")
    model.train_meta_model(X_train, y_train, X_val, y_val)

    # Calibrate
    logger.info("Calibrating probabilities...")
    model.calibrate(X_train, y_train, X_val, y_val)

    # Evaluate
    logger.info("Evaluating on test set...")
    y_pred_proba = model.predict_proba_calibrated(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)
    logloss = log_loss(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)

    logger.info("")
    logger.info("============================================================")
    logger.info("Quick Mode Model Performance")
    logger.info("============================================================")
    logger.info("Test Set: %d matches", len(X_test))
    logger.info("AUC:       %.4f", auc)
    logger.info("LogLoss:   %.4f", logloss)
    logger.info("Brier:     %.4f", brier)
    logger.info("Accuracy:  %.4f (%.1f%%)", acc, acc * 100)
    logger.info("============================================================")

    # Save model
    model_path = project_root / "models" / "quick_ensemble.pkl"
    joblib.dump(model, model_path)
    logger.info("Saved model to %s", model_path)

    # Save results
    results = {
        "model_type": "QuickEnsemble",
        "version": "1.0",
        "trained_at": datetime.now().isoformat(),
        "n_features": len(QUICK_FEATURES),
        "features": QUICK_FEATURES,
        "neutral_values": {
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
            "h2h_win_rate_bayes": 0.5,
            "rank_closeness_x_h2h": 0.0,
            "gender_x_rank": 0.0,
            "same_nationality": 0.0,
            "nat_matchup_win_diff": 0.0,
            "winner_continent_home": 0.0,
            "loser_continent_home": 0.0,
            "continent_advantage_diff": 0.0,
            "winner_elo": 1616.0,
            "loser_elo": 1616.0,
            "elo_diff": 0.0,
        },
        "metrics": {
            "test_auc": float(auc),
            "test_logloss": float(logloss),
            "test_brier": float(brier),
            "test_accuracy": float(acc),
        },
        "train_size": int(train_size),
        "val_size": int(val_end - train_end),
        "test_size": int(n_total - val_end),
    }

    results_path = project_root / "models" / "quick_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results to %s", results_path)

    # Feature importance (from LightGBM)
    try:
        lgb = model.base_models[0]
        importances = lgb.feature_importances_
        feature_imp = dict(zip(QUICK_FEATURES, importances.tolist()))
        imp_path = project_root / "models" / "quick_feature_importance.json"
        with open(imp_path, "w") as f:
            json.dump(feature_imp, f, indent=2)
        logger.info("Saved feature importances to %s", imp_path)
    except Exception as e:
        logger.warning("Could not extract feature importances: %s", e)

    logger.info("")
    logger.info("✅ Quick Mode model training complete!")
    logger.info("   Model: %s", model_path)
    logger.info("   Results: %s", results_path)


if __name__ == "__main__":
    main()
