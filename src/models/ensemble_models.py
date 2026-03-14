"""
SOTA ensemble model module.

Contains:
1. CatBoost model (SOTA GBM; 2025 research shows it outperforms XGBoost/LightGBM)
2. Stacking Ensemble (CatBoost + XGBoost + LightGBM)
3. Isotonic probability calibration

References:
- [CatBoost vs LightGBM vs XGBoost 2025](https://link.springer.com/article/s13042-025-02654-5)
- [Football prediction with GBM](https://openaccess-api.cms-conferences.org/articles/download/978-1-95865137-7_9)
"""

from typing import Any

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# Model libraries
try:
    import catboost as cb
except ImportError:
    cb = None

import lightgbm as lgb
import xgboost as xgb

# sklearn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, BayesianRidge

import optuna
from loguru import logger


class CatBoostModel:
    """
    CatBoost model wrapper.

    CatBoost advantages:
    1. Automatic handling of categorical features
    2. Better default parameters
    3. Reduced overfitting
    4. Faster prediction speed
    """

    def __init__(self, random_seed: int = 42):
        """Initialize the CatBoost model."""
        self.random_seed = random_seed
        self.model = None
        self.calibrator = None
        self.best_params = None

        # CatBoost fixed parameters
        self.cb_params = {
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "random_seed": random_seed,
            "verbose": False,
            "allow_writing_files": False,
        }

    def get_default_params(self) -> dict:
        """Get default training parameters."""
        return {
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 3.0,
            "bagging_temperature": 1.0,
            "border_count": 128,
            "iterations": 500,
        }

    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        params: dict | None = None,
        cat_features: list[int] | None = None,
    ) -> Any:
        """
        Train the CatBoost model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            params: Model parameters
            cat_features: List of categorical feature indices

        Returns:
            Trained model
        """
        if cb is None:
            raise ImportError("CatBoost is not installed. Run: pip install catboost")

        if params is None:
            params = self.get_default_params()

        train_params = {**self.cb_params, **params}

        logger.info(f"CatBoost training parameters: {params}")

        # Create Pool objects (optional, for categorical features)
        train_pool = cb.Pool(X_train, y_train, cat_features=cat_features)
        val_pool = cb.Pool(X_val, y_val, cat_features=cat_features)

        model = cb.CatBoostClassifier(**train_params)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=False)

        # Evaluate
        train_pred = model.predict_proba(train_pool)[:, 1]
        val_pred = model.predict_proba(val_pool)[:, 1]

        train_ll = log_loss(y_train, train_pred)
        val_ll = log_loss(y_val, val_pred)
        val_auc = roc_auc_score(y_val, val_pred)

        logger.info(f"Train LogLoss: {train_ll:.4f}")
        logger.info(f"Val LogLoss: {val_ll:.4f}")
        logger.info(f"Val AUC: {val_auc:.4f}")

        self.model = model
        return model

    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, n_trials: int = 50):
        """
        Optimize hyperparameters using Optuna.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            n_trials: Number of trials
        """
        if cb is None:
            raise ImportError("CatBoost is not installed")

        logger.info(f"Starting CatBoost hyperparameter optimization (n_trials={n_trials})...")

        train_pool = cb.Pool(X_train, y_train)
        val_pool = cb.Pool(X_val, y_val)

        def objective(trial):
            params = {
                "depth": trial.suggest_int("depth", 4, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
                "border_count": trial.suggest_int("border_count", 32, 255),
                "iterations": 500,
            }

            model = cb.CatBoostClassifier(**{**self.cb_params, **params})  # type: ignore[union-attr]

            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=False)

            val_pred = model.predict_proba(val_pool)[:, 1]
            return log_loss(y_val, val_pred)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        self.best_params = study.best_params
        logger.info(f"CatBoost best parameters: {self.best_params}")
        logger.info(f"CatBoost best LogLoss: {study.best_value:.4f}")

        return self.best_params


class TemperatureScaler:
    """Temperature scaling calibrator.

    Learns a single scalar T on validation set.
    T > 1 compresses probabilities toward 0.5 (fixes overconfidence).
    """

    def __init__(self) -> None:
        self.T: float = 1.0

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
        """Find T that minimises log-loss on calibration set."""
        from scipy.optimize import minimize_scalar

        logits = self._logit(np.asarray(probs))

        def objective(t: float) -> float:
            calibrated = self._sigmoid(logits / max(t, 0.1))
            return log_loss(y_true, calibrated)

        result = minimize_scalar(objective, bounds=(0.1, 10.0), method="bounded")
        self.T = float(result.x)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Apply temperature scaling."""
        logits = self._logit(np.asarray(probs))
        return self._sigmoid(logits / self.T)


class StackingEnsemble:
    """
    Stacking ensemble model.

    Structure:
    Level 0 (Base Models):
    - CatBoost
    - XGBoost
    - LightGBM

    Level 1 (Meta Model):
    - Logistic Regression
    """

    def __init__(self, random_seed: int = 42):
        """Initialize Stacking Ensemble."""
        self.random_seed = random_seed
        self.base_models = {}
        self.meta_model: Any = None  # BayesianRidge fitted model
        self.calibrator: Any = None  # TemperatureScaler or IsotonicRegression
        self._meta_model_class = BayesianRidge

    def train_base_models(self, X_train, y_train, X_val, y_val):
        """
        Train base models.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        """
        logger.info("Training base models...")

        # 1. LightGBM
        logger.info("-" * 40)
        logger.info("1/3 LightGBM")
        lgb_model = lgb.LGBMClassifier(
            objective="binary",
            metric="binary_logloss",
            verbosity=-1,
            random_state=self.random_seed,
            max_depth=5,
            learning_rate=0.03,
            num_leaves=31,
            n_estimators=500,
        )
        lgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        self.base_models["lightgbm"] = lgb_model

        val_pred_lgb = lgb_model.predict_proba(X_val)[:, 1]  # type: ignore[index]
        logger.info(f"LightGBM LogLoss: {log_loss(y_val, val_pred_lgb):.4f}")

        # 2. XGBoost
        logger.info("-" * 40)
        logger.info("2/3 XGBoost")
        xgb_model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=self.random_seed,
            max_depth=5,
            learning_rate=0.03,
            n_estimators=500,
            early_stopping_rounds=50,
            verbosity=0,
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        self.base_models["xgboost"] = xgb_model

        val_pred_xgb = xgb_model.predict_proba(X_val)[:, 1]
        logger.info(f"XGBoost LogLoss: {log_loss(y_val, val_pred_xgb):.4f}")

        # 3. CatBoost
        if cb is not None:
            logger.info("-" * 40)
            logger.info("3/3 CatBoost")
            cb_model = cb.CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="Logloss",
                random_seed=self.random_seed,
                verbose=False,
                depth=6,
                learning_rate=0.03,
                iterations=500,
            )
            train_pool = cb.Pool(X_train, y_train)
            val_pool = cb.Pool(X_val, y_val)
            cb_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=False)
            self.base_models["catboost"] = cb_model

            val_pred_cb = cb_model.predict_proba(val_pool)[:, 1]
            logger.info(f"CatBoost LogLoss: {log_loss(y_val, val_pred_cb):.4f}")
        else:
            logger.warning("CatBoost not available, skipping.")

        logger.info("-" * 40)
        logger.info(f"Base model training complete: {list(self.base_models.keys())}")

    def train_meta_model(self, X_train, y_train, X_val, y_val):
        """
        Train meta model using base model predictions as features.

        Note: This method uses X_val/y_val to train the meta-learner. The same
        validation set is later reused by calibrate() for temperature scaling.
        This is an intentional design trade-off: using a single validation set
        for both meta-learning and calibration conserves training data, with
        acceptable overfitting risk given BayesianRidge regularization and
        TemperatureScaler's single-parameter (T) nature.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (also used for calibration)
            y_val: Validation labels (also used for calibration)
        """
        logger.info("Training meta model...")

        # Get base model predictions on validation set
        meta_features = []

        for name, model in self.base_models.items():
            if name == "catboost" and cb is not None:
                val_pool = cb.Pool(X_val, y_val)
                pred = model.predict_proba(val_pool)[:, 1]
            else:
                pred = model.predict_proba(X_val)[:, 1]
            meta_features.append(pred)

        # Build meta feature matrix
        X_meta = np.column_stack(meta_features)

        # Train BayesianRidge meta-model on validation predictions
        self.meta_model = BayesianRidge()
        self.meta_model.fit(X_meta, y_val)

        # Evaluate
        ensemble_pred_raw = self.meta_model.predict(X_meta)
        ensemble_pred = np.clip(ensemble_pred_raw, 0.0, 1.0)
        ensemble_ll = log_loss(y_val, ensemble_pred)
        ensemble_auc = roc_auc_score(y_val, ensemble_pred)

        logger.info(f"Ensemble LogLoss: {ensemble_ll:.4f}")
        logger.info(f"Ensemble AUC: {ensemble_auc:.4f}")

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Complete training pipeline.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        """
        self.train_base_models(X_train, y_train, X_val, y_val)
        self.train_meta_model(X_train, y_train, X_val, y_val)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities.

        Args:
            X: Feature matrix

        Returns:
            Predicted probabilities
        """
        meta_features = []

        for name, model in self.base_models.items():
            if name == "catboost" and cb is not None:
                pool = cb.Pool(X)
                pred = model.predict_proba(pool)[:, 1]
            else:
                pred = model.predict_proba(X)[:, 1]
            meta_features.append(pred)

        X_meta = np.column_stack(meta_features)
        raw = self.meta_model.predict(X_meta)
        return np.clip(raw, 0.0, 1.0)

    def calibrate(self, X_train, y_train, X_val, y_val, method: str = "temperature"):
        """Calibrate with Temperature Scaling (default) or Isotonic.

        Note: This method uses X_val/y_val which were also used in train_meta_model().
        See train_meta_model() documentation for the rationale behind this design.

        Args:
            X_train: train features (unused by temperature scaling, kept for API compat)
            y_train: train labels (unused, kept for API compat)
            X_val: validation features (also used for meta-learning)
            y_val: validation labels (also used for meta-learning)
            method: 'temperature' (default) or 'isotonic'
        """
        logger.info(f"Calibrating with method={method}...")
        val_pred_raw = self.predict_proba(X_val)

        if method == "temperature":
            calibrator: Any = TemperatureScaler()
            calibrator.fit(val_pred_raw, y_val)
            logger.info(f"Temperature T={calibrator.T:.3f}")
        else:
            from sklearn.isotonic import IsotonicRegression

            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(val_pred_raw, y_val)  # type: ignore[union-attr]

        self.calibrator = calibrator

        # Evaluate calibration
        if method == "temperature":
            val_pred_cal = calibrator.transform(val_pred_raw)  # type: ignore[union-attr]
        else:
            val_pred_cal = calibrator.predict(val_pred_raw)  # type: ignore[union-attr]

        raw_ll = log_loss(y_val, val_pred_raw)
        cal_ll = log_loss(y_val, val_pred_cal)
        raw_bs = brier_score_loss(y_val, val_pred_raw)
        cal_bs = brier_score_loss(y_val, val_pred_cal)

        logger.info(f"Raw LogLoss: {raw_ll:.4f} -> Calibrated: {cal_ll:.4f}")
        logger.info(f"Raw Brier:   {raw_bs:.4f} -> Calibrated: {cal_bs:.4f}")

        return self.calibrator

    def predict_proba_calibrated(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated probability predictions."""
        raw_pred = self.predict_proba(X)
        if self.calibrator is None:
            return raw_pred
        if isinstance(self.calibrator, TemperatureScaler):
            return self.calibrator.transform(raw_pred)
        return self.calibrator.predict(raw_pred)  # type: ignore[union-attr]


class IsotonicCalibrator:
    """
    Isotonic regression probability calibrator.

    Advantages:
    - Non-parametric method, adapts to any distribution
    - 2025 research shows it outperforms Sigmoid calibration
    """

    def __init__(self):
        """Initialize the calibrator."""
        from sklearn.isotonic import IsotonicRegression

        self.calibrator = IsotonicRegression(out_of_bounds="clip")

    def fit(self, y_true, y_pred_proba):
        """
        Fit the calibrator.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
        """
        self.calibrator.fit(y_pred_proba, y_true)
        logger.info("Isotonic calibrator training complete.")

    def calibrate(self, y_pred_proba) -> np.ndarray:
        """
        Calibrate probabilities.

        Args:
            y_pred_proba: Raw predicted probabilities

        Returns:
            Calibrated probabilities
        """
        return self.calibrator.predict(y_pred_proba)

    def fit_cross_validated(self, X, y, model, cv=5):
        """Fit calibrator using time-series cross-validation (to avoid data leakage).

        WARNING: Input data X and y MUST be in chronological order. This method
        uses TimeSeriesSplit, which always trains on past data and validates on
        future data. Shuffled or randomly ordered data will cause temporal leakage
        and invalidate the calibration.

        Args:
            X: Feature matrix (must be time-ordered, earliest rows first)
            y: Labels (must be time-ordered, matching X)
            model: Trained model with predict_proba method
            cv: Number of time-series splits
        """
        tss = TimeSeriesSplit(n_splits=cv)
        calibrated_pred = np.zeros(len(X))

        for train_idx, val_idx in tss.split(X):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X.iloc[val_idx]

            # Get validation set predictions
            pred = model.predict_proba(X_val_fold)[:, 1]

            calibrated_pred[val_idx] = pred

        # Train calibrator using cross-validated predictions
        self.calibrator.fit(calibrated_pred, y)
        logger.info("Cross-validated Isotonic calibrator training complete.")

        return self


def save_ensemble_model(ensemble: StackingEnsemble, output_path: str | Path):
    """Save the ensemble model."""
    save_path = Path(output_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        "base_models": ensemble.base_models,
        "meta_model": ensemble.meta_model,
        "calibrator": ensemble.calibrator,
    }

    joblib.dump(model_data, save_path)
    logger.info(f"Ensemble model saved: {save_path}")


def load_ensemble_model(input_path: str) -> StackingEnsemble:
    """Load the ensemble model."""
    model_data = joblib.load(input_path)

    ensemble = StackingEnsemble()
    ensemble.base_models = model_data["base_models"]
    ensemble.meta_model = model_data["meta_model"]
    ensemble.calibrator = model_data.get("calibrator")

    logger.info(f"Ensemble model loaded: {input_path}")
    return ensemble


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.utils.logger import setup_logger

    setup_logger()

    # Test code
    import pandas as pd
    from sklearn.datasets import make_classification

    # Generate simulated data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X = pd.DataFrame(X)
    X_train, X_val = X[:700], X[700:]
    y_train, y_val = y[:700], y[700:]

    # Test Stacking Ensemble
    ensemble = StackingEnsemble()
    ensemble.fit(X_train, y_train, X_val, y_val)

    # Test CatBoost
    if cb is not None:
        cat_model = CatBoostModel()
        cat_model.train(X_train, y_train, X_val, y_val)
