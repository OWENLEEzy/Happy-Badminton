"""Comprehensive tests for ensemble_models.py - CatBoost, StackingEnsemble, IsotonicCalibrator."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import tempfile

import numpy as np
import pandas as pd
import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.ensemble_models import (
    CatBoostModel,
    StackingEnsemble,
    IsotonicCalibrator,
    TemperatureScaler,
)


# =============================================================================
# CatBoostModel Tests
# =============================================================================


class TestCatBoostModel:
    """Test suite for CatBoostModel wrapper."""

    @pytest.fixture
    def catboost_model(self):
        """Fresh CatBoostModel instance for each test."""
        return CatBoostModel(random_seed=42)

    @pytest.fixture
    def sample_data(self):
        """Create synthetic training/validation data."""
        from sklearn.datasets import make_classification

        X_train, y_train = make_classification(n_samples=200, n_features=10, random_state=42)
        X_val, y_val = make_classification(n_samples=50, n_features=10, random_state=43)
        return (
            pd.DataFrame(X_train),
            y_train,
            pd.DataFrame(X_val),
            y_val,
        )

    def test_init_creates_model_attributes(self, catboost_model):
        """CatBoostModel should initialize with correct attributes."""
        assert catboost_model.random_seed == 42
        assert catboost_model.model is None
        assert catboost_model.calibrator is None
        assert catboost_model.best_params is None

    def test_init_sets_cb_params_correctly(self, catboost_model):
        """CatBoostModel should set fixed CatBoost parameters."""
        expected_params = {
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "random_seed": 42,
            "verbose": False,
            "allow_writing_files": False,
        }
        assert catboost_model.cb_params == expected_params

    def test_get_default_params_returns_dict(self, catboost_model):
        """get_default_params should return parameters dictionary."""
        params = catboost_model.get_default_params()
        assert isinstance(params, dict)
        assert "depth" in params
        assert "learning_rate" in params
        assert "iterations" in params

    def test_default_params_are_reasonable(self, catboost_model):
        """Default parameters should be within reasonable ranges."""
        params = catboost_model.get_default_params()
        assert 3 <= params["depth"] <= 10
        assert 0.01 <= params["learning_rate"] <= 0.3
        assert 100 <= params["iterations"] <= 1000

    @pytest.mark.skipif(not hasattr(CatBoostModel, "train"), reason="CatBoost not installed")
    def test_train_returns_fitted_model(self, catboost_model, sample_data):
        """train() should return a fitted model."""
        X_train, y_train, X_val, y_val = sample_data
        result = catboost_model.train(X_train, y_train, X_val, y_val)
        assert result is not None

    @pytest.mark.skipif(not hasattr(CatBoostModel, "train"), reason="CatBoost not installed")
    def test_train_sets_model_attribute(self, catboost_model, sample_data):
        """train() should set self.model attribute."""
        X_train, y_train, X_val, y_val = sample_data
        catboost_model.train(X_train, y_train, X_val, y_val)
        assert catboost_model.model is not None

    @pytest.mark.skipif(
        not hasattr(CatBoostModel, "optimize_hyperparameters"),
        reason="CatBoost not installed",
    )
    def test_optimize_hyperparameters_returns_best_params(self, catboost_model, sample_data):
        """optimize_hyperparameters should return best parameters."""
        X_train, y_train, X_val, y_val = sample_data
        best_params = catboost_model.optimize_hyperparameters(
            X_train, y_train, X_val, y_val, n_trials=2
        )
        assert isinstance(best_params, dict)
        assert "depth" in best_params or "learning_rate" in best_params


# =============================================================================
# StackingEnsemble Training Tests
# =============================================================================


class TestStackingEnsembleTraining:
    """Test suite for StackingEnsemble training methods."""

    @pytest.fixture
    def ensemble(self):
        """Fresh StackingEnsemble instance."""
        return StackingEnsemble(random_seed=42)

    @pytest.fixture
    def sample_data(self):
        """Create synthetic training data."""
        from sklearn.datasets import make_classification

        X_train, y_train = make_classification(n_samples=300, n_features=15, random_state=42)
        X_val, y_val = make_classification(n_samples=100, n_features=15, random_state=43)
        return pd.DataFrame(X_train), y_train, pd.DataFrame(X_val), y_val

    def test_init_creates_base_models_dict(self, ensemble):
        """StackingEnsemble should initialize with empty base_models dict."""
        assert ensemble.base_models == {}
        assert ensemble.meta_model is None
        assert ensemble.calibrator is None
        assert ensemble.random_seed == 42

    def test_train_base_models_populates_dict(self, ensemble, sample_data):
        """train_base_models should populate self.base_models with 3 models."""
        X_train, y_train, X_val, y_val = sample_data
        ensemble.train_base_models(X_train, y_train, X_val, y_val)
        assert len(ensemble.base_models) == 3
        assert "lightgbm" in ensemble.base_models
        assert "xgboost" in ensemble.base_models
        assert "catboost" in ensemble.base_models

    def test_train_base_models_models_have_predict(self, ensemble, sample_data):
        """Trained base models should have predict_proba method."""
        X_train, y_train, X_val, y_val = sample_data
        ensemble.train_base_models(X_train, y_train, X_val, y_val)
        for name, model in ensemble.base_models.items():
            assert hasattr(model, "predict_proba"), f"{name} missing predict_proba"

    def test_train_meta_model_creates_bayesian_ridge(self, ensemble, sample_data):
        """train_meta_model should create BayesianRidge meta-learner."""
        X_train, y_train, X_val, y_val = sample_data
        ensemble.train_base_models(X_train, y_train, X_val, y_val)
        ensemble.train_meta_model(X_train, y_train, X_val, y_val)
        from sklearn.linear_model import BayesianRidge

        assert isinstance(ensemble.meta_model, BayesianRidge)

    def test_fit_is_convenience_method(self, ensemble, sample_data):
        """fit() should call train_base_models and train_meta_model."""
        X_train, y_train, X_val, y_val = sample_data
        ensemble.fit(X_train, y_train, X_val, y_val)
        assert len(ensemble.base_models) == 3
        assert ensemble.meta_model is not None


# =============================================================================
# StackingEnsemble Calibration Tests
# =============================================================================


class TestStackingEnsembleCalibration:
    """Test suite for probability calibration methods."""

    @pytest.fixture
    def trained_ensemble(self):
        """Create a trained ensemble for calibration tests."""
        from sklearn.datasets import make_classification

        X_train, y_train = make_classification(n_samples=200, n_features=10, random_state=42)
        X_val, y_val = make_classification(n_samples=50, n_features=10, random_state=43)
        X_train_df = pd.DataFrame(X_train)
        X_val_df = pd.DataFrame(X_val)

        ensemble = StackingEnsemble(random_seed=42)
        ensemble.fit(X_train_df, y_train, X_val_df, y_val)
        return ensemble, X_train_df, y_train, X_val_df, y_val

    def test_calibrate_with_temperature_sets_calibrator(self, trained_ensemble):
        """calibrate(method='temperature') should set calibrator attribute."""
        ensemble, X_train, y_train, X_val, y_val = trained_ensemble
        ensemble.calibrate(X_train, y_train, X_val, y_val, method="temperature")
        assert ensemble.calibrator is not None
        assert isinstance(ensemble.calibrator, TemperatureScaler)

    def test_calibrate_with_isotonic_sets_calibrator(self, trained_ensemble):
        """calibrate(method='isotonic') should set IsotonicRegression calibrator."""
        ensemble, X_train, y_train, X_val, y_val = trained_ensemble
        ensemble.calibrate(X_train, y_train, X_val, y_val, method="isotonic")
        assert ensemble.calibrator is not None
        from sklearn.isotonic import IsotonicRegression

        assert isinstance(ensemble.calibrator, IsotonicRegression)

    def test_calibrate_with_invalid_method_fallback(self, trained_ensemble):
        """calibrate() with invalid method should fallback to isotonic."""
        ensemble, X_train, y_train, X_val, y_val = trained_ensemble
        # Invalid method still works (fallback to isotonic)
        ensemble.calibrate(X_train, y_train, X_val, y_val, method="unknown")
        assert ensemble.calibrator is not None


# =============================================================================
# IsotonicCalibrator Tests
# =============================================================================


class TestIsotonicCalibrator:
    """Test suite for IsotonicCalibrator class."""

    @pytest.fixture
    def calibrator(self):
        """Fresh IsotonicCalibrator instance."""
        return IsotonicCalibrator()

    def test_init_creates_calibrator_attribute(self, calibrator):
        """IsotonicCalibrator should initialize with calibrator attribute."""
        from sklearn.isotonic import IsotonicRegression

        assert isinstance(calibrator.calibrator, IsotonicRegression)

    def test_fit_returns_none(self, calibrator):
        """fit() returns None (it fits in-place)."""
        probs = np.array([0.3, 0.7, 0.5, 0.8])
        y_true = np.array([0, 1, 0, 1])
        result = calibrator.fit(y_true, probs)
        assert result is None

    def test_calibrate_returns_ndarray(self, calibrator):
        """calibrate() should return numpy array."""
        probs = np.array([0.3, 0.7, 0.5, 0.8])
        y_true = np.array([0, 1, 0, 1])
        calibrator.fit(y_true, probs)
        result = calibrator.calibrate(probs)
        assert isinstance(result, np.ndarray)

    def test_calibrate_output_in_unit_interval(self, calibrator):
        """calibrated probabilities should be in [0, 1]."""
        probs = np.linspace(0.01, 0.99, 100)
        y_true = (probs > 0.5).astype(int)
        calibrator.fit(y_true, probs)
        result = calibrator.calibrate(probs)
        assert (result >= 0.0).all()
        assert (result <= 1.0).all()

    def test_calibrate_monotonic(self, calibrator):
        """Isotonic calibration should preserve monotonicity."""
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        y_true = np.array([0, 0, 1, 1, 1])
        calibrator.fit(y_true, probs)
        result = calibrator.calibrate(probs)
        # Check monotonic: larger input -> larger or equal output
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1]

    def test_fit_cross_validated_with_dataframe(self, calibrator):
        """fit_cross_validated() should work with DataFrame input."""
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier

        X_arr, y = make_classification(n_samples=200, n_features=10, random_state=42)
        X = pd.DataFrame(X_arr)  # Must be DataFrame
        model = RandomForestClassifier(random_state=42, n_estimators=10)
        model.fit(X, y)
        result = calibrator.fit_cross_validated(X, y, model, cv=3)
        assert result is calibrator


# =============================================================================
# Integration Tests
# =============================================================================


class TestEnsembleIntegration:
    """Integration tests for complete ensemble workflow."""

    @pytest.fixture
    def full_data(self):
        """Create larger dataset for integration tests."""
        from sklearn.datasets import make_classification

        X_train, y_train = make_classification(n_samples=500, n_features=20, random_state=42)
        X_val, y_val = make_classification(n_samples=150, n_features=20, random_state=43)
        X_test, y_test = make_classification(n_samples=100, n_features=20, random_state=44)
        return (
            pd.DataFrame(X_train),
            y_train,
            pd.DataFrame(X_val),
            y_val,
            pd.DataFrame(X_test),
            y_test,
        )

    def test_full_training_pipeline(self, full_data):
        """Test complete training: fit -> calibrate -> predict."""
        X_train, y_train, X_val, y_val, X_test, y_test = full_data
        ensemble = StackingEnsemble(random_seed=42)
        ensemble.fit(X_train, y_train, X_val, y_val)
        ensemble.calibrate(X_train, y_train, X_val, y_val, method="temperature")
        predictions = ensemble.predict_proba_calibrated(X_test)
        assert len(predictions) == len(y_test)
        assert (predictions >= 0).all() and (predictions <= 1).all()

    def test_prediction_shape_matches_input(self, full_data):
        """Prediction output shape should match input rows."""
        X_train, y_train, X_val, y_val, X_test, _ = full_data
        ensemble = StackingEnsemble(random_seed=42)
        ensemble.fit(X_train, y_train, X_val, y_val)
        predictions = ensemble.predict_proba(X_test)
        assert predictions.shape[0] == X_test.shape[0]

    def test_multiple_calibration_methods_work(self, full_data):
        """Both temperature and isotonic calibration should work."""
        X_train, y_train, X_val, y_val, X_test, _ = full_data

        for method in ["temperature", "isotonic"]:
            ensemble = StackingEnsemble(random_seed=42)
            ensemble.fit(X_train, y_train, X_val, y_val)
            ensemble.calibrate(X_train, y_train, X_val, y_val, method=method)
            predictions = ensemble.predict_proba_calibrated(X_test)
            assert len(predictions) == X_test.shape[0]
            assert predictions.min() >= 0.0
            assert predictions.max() <= 1.0
