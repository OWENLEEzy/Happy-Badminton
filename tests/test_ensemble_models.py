"""Tests for StackingEnsemble and TemperatureScaler."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.ensemble_models import StackingEnsemble, TemperatureScaler


def test_meta_learner_is_bayesian_ridge():
    """After fit(), meta_model must be a BayesianRidge instance."""
    from sklearn.datasets import make_classification
    from sklearn.linear_model import BayesianRidge

    X, y = make_classification(n_samples=300, n_features=10, random_state=42)
    X = pd.DataFrame(X)
    ens = StackingEnsemble(random_seed=42)
    ens.fit(X[:210], y[:210], X[210:], y[210:])
    assert isinstance(ens.meta_model, BayesianRidge)


def test_temperature_scaling_compresses_extremes():
    ts = TemperatureScaler()
    ts.T = 2.0  # T > 1 compresses toward 0.5
    probs = np.array([0.99, 0.5, 0.01])
    calibrated = ts.transform(probs)
    assert calibrated[0] < 0.99
    assert calibrated[2] > 0.01
    assert abs(calibrated[1] - 0.5) < 0.01


def test_temperature_scaler_fit():
    rng = np.random.default_rng(42)
    probs = rng.uniform(0.1, 0.9, 100)
    y_true = (probs > 0.5).astype(int)
    ts = TemperatureScaler()
    ts.fit(probs, y_true)
    assert ts.T > 0
    calibrated = ts.transform(probs)
    assert calibrated.min() >= 0.0
    assert calibrated.max() <= 1.0


# ---------------------------------------------------------------------------
# TemperatureScaler — extended tests
# ---------------------------------------------------------------------------


class TestTemperatureScaler:
    def test_default_temperature_is_one(self) -> None:
        """Freshly constructed scaler must have T == 1.0."""
        ts = TemperatureScaler()
        assert ts.T == pytest.approx(1.0)

    def test_transform_identity_at_t_one(self) -> None:
        """With T=1.0, transform must return probabilities unchanged."""
        ts = TemperatureScaler()
        ts.T = 1.0
        probs = np.array([0.2, 0.5, 0.8])
        result = ts.transform(probs)
        np.testing.assert_allclose(result, probs, atol=1e-6)

    def test_transform_half_is_fixed_point(self) -> None:
        """p=0.5 must map to 0.5 regardless of T (logit(0.5)=0 → sigmoid(0)=0.5)."""
        ts = TemperatureScaler()
        for t_val in [0.5, 1.0, 2.0, 5.0]:
            ts.T = t_val
            result = ts.transform(np.array([0.5]))
            assert result[0] == pytest.approx(0.5, abs=1e-6)

    def test_transform_high_t_compresses_above_half(self) -> None:
        """T > 1 must compress p > 0.5 toward 0.5."""
        ts = TemperatureScaler()
        ts.T = 3.0
        p = np.array([0.8])
        result = ts.transform(p)
        assert result[0] < p[0]
        assert result[0] > 0.5

    def test_transform_high_t_compresses_below_half(self) -> None:
        """T > 1 must compress p < 0.5 toward 0.5 (symmetric)."""
        ts = TemperatureScaler()
        ts.T = 3.0
        p = np.array([0.2])
        result = ts.transform(p)
        assert result[0] > p[0]
        assert result[0] < 0.5

    def test_transform_output_in_unit_interval(self) -> None:
        """transform output must always be in [0, 1]."""
        ts = TemperatureScaler()
        ts.T = 2.0
        probs = np.linspace(0.01, 0.99, 50)
        result = ts.transform(probs)
        assert (result >= 0.0).all()
        assert (result <= 1.0).all()

    def test_fit_returns_self(self) -> None:
        """fit() must return self to allow method chaining."""
        ts = TemperatureScaler()
        probs = np.array([0.7, 0.3, 0.8, 0.2])
        y_true = np.array([1, 0, 1, 0])
        assert ts.fit(probs, y_true) is ts

    def test_fit_overconfident_model_yields_t_greater_than_one(self) -> None:
        """Systematically overconfident predictions must produce T > 1."""
        probs = np.array([0.95, 0.95, 0.05, 0.05] * 25)
        y_true = np.array([1, 0, 1, 0] * 25)
        ts = TemperatureScaler()
        ts.fit(probs, y_true)
        assert ts.T > 1.0


# ---------------------------------------------------------------------------
# StackingEnsemble — predict_proba / predict_proba_calibrated
# ---------------------------------------------------------------------------


def _make_ensemble(raw_pred: float) -> StackingEnsemble:
    """StackingEnsemble with two mocked base models returning raw_pred."""
    ensemble = StackingEnsemble()
    mock_lgbm = MagicMock()
    mock_lgbm.predict_proba.return_value = np.array([[1 - raw_pred, raw_pred]])
    mock_xgb = MagicMock()
    mock_xgb.predict_proba.return_value = np.array([[1 - raw_pred, raw_pred]])
    ensemble.base_models = {"lightgbm": mock_lgbm, "xgboost": mock_xgb}
    meta = MagicMock()
    meta.predict.return_value = np.array([raw_pred])
    ensemble.meta_model = meta
    ensemble.calibrator = None
    return ensemble


class TestStackingEnsemblePredictProba:
    def test_predict_proba_clips_to_unit_interval(self) -> None:
        """predict_proba must clip its output to [0, 1]."""
        ensemble = _make_ensemble(0.5)
        ensemble.meta_model.predict.return_value = np.array([1.8])
        X = pd.DataFrame({"a": [1.0]})
        result = ensemble.predict_proba(X)
        assert 0.0 <= result[0] <= 1.0

    def test_predict_proba_returns_ndarray(self) -> None:
        """predict_proba must return a numpy ndarray."""
        ensemble = _make_ensemble(0.6)
        result = ensemble.predict_proba(pd.DataFrame({"a": [1.0]}))
        assert isinstance(result, np.ndarray)

    def test_calibrated_no_calibrator_returns_raw(self) -> None:
        """With calibrator=None, predict_proba_calibrated returns predict_proba output."""
        ensemble = _make_ensemble(raw_pred=0.7)
        result = ensemble.predict_proba_calibrated(pd.DataFrame({"a": [1.0]}))
        assert result[0] == pytest.approx(0.7)

    def test_calibrated_with_temperature_scaler_compresses(self) -> None:
        """With TemperatureScaler T=2, calibrated output for 0.8 must be < 0.8."""
        ensemble = _make_ensemble(raw_pred=0.8)
        ts = TemperatureScaler()
        ts.T = 2.0
        ensemble.calibrator = ts
        result = ensemble.predict_proba_calibrated(pd.DataFrame({"a": [1.0]}))
        assert result[0] < 0.8
        assert result[0] > 0.5

    def test_calibrated_output_in_unit_interval(self) -> None:
        """Calibrated probabilities must always be in [0, 1]."""
        ensemble = _make_ensemble(raw_pred=0.65)
        ts = TemperatureScaler()
        ts.T = 1.5
        ensemble.calibrator = ts
        result = ensemble.predict_proba_calibrated(pd.DataFrame({"a": [1.0]}))
        assert 0.0 <= result[0] <= 1.0
