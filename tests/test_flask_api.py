"""Tests for the Flask API (frontend/app.py).

Uses pytest-flask / Flask test client with mocked model to avoid loading
the full ML pipeline.  Only the HTTP contract is tested here.
"""

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import app BEFORE patching so the module-level globals are visible
import frontend.app as app_module
from frontend.app import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    """Flask test client.

    TESTING is intentionally NOT set so that HTTP exceptions (400, 415 …) are
    returned as proper HTTP responses rather than propagated as test failures.
    """
    with app.test_client() as c:
        yield c


def _mock_model() -> MagicMock:
    """Return a minimal mock that satisfies the model interface."""
    model = MagicMock()
    model.predict_proba_calibrated.return_value = np.array([0.65])

    lgbm_mock = MagicMock()
    lgbm_mock.predict_proba.return_value = np.array([[0.35, 0.65]])
    xgb_mock = MagicMock()
    xgb_mock.predict_proba.return_value = np.array([[0.35, 0.65]])
    model.base_models = {"lightgbm": lgbm_mock, "xgboost": xgb_mock}

    model.meta_model = MagicMock()
    model.meta_model.predict.return_value = np.array([0.65])
    model.calibrator = None
    return model


_FEATURE_COLS = [
    "log_rank_diff",
    "rank_closeness",
    "category_flag",
    "level_numeric",
    "round_stage",
    "match_month",
    "winner_home",
    "loser_home",
    "level_x_home",
    "home_x_closeness",
    "winner_form_5",
    "loser_form_5",
    "form_diff_5",
    "form_diff_10",
    "form_diff_20",
    "form_momentum_w",
    "form_momentum_l",
    "momentum_diff",
    "streak_capped_w",
    "streak_capped_l",
    "streak_capped_diff",
    "h2h_win_rate_bayes",
    "career_stage",
    "career_stage_l",
    "rank_x_form_diff",
    "rank_closeness_x_h2h",
    "gender_x_rank",
    "same_nationality",
    "nat_matchup_win_diff",
    "winner_continent_home",
    "loser_continent_home",
    "continent_advantage_diff",
    "winner_elo",
    "loser_elo",
    "elo_diff",
]

_FEATURE_IMPORTANCE = {col: 1.0 / len(_FEATURE_COLS) for col in _FEATURE_COLS}
_NEUTRAL_VALUES = {col: 0.0 for col in _FEATURE_COLS}
_NEUTRAL_VALUES["h2h_win_rate_bayes"] = 0.5
_NEUTRAL_VALUES["winner_elo"] = 1616.0
_NEUTRAL_VALUES["loser_elo"] = 1616.0

_VALID_PAYLOAD: dict[str, Any] = {
    "match_type": "MS",
    "tournament_level": "S1000",
    "round_stage": 6,
    "match_month": 3,
    "host_country": "CHN",
    "player1": {
        "name": "Player A",
        "ranking": 5,
        "nationality": "CHN",
        "form5_wins": 3,
        "form10_wins": 7,
        "form20_wins": 14,
        "streak": 3,
        "career_matches": 200,
        "elo": 1750.0,
    },
    "player2": {
        "name": "Player B",
        "ranking": 12,
        "nationality": "JPN",
        "form5_wins": 2,
        "form10_wins": 5,
        "form20_wins": 10,
        "streak": -1,
        "career_matches": 150,
        "elo": 1680.0,
    },
    "h2h": {"p1_wins": 3, "total": 7},
}


@pytest.fixture()
def mock_model(monkeypatch):
    """Patch get_simplified_model and get_set_count_model to avoid loading files."""
    model = _mock_model()
    monkeypatch.setattr(
        app_module,
        "get_simplified_model",
        lambda: (model, _FEATURE_COLS, _FEATURE_IMPORTANCE, _NEUTRAL_VALUES),
    )
    # Also patch nat pair lookup
    monkeypatch.setattr(app_module, "get_nat_pair_lookup", lambda: {})
    # Patch set count model to raise so it falls back gracefully
    sc_model = _mock_model()
    sc_model.predict_proba_calibrated.return_value = np.array([0.4])
    monkeypatch.setattr(
        app_module,
        "get_set_count_model",
        lambda: (sc_model, _FEATURE_COLS),
    )
    # Stub bootstrap CI to return deterministic values
    monkeypatch.setattr(
        app_module,
        "bootstrap_confidence_interval",
        lambda *args, **kwargs: (0.55, 0.75),
    )
    return model


# ---------------------------------------------------------------------------
# GET / — index page
# ---------------------------------------------------------------------------


def test_index_returns_200(client) -> None:
    response = client.get("/")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# GET /api/health
# ---------------------------------------------------------------------------


def test_health_returns_ok(client) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# POST /api/predict-general — success path
# ---------------------------------------------------------------------------


def test_predict_general_success(client, mock_model) -> None:
    response = client.post("/api/predict-general", json=_VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.get_json()

    assert "player1_win_prob" in data
    assert 0.0 <= data["player1_win_prob"] <= 1.0
    assert "predicted_winner" in data
    assert "confidence" in data
    assert "ci_low" in data
    assert "ci_high" in data


def test_predict_general_probability_range(client, mock_model) -> None:
    """Mock returns 0.65 — both win probabilities must be in [0, 1]."""
    response = client.post("/api/predict-general", json=_VALID_PAYLOAD)
    data = response.get_json()
    assert 0.0 <= data["player1_win_prob"] <= 1.0
    assert 0.0 <= data["player2_win_prob"] <= 1.0
    assert data["player1_win_prob"] + data["player2_win_prob"] == pytest.approx(1.0)


def test_predict_general_winner_is_player_name(client, mock_model) -> None:
    """When prob > 0.5, predicted_winner must be player1's name."""
    response = client.post("/api/predict-general", json=_VALID_PAYLOAD)
    data = response.get_json()
    # mock returns 0.65 > 0.5, so player 1 wins
    assert data["predicted_winner"] == "Player A"


def test_predict_general_driving_factors_present(client, mock_model) -> None:
    """Response should include a driving_factors list."""
    response = client.post("/api/predict-general", json=_VALID_PAYLOAD)
    data = response.get_json()
    assert "driving_factors" in data
    assert isinstance(data["driving_factors"], list)


def test_predict_general_ci_ordering(client, mock_model) -> None:
    """ci_low must be <= player1_win_prob <= ci_high."""
    response = client.post("/api/predict-general", json=_VALID_PAYLOAD)
    data = response.get_json()
    assert data["ci_low"] <= data["player1_win_prob"] <= data["ci_high"]


# ---------------------------------------------------------------------------
# POST /api/predict-general — error paths
# ---------------------------------------------------------------------------


def test_predict_general_no_body_returns_400(client, mock_model) -> None:
    # Sending an empty JSON body causes get_json() to return None -> 400
    response = client.post("/api/predict-general", data=b"", content_type="application/json")
    assert response.status_code == 400


def test_predict_general_missing_ranking_returns_400(client, mock_model) -> None:
    payload = {**_VALID_PAYLOAD}
    payload["player1"] = {k: v for k, v in _VALID_PAYLOAD["player1"].items() if k != "ranking"}
    response = client.post("/api/predict-general", json=payload)
    assert response.status_code == 400


def test_predict_general_invalid_match_type_returns_400(client, mock_model) -> None:
    payload = {**_VALID_PAYLOAD, "match_type": "INVALID"}
    response = client.post("/api/predict-general", json=payload)
    assert response.status_code == 400


def test_predict_general_ws_category_flag(client, mock_model) -> None:
    """WS match type must produce category_flag=1 in the feature vector."""
    payload = {**_VALID_PAYLOAD, "match_type": "WS"}
    response = client.post("/api/predict-general", json=payload)
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# build_general_features — unit tests (no HTTP)
# ---------------------------------------------------------------------------


def test_build_general_features_keys_complete() -> None:
    """Every feature in _FEATURE_COLS must appear in the returned dict."""
    from frontend.app import build_general_features

    with patch.object(app_module, "get_nat_pair_lookup", return_value={}):
        features = build_general_features(
            match_type="MS",
            tournament_level="S1000",
            round_stage=6,
            match_month=3,
            host_country="CHN",
            p1=_VALID_PAYLOAD["player1"],
            p2=_VALID_PAYLOAD["player2"],
            h2h=_VALID_PAYLOAD["h2h"],
        )

    for col in _FEATURE_COLS:
        assert col in features, f"Missing feature: {col}"


def test_build_general_features_h2h_bayes_range() -> None:
    """Bayesian H2H must be in (0, 1)."""
    from frontend.app import build_general_features

    with patch.object(app_module, "get_nat_pair_lookup", return_value={}):
        features = build_general_features(
            match_type="MS",
            tournament_level="S1000",
            round_stage=4,
            match_month=6,
            host_country="",
            p1={"ranking": 1},
            p2={"ranking": 2},
            h2h={"p1_wins": 10, "total": 10},
        )

    assert 0.0 < features["h2h_win_rate_bayes"] < 1.0


def test_build_general_features_equal_elo_zero_diff() -> None:
    """Equal ELO inputs must produce elo_diff == 0."""
    from frontend.app import build_general_features

    with patch.object(app_module, "get_nat_pair_lookup", return_value={}):
        features = build_general_features(
            match_type="MS",
            tournament_level="IS",
            round_stage=4,
            match_month=6,
            host_country="",
            p1={"ranking": 10, "elo": 1700.0},
            p2={"ranking": 10, "elo": 1700.0},
            h2h={},
        )

    assert features["elo_diff"] == pytest.approx(0.0)
    assert features["log_rank_diff"] == pytest.approx(0.0)


def test_build_general_features_career_stage_early_levels() -> None:
    """career_stage must be 0.0 for n<=20 and 1.0 for n<=50."""
    from frontend.app import build_general_features

    with patch.object(app_module, "get_nat_pair_lookup", return_value={}):
        features = build_general_features(
            match_type="MS",
            tournament_level="S1000",
            round_stage=4,
            match_month=6,
            host_country="",
            p1={"ranking": 10, "career_matches": 10},
            p2={"ranking": 10, "career_matches": 30},
            h2h={},
        )

    assert features["career_stage"] == pytest.approx(0.0)  # n=10 <= 20
    assert features["career_stage_l"] == pytest.approx(1.0)  # n=30 <= 50


def test_build_general_features_career_stage_late_levels() -> None:
    """career_stage must be 0.5 for n<=500 and 0.0 for n>500."""
    from frontend.app import build_general_features

    with patch.object(app_module, "get_nat_pair_lookup", return_value={}):
        features = build_general_features(
            match_type="MS",
            tournament_level="S1000",
            round_stage=4,
            match_month=6,
            host_country="",
            p1={"ranking": 10, "career_matches": 300},
            p2={"ranking": 10, "career_matches": 700},
            h2h={},
        )

    assert features["career_stage"] == pytest.approx(0.5)  # n=300: 200 < n <= 500
    assert features["career_stage_l"] == pytest.approx(0.0)  # n=700: n > 500


# ---------------------------------------------------------------------------
# POST /api/predict-general — additional error paths
# ---------------------------------------------------------------------------


def test_predict_general_non_json_returns_400(client, mock_model) -> None:
    """Non-JSON content type must return 400."""
    response = client.post("/api/predict-general", data=b"not json", content_type="text/plain")
    assert response.status_code == 400


def test_predict_general_model_not_found_returns_503(client) -> None:
    """FileNotFoundError from get_simplified_model must return 503."""
    with patch.object(app_module, "get_simplified_model", side_effect=FileNotFoundError):
        response = client.post("/api/predict-general", json=_VALID_PAYLOAD)
    assert response.status_code == 503
    data = response.get_json()
    assert "error" in data


def test_predict_general_set_count_file_not_found_graceful(client) -> None:
    """When set count model file not found, endpoint still returns 200 without scenarios."""
    model = _mock_model()
    with (
        patch.object(
            app_module,
            "get_simplified_model",
            return_value=(model, _FEATURE_COLS, _FEATURE_IMPORTANCE, _NEUTRAL_VALUES),
        ),
        patch.object(app_module, "get_nat_pair_lookup", return_value={}),
        patch.object(app_module, "bootstrap_confidence_interval", return_value=(0.55, 0.75)),
        patch.object(app_module, "get_set_count_model", side_effect=FileNotFoundError),
    ):
        response = client.post("/api/predict-general", json=_VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.get_json()
    assert data["set_count_scenarios"] is None


def test_predict_general_set_count_generic_exception_graceful(client) -> None:
    """Generic exception from set count model must be swallowed; endpoint still returns 200."""
    model = _mock_model()
    with (
        patch.object(
            app_module,
            "get_simplified_model",
            return_value=(model, _FEATURE_COLS, _FEATURE_IMPORTANCE, _NEUTRAL_VALUES),
        ),
        patch.object(app_module, "get_nat_pair_lookup", return_value={}),
        patch.object(app_module, "bootstrap_confidence_interval", return_value=(0.55, 0.75)),
        patch.object(app_module, "get_set_count_model", side_effect=RuntimeError("sc broken")),
    ):
        response = client.post("/api/predict-general", json=_VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.get_json()
    assert data["set_count_scenarios"] is None


def test_predict_general_fallback_predict_proba(client) -> None:
    """When model lacks predict_proba_calibrated, falls back to predict_proba."""
    model = MagicMock(spec=["predict_proba", "base_models", "meta_model", "calibrator"])
    model.predict_proba.return_value = np.array([[0.30, 0.70]])
    lgbm = MagicMock()
    lgbm.predict_proba.return_value = np.array([[0.35, 0.65]])
    xgb = MagicMock()
    xgb.predict_proba.return_value = np.array([[0.30, 0.70]])
    model.base_models = {"lightgbm": lgbm, "xgboost": xgb}
    model.meta_model.predict.return_value = np.array([0.65])
    model.calibrator = None

    with (
        patch.object(
            app_module,
            "get_simplified_model",
            return_value=(model, _FEATURE_COLS, _FEATURE_IMPORTANCE, _NEUTRAL_VALUES),
        ),
        patch.object(app_module, "get_nat_pair_lookup", return_value={}),
        patch.object(app_module, "bootstrap_confidence_interval", return_value=(0.60, 0.80)),
        patch.object(app_module, "get_set_count_model", side_effect=FileNotFoundError),
    ):
        response = client.post("/api/predict-general", json=_VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.get_json()
    assert 0.0 <= data["player1_win_prob"] <= 1.0


# ---------------------------------------------------------------------------
# get_nat_pair_lookup — missing file returns empty dict
# ---------------------------------------------------------------------------


def test_get_nat_pair_lookup_missing_file(monkeypatch) -> None:
    """When nat pair JSON file does not exist, lookup returns empty dict."""
    monkeypatch.setattr(app_module, "_nat_pair_lookup", None)
    monkeypatch.setattr(app_module, "project_root", Path("/nonexistent_hb_test"))
    result = app_module.get_nat_pair_lookup()
    assert result == {}


# ---------------------------------------------------------------------------
# bootstrap_confidence_interval — direct unit test
# ---------------------------------------------------------------------------


def test_bootstrap_confidence_interval_valid_range() -> None:
    """Bootstrap CI must have 0 <= ci_low <= ci_high <= 1."""
    import pandas as pd

    from frontend.app import bootstrap_confidence_interval

    model = MagicMock()
    lgbm = MagicMock()
    lgbm.predict_proba.return_value = np.array([[0.35, 0.65]])
    xgb = MagicMock()
    xgb.predict_proba.return_value = np.array([[0.30, 0.70]])
    model.base_models = {"lightgbm": lgbm, "xgboost": xgb}
    model.meta_model.predict.return_value = np.array([0.65])
    model.calibrator = None

    features_df = pd.DataFrame([[0.0] * len(_FEATURE_COLS)], columns=_FEATURE_COLS)
    ci_low, ci_high = bootstrap_confidence_interval(model, features_df, n_bootstrap=20)

    assert 0.0 <= ci_low <= 1.0
    assert 0.0 <= ci_high <= 1.0
    assert ci_low <= ci_high


def test_bootstrap_confidence_interval_predict_only_base_model() -> None:
    """Base model with only predict (no predict_proba) must use predict fallback."""
    import pandas as pd

    from frontend.app import bootstrap_confidence_interval

    model = MagicMock()
    # base model without predict_proba — has predict only
    reg_mock = MagicMock(spec=["predict"])
    reg_mock.predict.return_value = np.array([0.6])
    model.base_models = {"regressor": reg_mock}
    model.meta_model.predict.return_value = np.array([0.6])
    model.calibrator = None

    features_df = pd.DataFrame([[0.0] * len(_FEATURE_COLS)], columns=_FEATURE_COLS)
    ci_low, ci_high = bootstrap_confidence_interval(model, features_df, n_bootstrap=10)

    assert 0.0 <= ci_low <= 1.0
    assert ci_low <= ci_high


# ---------------------------------------------------------------------------
# POST /api/predict-general — generic 500 error path
# ---------------------------------------------------------------------------


def test_predict_general_generic_exception_returns_500(client) -> None:
    """Unexpected exception from model inference must return 500."""
    with patch.object(app_module, "get_simplified_model", side_effect=RuntimeError("unexpected")):
        response = client.post("/api/predict-general", json=_VALID_PAYLOAD)
    assert response.status_code == 500
    data = response.get_json()
    assert "error" in data
