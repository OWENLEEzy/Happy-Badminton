"""Flask API for badminton match prediction (general predictor only)."""

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.constants import LEVEL_MAP, CONTINENT_MAP

_frontend_dir = Path(__file__).parent
app = Flask(
    __name__,
    static_folder=str(_frontend_dir / "static"),
    template_folder=str(_frontend_dir / "templates"),
)
# Set CORS_ORIGINS env var in production (e.g. "https://example.com")
# Defaults to "*" for local development only
CORS(app, origins=os.environ.get("CORS_ORIGINS", "*").split(","))

# Simplified (general predictor) model cache
_simplified_model: Any = None
_simplified_features: list[str] | None = None
_simplified_feature_importance: dict[str, float] | None = None
_simplified_neutral_values: dict[str, float] | None = None

# Set count model cache
_set_count_model: Any = None
_set_count_features: list[str] | None = None

# Nationality pair win rates lookup cache
_nat_pair_lookup: dict[str, float] | None = None

# Maps BWF 3-letter association codes to full country names used in training data.
# Allows continent and nationality-pair features to work from ISO code input.
_BWF_TO_ASSOC: dict[str, str] = {
    "AUS": "Australia",
    "AUT": "Austria",
    "AZE": "Azerbaijan",
    "BAN": "Bangladesh",
    "BEL": "Belgium",
    "BRA": "Brazil",
    "BUL": "Bulgaria",
    "CAN": "Canada",
    "CHN": "China",
    "CMR": "Cameroon",
    "CRO": "Croatia",
    "CZE": "Czechia",
    "DEN": "Denmark",
    "EGY": "Egypt",
    "ENG": "England",
    "ESP": "Spain",
    "EST": "Estonia",
    "FIN": "Finland",
    "FRA": "France",
    "GER": "Germany",
    "GHA": "Ghana",
    "GRE": "Greece",
    "HKG": "Hong Kong",
    "HUN": "Hungary",
    "INA": "Indonesia",
    "IND": "India",
    "IRL": "Ireland",
    "IRI": "Iran",
    "ISR": "Israel",
    "ITA": "Italy",
    "JPN": "Japan",
    "KAZ": "Kazakhstan",
    "KEN": "Kenya",
    "KOR": "Korea",
    "KGZ": "Kyrgyzstan",
    "LAO": "Lao",
    "LAT": "Latvia",
    "LTU": "Lithuania",
    "MAC": "Macau",
    "MAD": "Madagascar",
    "MAS": "Malaysia",
    "MDA": "Moldova",
    "MEX": "Mexico",
    "MGL": "Mongolia",
    "MRI": "Mauritius",
    "MYA": "Myanmar",
    "NED": "Netherlands",
    "NEP": "Nepal",
    "NGA": "Nigeria",
    "NOR": "Norway",
    "NZL": "New Zealand",
    "PAK": "Pakistan",
    "PER": "Peru",
    "PHI": "Philippines",
    "POL": "Poland",
    "POR": "Portugal",
    "ROU": "Romania",
    "RSA": "South Africa",
    "RUS": "Russia",
    "RWA": "Rwanda",
    "SCO": "Scotland",
    "SGP": "Singapore",
    "SLO": "Slovenia",
    "SRB": "Serbia",
    "SRI": "Sri Lanka",
    "SUI": "Switzerland",
    "SVK": "Slovakia",
    "SWE": "Sweden",
    "THA": "Thailand",
    "TPE": "Chinese Taipei",
    "TTO": "Trinidad and Tobago",
    "TUN": "Tunisia",
    "TUR": "Turkey",
    "UGA": "Uganda",
    "UKR": "Ukraine",
    "USA": "United States",
    "UZB": "Uzbekistan",
    "VIE": "Vietnam",
    "WAL": "Wales",
    "ZAM": "Zambia",
    "ZIM": "Zimbabwe",
}


def get_simplified_model() -> tuple[Any, list[str], dict[str, float], dict[str, float]]:
    """Lazy-load the simplified ensemble model."""
    global \
        _simplified_model, \
        _simplified_features, \
        _simplified_feature_importance, \
        _simplified_neutral_values
    if _simplified_model is None:
        import loguru

        logger = loguru.logger

        model_path = project_root / "models" / "simplified_ensemble.pkl"
        results_path = project_root / "models" / "simplified_results.json"
        fi_path = project_root / "models" / "simplified_feature_importance.json"
        logger.info(f"Loading simplified model from {model_path}...")
        _simplified_model = joblib.load(str(model_path))

        with open(results_path) as f:
            results = json.load(f)
        _simplified_features = results["features"]
        _simplified_neutral_values = results.get("neutral_values", {})

        with open(fi_path) as f:
            _simplified_feature_importance = json.load(f)

        logger.info("Simplified model loaded.")

    # Bind to locals so Pyright can narrow Optional types
    features = _simplified_features
    feat_importance = _simplified_feature_importance
    neutral_vals = _simplified_neutral_values
    if features is None:
        raise RuntimeError("Simplified model feature list failed to load from results JSON")
    if feat_importance is None:
        raise RuntimeError("Simplified model feature importance failed to load from JSON")
    if neutral_vals is None:
        raise RuntimeError("Simplified model neutral values failed to load from results JSON")
    return _simplified_model, features, feat_importance, neutral_vals


def get_set_count_model() -> tuple[Any, list[str]]:
    """Lazy-load the set count prediction model."""
    global _set_count_model, _set_count_features
    if _set_count_model is None:
        import loguru

        logger = loguru.logger

        model_path = project_root / "models" / "set_count_model.pkl"
        results_path = project_root / "models" / "set_count_results.json"
        logger.info(f"Loading set count model from {model_path}...")
        _set_count_model = joblib.load(str(model_path))

        with open(results_path) as f:
            _set_count_features = json.load(f)["features"]

        logger.info("Set count model loaded.")

    # Bind to local so Pyright can narrow Optional type
    sc_features = _set_count_features
    if sc_features is None:
        raise RuntimeError("Set count model feature list failed to load from results JSON")
    return _set_count_model, sc_features


def _bwf_to_full_name(code: str) -> str:
    """Convert BWF 3-letter code to full country name; falls back to input if not found."""
    return _BWF_TO_ASSOC.get(code.upper(), code)


def _code_to_continent(code: str) -> str:
    """Convert BWF code or full country name to continent string."""
    full_name = _bwf_to_full_name(code)
    return CONTINENT_MAP.get(full_name, CONTINENT_MAP.get(code, "Unknown"))


def get_nat_pair_lookup() -> dict[str, float]:
    """Lazy-load nationality pair win rates lookup."""
    global _nat_pair_lookup
    if _nat_pair_lookup is None:
        path = project_root / "models" / "nat_pair_win_rates.json"
        if path.exists():
            with open(path) as f:
                _nat_pair_lookup = json.load(f)
        else:
            _nat_pair_lookup = {}
    # Bind to local so Pyright can narrow Optional type
    lookup = _nat_pair_lookup
    if lookup is None:
        raise RuntimeError("Nationality pair win rates lookup failed to initialise")
    return lookup


def build_general_features(
    match_type: str,
    tournament_level: str,
    round_stage: int,
    match_month: int,
    host_country: str,
    p1: dict[str, Any],
    p2: dict[str, Any],
    h2h: dict[str, Any],
) -> dict[str, float]:
    """Convert user-supplied match parameters into the simplified feature vector.

    All values are pre-match only (no post-match data).
    Note: avg_opp_rank (opponent quality) is not included here because
    form_quality_diff was excluded from SIMPLIFIED_FEATURES during training.
    """
    p1_rank = max(1, p1.get("ranking", 100))
    p2_rank = max(1, p2.get("ranking", 100))
    p1_elo = float(p1.get("elo", 1616))
    p2_elo = float(p2.get("elo", 1616))

    log_rank_diff = math.log(p1_rank + 1) - math.log(p2_rank + 1)
    rank_closeness = 1.0 / (1.0 + abs(log_rank_diff))

    category_flag = 1 if match_type == "WS" else 0
    winner_home = 1 if p1.get("nationality", "").upper() == host_country.upper() else 0
    loser_home = 1 if p2.get("nationality", "").upper() == host_country.upper() else 0

    level_numeric = LEVEL_MAP.get(tournament_level.upper(), 5)
    level_x_home = level_numeric * winner_home
    home_x_closeness = winner_home * rank_closeness

    def _form(d: dict[str, Any], key: str, default: float, window: float) -> float:
        """Extract a form win count, clamp to [0, window], normalise to [0, 1]."""
        raw = d.get(key)
        val: float = float(raw) if raw is not None else default
        return min(max(val, 0.0), window) / window

    # Clamp win counts to [0, window] so they cannot exceed the window size.
    winner_form_5 = _form(p1, "form5_wins", 2.5, 5.0)
    loser_form_5 = _form(p2, "form5_wins", 2.5, 5.0)
    form_diff_5 = winner_form_5 - loser_form_5

    winner_form_10 = _form(p1, "form10_wins", 5.0, 10.0)
    loser_form_10 = _form(p2, "form10_wins", 5.0, 10.0)
    form_diff_10 = winner_form_10 - loser_form_10

    winner_form_20 = _form(p1, "form20_wins", 10.0, 20.0)
    loser_form_20 = _form(p2, "form20_wins", 10.0, 20.0)
    form_diff_20 = winner_form_20 - loser_form_20

    form_momentum_w = winner_form_5 - winner_form_10
    form_momentum_l = loser_form_5 - loser_form_10
    momentum_diff = form_momentum_w - form_momentum_l

    raw_streak_w = p1.get("streak", 0)
    raw_streak_l = p2.get("streak", 0)
    streak_capped_w = float(np.sign(raw_streak_w) * min(abs(raw_streak_w), 5))
    streak_capped_l = float(np.sign(raw_streak_l) * min(abs(raw_streak_l), 5))
    streak_capped_diff = streak_capped_w - streak_capped_l

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

    career = max(0, p1.get("career_matches", 100))
    career_stage = _career_stage(career)
    career_l = max(0, p2.get("career_matches", 100))
    career_stage_l = _career_stage(career_l)

    # Bayesian H2H smoothing — wins cannot exceed total matches.
    PRIOR = 5
    h2h_total = max(0, h2h.get("total", 0))
    h2h_wins = min(max(h2h.get("p1_wins", 0), 0), h2h_total)
    h2h_win_rate_bayes = (h2h_wins + PRIOR * 0.5) / (h2h_total + PRIOR)

    rank_x_form_diff = log_rank_diff * form_diff_10
    rank_closeness_x_h2h = rank_closeness * (h2h_win_rate_bayes - 0.5)
    gender_x_rank = category_flag * log_rank_diff

    # Nationality & continent features
    p1_nat = (p1.get("nationality") or "").strip().upper()
    p2_nat = (p2.get("nationality") or "").strip().upper()
    same_nationality = 1 if p1_nat and p1_nat == p2_nat else 0

    # Nationality matchup win diff (Bayesian prior = 0.5 when pair unknown)
    nat_matchup_win_diff = 0.0
    if p1_nat and p2_nat and p1_nat != p2_nat:
        p1_full = _bwf_to_full_name(p1_nat)
        p2_full = _bwf_to_full_name(p2_nat)
        pair_key = f"{min(p1_full, p2_full)}|{max(p1_full, p2_full)}"
        lookup = get_nat_pair_lookup()
        if pair_key in lookup:
            first_win_rate = lookup[pair_key]
            p1_win_rate = first_win_rate if p1_full <= p2_full else 1.0 - first_win_rate
            nat_matchup_win_diff = p1_win_rate - 0.5

    # Continent home advantage
    host_cont = _code_to_continent(host_country) if host_country else "Unknown"
    p1_cont = _code_to_continent(p1_nat) if p1_nat else "Unknown"
    p2_cont = _code_to_continent(p2_nat) if p2_nat else "Unknown"
    winner_continent_home = int(p1_cont != "Unknown" and p1_cont == host_cont)
    loser_continent_home = int(p2_cont != "Unknown" and p2_cont == host_cont)
    continent_advantage_diff = winner_continent_home - loser_continent_home

    # Historical 3-set rate: fraction of prior matches that went to 3 sets.
    # Default = 0.40 (population prior). User supplies as a decimal in [0, 1].
    _PRIOR_3SET = 0.40
    p1_3set_rate = min(1.0, max(0.0, float(p1.get("3set_rate", _PRIOR_3SET))))
    p2_3set_rate = min(1.0, max(0.0, float(p2.get("3set_rate", _PRIOR_3SET))))
    threeset_rate_diff = p1_3set_rate - p2_3set_rate

    return {
        "log_rank_diff": log_rank_diff,
        "rank_closeness": rank_closeness,
        "category_flag": category_flag,
        "level_numeric": level_numeric,
        "round_stage": round_stage,
        "match_month": match_month,
        "winner_home": winner_home,
        "loser_home": loser_home,
        "level_x_home": level_x_home,
        "home_x_closeness": home_x_closeness,
        "winner_form_5": winner_form_5,
        "loser_form_5": loser_form_5,
        "form_diff_5": form_diff_5,
        "form_diff_10": form_diff_10,
        "form_diff_20": form_diff_20,
        "form_momentum_w": form_momentum_w,
        "form_momentum_l": form_momentum_l,
        "momentum_diff": momentum_diff,
        "streak_capped_w": streak_capped_w,
        "streak_capped_l": streak_capped_l,
        "streak_capped_diff": streak_capped_diff,
        "h2h_win_rate_bayes": h2h_win_rate_bayes,
        "career_stage": career_stage,
        "career_stage_l": career_stage_l,
        "rank_x_form_diff": rank_x_form_diff,
        "rank_closeness_x_h2h": rank_closeness_x_h2h,
        "gender_x_rank": gender_x_rank,
        # Nationality & continent
        "same_nationality": same_nationality,
        "nat_matchup_win_diff": nat_matchup_win_diff,
        "winner_continent_home": winner_continent_home,
        "loser_continent_home": loser_continent_home,
        "continent_advantage_diff": continent_advantage_diff,
        # ELO (pre-match, from BWF website — defaults to median if not provided)
        "winner_elo": p1_elo,
        "loser_elo": p2_elo,
        "elo_diff": p1_elo - p2_elo,
        # Historical 3-set rate (for set count model; win model ignores these)
        "winner_3set_rate": p1_3set_rate,
        "loser_3set_rate": p2_3set_rate,
        "threeset_rate_diff": threeset_rate_diff,
    }


def bootstrap_confidence_interval(
    model,
    features_df: pd.DataFrame,
    n_bootstrap: int = 100,
    alpha: float = 0.1,
) -> tuple[float, float]:
    """Return (p_low, p_high) bootstrap confidence interval for prediction.

    Samples base-model predictions with replacement to estimate variance.
    """
    try:
        import catboost as _cb
    except ImportError:
        _cb = None  # type: ignore[assignment]

    def _base_pred(name: str, bm: Any) -> float:
        """Return scalar win-probability from a single base model."""
        if name == "catboost" and _cb is not None:
            pool = _cb.Pool(features_df)
            return float(bm.predict_proba(pool)[:, 1][0])
        if hasattr(bm, "predict_proba"):
            return float(bm.predict_proba(features_df)[:, 1][0])
        return float(bm.predict(features_df)[0])

    base_preds = [_base_pred(name, bm) for name, bm in model.base_models.items()]

    rng = np.random.default_rng()
    samples = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(base_preds, size=len(base_preds), replace=True)
        meta_input = np.array([sampled])
        raw = model.meta_model.predict(meta_input)[0]
        raw = float(np.clip(raw, 0.0, 1.0))
        if model.calibrator is not None:
            from src.models.ensemble_models import TemperatureScaler

            if isinstance(model.calibrator, TemperatureScaler):
                raw = float(model.calibrator.transform(np.array([raw]))[0])
            else:
                raw = float(model.calibrator.predict(np.array([raw]))[0])
        samples.append(raw)

    ci_low = float(np.percentile(samples, alpha * 100))
    ci_high = float(np.percentile(samples, (1 - alpha) * 100))
    return ci_low, ci_high


def compute_driving_factors(
    features_dict: dict[str, float],
    feature_importance: dict[str, float],
    neutral_values: dict[str, float],
    p1_name: str,
    p2_name: str,
) -> list[dict[str, str | float]]:
    """Compute top driving factors as signed contribution scores.

    Positive contribution favours player 1.
    Results normalised so the strongest factor = 100%.
    """
    factor_labels = {
        "log_rank_diff": "Ranking gap",
        "rank_closeness": "Rank closeness",
        "form_diff_5": "Recent form (5)",
        "form_diff_10": "Recent form (10)",
        "form_diff_20": "Recent form (20)",
        "momentum_diff": "Form momentum",
        "streak_capped_diff": "Current streak",
        "h2h_win_rate_bayes": "H2H record",
        "winner_home": "Home ground",
        "level_x_home": "Home × level",
        "home_x_closeness": "Home advantage",
        "level_numeric": "Tournament level",
        "winner_form_5": f"{p1_name} form (5)",
        "loser_form_5": f"{p2_name} form (5)",
        "career_stage": "Career stage",
        "category_flag": "Match category",
        "streak_capped_w": f"{p1_name} streak",
        "streak_capped_l": f"{p2_name} streak",
        "rank_x_form_diff": "Rank × form",
        "rank_closeness_x_h2h": "Closeness × H2H",
        "gender_x_rank": "Gender × rank",
        "elo_diff": "ELO difference",
        "winner_elo": f"{p1_name} ELO",
        "loser_elo": f"{p2_name} ELO",
    }

    display_exclude = {
        "category_flag",
        "level_x_home",
        "gender_x_rank",
        "home_x_closeness",
        "rank_closeness",
        "match_month",
        "round_stage",
        "loser_home",
    }

    feature_scales = {
        "streak_capped_w": 5.0,
        "streak_capped_l": 5.0,
        "level_numeric": 10.0,
    }

    p2_favoured_when_positive = {"log_rank_diff", "loser_form_5", "streak_capped_l", "loser_elo"}

    contributions = []
    for feat, value in features_dict.items():
        if feat in display_exclude:
            continue
        imp = feature_importance.get(feat, 0.0)
        neutral = neutral_values.get(feat, 0.0)
        raw_delta = (value - neutral) * imp
        scale = feature_scales.get(feat, 1.0)
        delta = raw_delta / scale * 100
        if abs(delta) < 0.001:
            continue
        label = factor_labels.get(feat, feat)
        if feat in p2_favoured_when_positive:
            direction = "p2" if delta > 0 else "p1"
        else:
            direction = "p1" if delta > 0 else "p2"
        contributions.append(
            {
                "feature": feat,
                "label": label,
                "delta": round(abs(delta), 3),
                "direction": direction,
            }
        )

    contributions.sort(key=lambda x: abs(x["delta"]), reverse=True)
    top = contributions[:5]

    max_abs = max((c["delta"] for c in top), default=1.0) or 1.0
    for c in top:
        pct = c["delta"] / max_abs * 100
        sign = "+" if c["direction"] == "p1" else "-"
        c["delta_str"] = f"{sign}{pct:.0f}%"

    return top


@app.route("/")
def index():
    """Serve the frontend page."""
    return render_template("index.html")


@app.route("/api/predict-general", methods=["POST"])
def predict_general():
    """General predictor using only pre-match, user-supplied features.

    Request body:
    {
        "match_type": "MS",
        "tournament_level": "S1000",
        "round_stage": 6,
        "match_month": 3,
        "host_country": "CHN",
        "player1": {
            "name": "optional",
            "ranking": 5,
            "nationality": "CHN",
            "form5_wins": 3,
            "form10_wins": 7,
            "form20_wins": 14,
            "streak": 3,
            "career_matches": 200
        },
        "player2": { ... },
        "h2h": { "p1_wins": 3, "total": 7 }
    }
    """
    try:
        if not request.is_json:
            return jsonify({"error": "Request body must be JSON"}), 400
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Request body required"}), 400

        match_type = data.get("match_type", "MS").upper()
        tournament_level = data.get("tournament_level", "S300")
        round_stage = max(0, min(8, int(data.get("round_stage", 4))))
        match_month = max(1, min(12, int(data.get("match_month", 6))))
        host_country = data.get("host_country", "")
        p1 = data.get("player1", {})
        p2 = data.get("player2", {})
        h2h = data.get("h2h", {})

        if match_type not in ["MS", "WS", "MD", "WD", "XD"]:
            return jsonify({"error": "Invalid match_type"}), 400
        if not p1.get("ranking") or not p2.get("ranking"):
            return jsonify({"error": "player1.ranking and player2.ranking are required"}), 400

        model, feature_cols, feature_importance, neutral_values = get_simplified_model()

        features_dict = build_general_features(
            match_type, tournament_level, round_stage, match_month, host_country, p1, p2, h2h
        )
        features_df = pd.DataFrame(
            [[features_dict.get(f, 0.0) for f in feature_cols]],
            columns=feature_cols,  # type: ignore
        )

        if hasattr(model, "predict_proba_calibrated"):
            prob = float(model.predict_proba_calibrated(features_df)[0])
        else:
            raw = model.predict_proba(features_df)
            prob = float(raw[0][1] if raw.ndim > 1 else raw[0])

        ci_low, ci_high = bootstrap_confidence_interval(model, features_df)

        p1_name = p1.get("name") or "Player 1"
        p2_name = p2.get("name") or "Player 2"
        predicted_winner = p1_name if prob > 0.5 else p2_name
        confidence = max(prob, 1 - prob)

        driving_factors = compute_driving_factors(
            features_dict, feature_importance, neutral_values, p1_name, p2_name
        )

        # Set count prediction (2-0 vs 2-1)
        set_count_scenarios = None
        try:
            sc_model, sc_features = get_set_count_model()
            sc_df = pd.DataFrame(
                [[features_dict.get(f, 0.0) for f in sc_features]],
                columns=sc_features,  # type: ignore
            )
            set3_prob = float(sc_model.predict_proba_calibrated(sc_df)[0])
            set2_prob = 1.0 - set3_prob
            set_count_scenarios = {
                "p1_2_0": round(prob * set2_prob, 4),
                "p1_2_1": round(prob * set3_prob, 4),
                "p2_0_2": round((1 - prob) * set2_prob, 4),
                "p2_1_2": round((1 - prob) * set3_prob, 4),
            }
        except FileNotFoundError:
            pass  # Set count model not trained yet
        except Exception:
            import loguru

            loguru.logger.warning("Set count prediction failed", exc_info=True)

        return jsonify(
            {
                "player1_name": p1_name,
                "player2_name": p2_name,
                "player1_win_prob": prob,
                "player2_win_prob": 1 - prob,
                "probability": round(prob, 4),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "predicted_winner": predicted_winner,
                "confidence": round(confidence, 4),
                "driving_factors": driving_factors,
                "set_count_scenarios": set_count_scenarios,
            }
        )

    except FileNotFoundError:
        return (
            jsonify(
                {"error": "Simplified model not found. Run: python scripts/train_simplified.py"}
            ),
            503,
        )
    except Exception:
        import loguru

        loguru.logger.exception("Prediction request failed")
        return jsonify({"error": "Prediction failed. Please check your input."}), 500


@app.route("/api/health", methods=["GET"])
def health():
    """Health check."""
    return jsonify({"status": "ok", "message": "Badminton Prediction API is running"})


if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 5001))
    print("=" * 60)
    print("Badminton Match Prediction API")
    print("=" * 60)
    print(f"Frontend: http://localhost:{PORT}")
    print(f"API:      http://localhost:{PORT}/api")
    print("=" * 60)
    debug = os.environ.get("FLASK_DEBUG", "false").lower() in ("1", "true")
    app.run(host="0.0.0.0", port=PORT, debug=debug)
