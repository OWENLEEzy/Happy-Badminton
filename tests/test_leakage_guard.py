"""ML safety tests: no leakage features in training, no future info in rolling."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest

from scripts.train_simplified import SIMPLIFIED_FEATURES, NEUTRAL_VALUES
from src.utils.constants import LEAK_FEATURES


def test_no_leak_features_in_simplified() -> None:
    """LEAK_FEATURES must not appear in SIMPLIFIED_FEATURES."""
    overlap = set(LEAK_FEATURES) & set(SIMPLIFIED_FEATURES)
    assert overlap == set(), f"Data leakage: {overlap} found in SIMPLIFIED_FEATURES"


def test_simplified_features_no_duplicates() -> None:
    """Feature list must have no duplicates (order matters for inference alignment)."""
    assert len(SIMPLIFIED_FEATURES) == len(set(SIMPLIFIED_FEATURES)), (
        "Duplicate feature names found — breaks feature-order alignment at inference time"
    )


def test_neutral_values_cover_all_features() -> None:
    """Every feature in SIMPLIFIED_FEATURES must have a NEUTRAL_VALUES entry."""
    missing = [f for f in SIMPLIFIED_FEATURES if f not in NEUTRAL_VALUES]
    assert missing == [], f"Missing NEUTRAL_VALUES for: {missing}"


def test_neutral_values_no_extra_keys() -> None:
    """NEUTRAL_VALUES must not contain keys absent from SIMPLIFIED_FEATURES."""
    extra = [k for k in NEUTRAL_VALUES if k not in SIMPLIFIED_FEATURES]
    assert extra == [], f"Extra keys in NEUTRAL_VALUES not in SIMPLIFIED_FEATURES: {extra}"


def test_h2h_bayes_neutral_is_half() -> None:
    """Neutral h2h_win_rate_bayes must be 0.5 (equal prior)."""
    assert NEUTRAL_VALUES["h2h_win_rate_bayes"] == pytest.approx(0.5)


def test_elo_diff_neutral_is_zero() -> None:
    """Neutral elo_diff must be 0.0 (equal ELO — no advantage)."""
    assert NEUTRAL_VALUES["elo_diff"] == pytest.approx(0.0)


def test_winner_loser_elo_neutrals_equal() -> None:
    """Neutral winner_elo and loser_elo must be equal (symmetric baseline)."""
    assert NEUTRAL_VALUES["winner_elo"] == NEUTRAL_VALUES["loser_elo"]
