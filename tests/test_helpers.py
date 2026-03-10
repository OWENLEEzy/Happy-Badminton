"""Tests for src/utils/helpers.py score-parsing utilities."""

import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import (
    calculate_point_diff_set1,
    count_sets,
    extract_set_scores,
    extract_total_points,
    is_retirement,
    parse_score,
)


# ---------------------------------------------------------------------------
# is_retirement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "score,expected",
    [
        ("21-11 / 21-18", False),
        ("21-6 / 20-22 / 21-13", False),
        ("21-14 Ret.", True),
        ("21-14 ret.", True),  # case-insensitive
        ("W.O.", True),
        ("Walkover", True),
        ("Retired", True),
        (float("nan"), False),
        (42, False),  # numeric input is never a retirement
    ],
)
def test_is_retirement(score: object, expected: bool) -> None:
    assert is_retirement(score) is expected  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# extract_total_points
# ---------------------------------------------------------------------------


def test_total_points_two_sets() -> None:
    """21+11+21+18 = 71."""
    assert extract_total_points("21-11 / 21-18") == 71


def test_total_points_three_sets() -> None:
    """21+6+20+22+21+13 = 103."""
    assert extract_total_points("21-6 / 20-22 / 21-13") == 103


def test_total_points_retirement_stripped() -> None:
    """Retirement marker must be stripped before summing."""
    assert extract_total_points("21-14 Ret.") == 35


def test_total_points_nan_returns_none() -> None:
    result = extract_total_points(float("nan"))  # type: ignore[arg-type]
    assert result is None


def test_total_points_empty_string_returns_none() -> None:
    assert extract_total_points("") is None


# ---------------------------------------------------------------------------
# count_sets
# ---------------------------------------------------------------------------


def test_count_sets_two() -> None:
    assert count_sets("21-11 / 21-18") == 2


def test_count_sets_three() -> None:
    assert count_sets("21-6 / 20-22 / 21-13") == 3


def test_count_sets_capped_at_three() -> None:
    """Badly formatted strings with many '/' should not exceed 3."""
    assert count_sets("21-11 / 21-18 / 21-14 / 21-10") == 3


def test_count_sets_nan_returns_none() -> None:
    result = count_sets(float("nan"))  # type: ignore[arg-type]
    assert result is None


# ---------------------------------------------------------------------------
# extract_set_scores
# ---------------------------------------------------------------------------


def test_extract_set_scores_two_sets() -> None:
    result = extract_set_scores("21-11 / 21-18")
    assert result == [(21, 11), (21, 18)]


def test_extract_set_scores_three_sets() -> None:
    result = extract_set_scores("21-6 / 20-22 / 21-13")
    assert result == [(21, 6), (20, 22), (21, 13)]


def test_extract_set_scores_nan_empty() -> None:
    assert extract_set_scores(float("nan")) == []  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# calculate_point_diff_set1
# ---------------------------------------------------------------------------


def test_point_diff_set1_positive() -> None:
    """21-11 in first set -> diff = 10."""
    assert calculate_point_diff_set1("21-11 / 21-18") == 10


def test_point_diff_set1_close() -> None:
    """22-20 (tiebreak) -> diff = 2."""
    assert calculate_point_diff_set1("22-20 / 21-18") == 2


def test_point_diff_set1_nan_returns_none() -> None:
    assert calculate_point_diff_set1(float("nan")) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# parse_score (integration)
# ---------------------------------------------------------------------------


def test_parse_score_normal_two_sets() -> None:
    result = parse_score("21-11 / 21-18")
    assert result["total_points"] == 71
    assert result["sets_played"] == 2
    assert result["point_diff_set1"] == 10
    assert result["is_retirement"] is False
    assert result["is_decider"] is False


def test_parse_score_three_sets_decider() -> None:
    result = parse_score("21-6 / 20-22 / 21-13")
    assert result["sets_played"] == 3
    assert result["is_decider"] is True


def test_parse_score_retirement_not_decider() -> None:
    """Retirement matches should never count as a decider."""
    result = parse_score("21-14 Ret.")
    assert result["is_retirement"] is True
    assert result["is_decider"] is False
