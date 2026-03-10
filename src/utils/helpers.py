"""
Utility helper functions for score parsing and match analysis.
"""

import re
from typing import Union, Tuple, List

import numpy as np
import pandas as pd


def is_retirement(score: Union[str, float]) -> bool:
    """
    Detect whether a match ended in retirement or walkover.

    Args:
        score: Score string from the dataset.

    Returns:
        True if the match was a retirement or walkover, False otherwise.
    """
    if pd.isna(score):
        return False
    if isinstance(score, (int, float)):
        return False
    return bool(re.search(r"Ret\.|W\.O\.|Walkover|Retired", score, re.IGNORECASE))


def extract_total_points(score: str) -> int | None:
    """
    Extract the total number of points from a score string.

    Args:
        score: Score string, e.g. "21-11 / 21-18" or "21-6 / 20-22 / 21-13".

    Returns:
        Total points scored across all sets, or None if unparseable.
    """
    if pd.isna(score):
        return None

    # Remove retirement marker
    score = re.sub(r"\s+Ret\.?\s*$", "", str(score))

    # Extract all numbers
    numbers = re.findall(r"\d+", score)
    if not numbers:
        return None

    return sum(int(n) for n in numbers)


def count_sets(score: str) -> int | None:
    """
    Count the number of sets played.

    Args:
        score: Score string.

    Returns:
        Number of sets (2 or 3), or None if unparseable.
    """
    if pd.isna(score):
        return None

    score_str = str(score)
    # Count "/" separators to determine set count
    set_count = score_str.count("/") + 1

    # For retirements the set count may be incomplete; cap at 3
    return min(set_count, 3)


def extract_set_scores(score: str) -> List[Tuple[int, int]]:
    """
    Extract per-set scores from a score string.

    Args:
        score: Score string.

    Returns:
        List of (winner_points, loser_points) tuples for each set.
    """
    if pd.isna(score):
        return []

    score_str = str(score)
    # Remove retirement marker
    score_str = re.sub(r"\s+Ret\.?\s*$", "", score_str)

    # Split into individual sets
    sets = score_str.split("/")
    result = []

    for s in sets:
        numbers = re.findall(r"\d+", s)
        if len(numbers) >= 2:
            result.append((int(numbers[0]), int(numbers[1])))

    return result


def calculate_point_diff_set1(score: str) -> int | None:
    """
    Calculate the point difference in the first set.

    Args:
        score: Score string.

    Returns:
        First-set point difference (winner - loser), or None if unparseable.
    """
    sets = extract_set_scores(score)
    if not sets:
        return None
    winner_score, loser_score = sets[0]
    return winner_score - loser_score


def parse_score(score: str) -> dict:
    """
    Fully parse a score string into a dictionary of match metrics.

    Args:
        score: Score string.

    Returns:
        Dictionary containing total_points, sets_played, point_diff_set1,
        is_retirement, and is_decider.
    """
    return {
        "total_points": extract_total_points(score),
        "sets_played": count_sets(score),
        "point_diff_set1": calculate_point_diff_set1(score),
        "is_retirement": is_retirement(score),
        "is_decider": count_sets(score) == 3 if not is_retirement(score) else False,
    }
