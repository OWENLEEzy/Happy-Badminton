"""Tests for the data loading module."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import load_all_sheets, merge_data


def test_load_all_sheets():
    """Test that all Excel sheets load successfully with expected columns."""
    # Use sample file so the test works without the real dataset
    path = "data/raw/Tournament Results - Sample.xlsx"
    matches, tournaments, players = load_all_sheets(path)
    assert len(matches) > 0
    assert len(tournaments) > 0
    assert len(players) > 0
    # Check that key columns exist
    assert "match_date" in matches.columns
    assert "winner_id" in matches.columns
    assert "loser_id" in matches.columns


def test_merge_data():
    """Test that merging matches, tournaments, and players produces expected columns."""
    path = "data/raw/Tournament Results - Sample.xlsx"
    matches, tournaments, players = load_all_sheets(path)
    merged = merge_data(matches, tournaments, players)
    assert len(merged) > 0
    # Merged result should include tournament-level columns
    assert "country" in merged.columns or "level" in merged.columns
