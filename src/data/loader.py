"""
Data loading module.

Functionality:
1. Read three Excel sheets (Matches, Tournament, Player)
2. Merge data
3. Return a complete DataFrame
"""

import pandas as pd
from pathlib import Path
from typing import Tuple
import loguru

logger = loguru.logger


def load_all_sheets(filepath: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all three sheets from an Excel file.

    Args:
        filepath: Path to the Excel file

    Returns:
        (matches, tournaments, players) as three DataFrames
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    logger.info(f"Loading data: {filepath}")

    # Read all three sheets
    matches = pd.read_excel(filepath, sheet_name="Matches")
    tournaments = pd.read_excel(filepath, sheet_name="Tournament")
    players = pd.read_excel(filepath, sheet_name="Player")

    logger.info(f"  Matches: {matches.shape[0]} rows x {matches.shape[1]} cols")
    logger.info(f"  Tournaments: {tournaments.shape[0]} rows x {tournaments.shape[1]} cols")
    logger.info(f"  Players: {players.shape[0]} rows x {players.shape[1]} cols")

    return matches, tournaments, players


def merge_data(
    matches: pd.DataFrame, tournaments: pd.DataFrame, players: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge all three sheets into a single DataFrame.

    Args:
        matches: Match data
        tournaments: Tournament data
        players: Player data

    Returns:
        Merged complete DataFrame
    """
    logger.info("Merging data...")

    # 1. Merge tournament info
    df = matches.merge(
        tournaments[["id", "level", "country"]],
        left_on="tournament_id",
        right_on="id",
        how="left",
        suffixes=("", "_tournament"),
    )

    # 2. Merge winner association info
    df = df.merge(
        players[["id", "association"]],
        left_on="winner_id",
        right_on="id",
        how="left",
        suffixes=("", "_player_winner"),
    )
    df = df.rename(columns={"association": "winner_assoc"})

    # 3. Merge loser association info
    df = df.merge(
        players[["id", "association"]],
        left_on="loser_id",
        right_on="id",
        how="left",
        suffixes=("", "_player_loser"),
    )
    df = df.rename(columns={"association": "loser_assoc"})

    # 4. Drop temporary suffix columns
    cols_to_drop = [
        c
        for c in df.columns
        if c.endswith("_tournament") or c.endswith("_player_winner") or c.endswith("_player_loser")
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    logger.info(f"Merge complete: {df.shape[0]} rows x {df.shape[1]} cols")

    return df


def load_and_merge(filepath: str) -> pd.DataFrame:
    """
    One-stop load: read Excel and merge all sheets.

    Args:
        filepath: Path to the Excel file

    Returns:
        Merged complete DataFrame
    """
    matches, tournaments, players = load_all_sheets(filepath)
    df = merge_data(matches, tournaments, players)
    return df


if __name__ == "__main__":
    # Test
    from src.utils.logger import setup_logger

    setup_logger()

    df = load_and_merge("data/raw/Tournament Results.xlsx")
    logger.info(f"Merged data preview:\n{df.head()}")
    logger.info(f"Data types:\n{df.dtypes}")
