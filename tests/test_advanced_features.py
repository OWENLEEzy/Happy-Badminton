"""Tests for src/data/advanced_features.py core feature classes."""

import sys
from pathlib import Path

import pandas as pd
import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.advanced_features import (
    HeadToHeadFeatures,
    MOVEloRating,
    NationalityFeatures,
    build_advanced_features,
    build_nat_pair_lookup,
    get_advanced_feature_columns,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _two_player_df(n_matches: int = 3) -> pd.DataFrame:
    """Minimal match DataFrame with two players A vs B, played on consecutive days."""
    dates = pd.date_range("2024-01-01", periods=n_matches, freq="D")
    return pd.DataFrame(
        {
            "match_date": dates,
            "winner_id": ["A"] * n_matches,
            "loser_id": ["B"] * n_matches,
            "winner_assoc": ["China"] * n_matches,
            "loser_assoc": ["Japan"] * n_matches,
            "country": ["China"] * n_matches,
            "level": ["IS"] * n_matches,
            "winner_points": [21] * n_matches,
            "loser_points": [15] * n_matches,
            "sets_played": [2] * n_matches,
        }
    )


# ---------------------------------------------------------------------------
# MOVEloRating — pre-match recording
# ---------------------------------------------------------------------------


class TestMOVEloRating:
    def test_first_match_starts_at_initial_elo(self) -> None:
        """Both players' recorded Elo on their first match must equal INITIAL_ELO."""
        df = _two_player_df(2)
        elo = MOVEloRating()
        result = elo.compute_mov_elo_history(df)

        assert result["winner_mov_elo"].iloc[0] == pytest.approx(MOVEloRating.INITIAL_ELO)
        assert result["loser_mov_elo"].iloc[0] == pytest.approx(MOVEloRating.INITIAL_ELO)

    def test_elo_recorded_before_update(self) -> None:
        """Second match must reflect ratings from after the first match, not initial."""
        df = _two_player_df(2)
        elo = MOVEloRating()
        result = elo.compute_mov_elo_history(df)

        # Winner should have gained Elo after match 1, so match 2 winner_mov_elo > INITIAL
        assert result["winner_mov_elo"].iloc[1] > MOVEloRating.INITIAL_ELO
        # Loser should have lost Elo, so match 2 loser_mov_elo < INITIAL
        assert result["loser_mov_elo"].iloc[1] < MOVEloRating.INITIAL_ELO

    def test_winner_gains_elo_loser_loses(self) -> None:
        """After one match, winner's post-match Elo > INITIAL, loser's < INITIAL."""
        elo = MOVEloRating()
        w_new, l_new = elo.update_ratings("A", "B", 21, 15, 2)
        assert w_new > MOVEloRating.INITIAL_ELO
        assert l_new < MOVEloRating.INITIAL_ELO

    def test_mov_diff_column_equals_w_minus_l(self) -> None:
        """mov_elo_diff must exactly equal winner_mov_elo - loser_mov_elo row-wise."""
        df = _two_player_df(3)
        elo = MOVEloRating()
        result = elo.compute_mov_elo_history(df)

        expected = result["winner_mov_elo"] - result["loser_mov_elo"]
        pd.testing.assert_series_equal(
            result["mov_elo_diff"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_mov_multiplier_three_sets_higher(self) -> None:
        """3-set match should have a higher MOV multiplier than 2-set with same scores."""
        elo = MOVEloRating()
        mult_2 = elo.calculate_mov_multiplier(21, 15, 2)
        mult_3 = elo.calculate_mov_multiplier(21, 15, 3)
        assert mult_3 > mult_2

    def test_mov_multiplier_equal_scores_returns_one(self) -> None:
        """When both players score 0 (edge case), multiplier must be 1.0."""
        elo = MOVEloRating()
        assert elo.calculate_mov_multiplier(0, 0, 2) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# HeadToHeadFeatures — shift-safe H2H
# ---------------------------------------------------------------------------


class TestHeadToHeadFeatures:
    def test_first_match_prior(self) -> None:
        """Before any H2H history exists, h2h_win_rate must be 0.5 and h2h_matches 0."""
        df = _two_player_df(3)
        h2h = HeadToHeadFeatures(df)
        result = h2h.compute_h2h_features()

        assert result["h2h_win_rate"].iloc[0] == pytest.approx(0.5)
        assert result["h2h_matches"].iloc[0] == 0

    def test_second_match_reflects_first_only(self) -> None:
        """After 1 match where A beat B, second match h2h should show A won 1/1."""
        df = _two_player_df(3)
        h2h = HeadToHeadFeatures(df)
        result = h2h.compute_h2h_features()

        # At second match: A has beaten B once, total=1
        assert result["h2h_matches"].iloc[1] == 1
        assert result["h2h_win_rate"].iloc[1] == pytest.approx(1.0)

    def test_h2h_accumulates_monotonically(self) -> None:
        """h2h_matches must be non-decreasing over consecutive A-vs-B matches."""
        df = _two_player_df(5)
        h2h = HeadToHeadFeatures(df)
        result = h2h.compute_h2h_features()

        counts = result["h2h_matches"].tolist()
        assert counts == sorted(counts), "h2h_matches should be non-decreasing"

    def test_no_leakage_current_match_not_counted(self) -> None:
        """The current match result must NOT appear in its own h2h_matches count."""
        df = _two_player_df(1)
        h2h = HeadToHeadFeatures(df)
        result = h2h.compute_h2h_features()

        assert result["h2h_matches"].iloc[0] == 0

    def test_symmetric_pair_key(self) -> None:
        """H2H counts must be the same regardless of which player is 'winner' alphabetically."""
        # A beats B three times, then B beats A (now B is winner) — total for pair should be 4
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        df = pd.DataFrame(
            {
                "match_date": dates,
                "winner_id": ["A", "A", "A", "B"],
                "loser_id": ["B", "B", "B", "A"],
                "winner_assoc": ["China"] * 4,
                "loser_assoc": ["Japan"] * 4,
                "country": ["China"] * 4,
                "level": ["IS"] * 4,
                "winner_points": [21] * 4,
                "loser_points": [15] * 4,
                "sets_played": [2] * 4,
            }
        )
        h2h = HeadToHeadFeatures(df)
        result = h2h.compute_h2h_features()

        # Fourth match: pair total should be 3 (3 prior matches between A and B)
        assert result["h2h_matches"].iloc[3] == 3


# ---------------------------------------------------------------------------
# NationalityFeatures
# ---------------------------------------------------------------------------


class TestNationalityFeatures:
    def _derby_df(self) -> pd.DataFrame:
        """Two matches between same-nation players (China derby)."""
        dates = pd.date_range("2024-01-01", periods=2, freq="D")
        return pd.DataFrame(
            {
                "match_date": dates,
                "winner_id": ["A", "C"],
                "loser_id": ["B", "D"],
                "winner_assoc": ["China", "China"],
                "loser_assoc": ["China", "China"],
                "country": ["Japan", "Japan"],
            }
        )

    def _cross_df(self) -> pd.DataFrame:
        """Two cross-nation matches: winner from Asia (China), loser from Europe (Denmark)."""
        dates = pd.date_range("2024-01-01", periods=2, freq="D")
        return pd.DataFrame(
            {
                "match_date": dates,
                "winner_id": ["A", "A"],
                "loser_id": ["B", "B"],
                "winner_assoc": ["China", "China"],
                "loser_assoc": ["Denmark", "Denmark"],
                "country": ["China", "Denmark"],
            }
        )

    def test_same_nationality_flag_derby(self) -> None:
        """same_nationality must be 1 when both players share the same assoc."""
        nat = NationalityFeatures(self._derby_df())
        result = nat.add_nationality_features()
        assert (result["same_nationality"] == 1).all()

    def test_same_nationality_flag_cross(self) -> None:
        """same_nationality must be 0 for different-nation matchups."""
        nat = NationalityFeatures(self._cross_df())
        result = nat.add_nationality_features()
        assert (result["same_nationality"] == 0).all()

    def test_nat_matchup_first_encounter_zero(self) -> None:
        """On first encounter between two nations, nat_matchup_win_diff must be 0.0."""
        nat = NationalityFeatures(self._cross_df())
        result = nat.add_nationality_features()
        assert result["nat_matchup_win_diff"].iloc[0] == pytest.approx(0.0)

    def test_continent_home_china_host(self) -> None:
        """When host country is China, winner from China should get continent_home=1."""
        nat = NationalityFeatures(self._cross_df())
        result = nat.add_nationality_features()

        # First match: host = China, winner from China -> winner_continent_home should be 1
        assert result["winner_continent_home"].iloc[0] == 1
        # Loser from Denmark -> loser_continent_home should be 0
        assert result["loser_continent_home"].iloc[0] == 0

    def test_continent_advantage_diff_equals_w_minus_l(self) -> None:
        """continent_advantage_diff must equal winner_continent_home - loser_continent_home."""
        nat = NationalityFeatures(self._cross_df())
        result = nat.add_nationality_features()

        expected = result["winner_continent_home"] - result["loser_continent_home"]
        pd.testing.assert_series_equal(
            result["continent_advantage_diff"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )


# ---------------------------------------------------------------------------
# Helpers for pipeline-level tests
# ---------------------------------------------------------------------------


def _full_match_df(n: int = 6) -> pd.DataFrame:
    """Full match DataFrame with every column required by build_advanced_features."""
    return pd.DataFrame(
        {
            "match_date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "winner_id": ["A"] * n,
            "loser_id": ["B"] * n,
            "winner_assoc": ["China"] * n,
            "loser_assoc": ["Japan"] * n,
            "country": ["China"] * n,
            "level": ["IS"] * n,
            "winner_points": [21] * n,
            "loser_points": [15] * n,
            "sets_played": [2] * n,
            "duration": [50.0] * n,
        }
    )


# ---------------------------------------------------------------------------
# build_nat_pair_lookup
# ---------------------------------------------------------------------------


class TestBuildNatPairLookup:
    def test_returns_dict(self) -> None:
        """build_nat_pair_lookup() must return a dict."""
        result = build_nat_pair_lookup(_full_match_df(5))
        assert isinstance(result, dict)

    def test_keys_are_canonical(self) -> None:
        """Keys must follow 'Min|Max' alphabetical ordering."""
        result = build_nat_pair_lookup(_full_match_df(5))
        for key in result:
            a, b = key.split("|", 1)
            assert a <= b, f"Key not in canonical form: {key}"

    def test_values_in_unit_interval(self) -> None:
        """All win-rate values must be in (0, 1)."""
        result = build_nat_pair_lookup(_full_match_df(5))
        for v in result.values():
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# build_advanced_features
# ---------------------------------------------------------------------------


class TestBuildAdvancedFeatures:
    def test_row_count_unchanged(self) -> None:
        """build_advanced_features() must not drop or duplicate any rows."""
        df = _full_match_df(6)
        result = build_advanced_features(df)
        assert len(result) == len(df)

    def test_key_feature_columns_added(self) -> None:
        """Output must include form, H2H, fatigue, and nationality columns."""
        df = _full_match_df(6)
        result = build_advanced_features(df)
        for col in [
            "winner_form_5",
            "h2h_win_rate",
            "winner_fatigue",
            "same_nationality",
            "winner_mov_elo",
        ]:
            assert col in result.columns, f"Missing column: {col}"

    def test_original_columns_preserved(self) -> None:
        """All original columns must still be present after the pipeline."""
        df = _full_match_df(6)
        original_cols = set(df.columns)
        result = build_advanced_features(df)
        assert original_cols.issubset(set(result.columns))


# ---------------------------------------------------------------------------
# get_advanced_feature_columns
# ---------------------------------------------------------------------------


class TestGetAdvancedFeatureColumns:
    def test_returns_non_empty_list(self) -> None:
        """Must return a non-empty list."""
        cols = get_advanced_feature_columns()
        assert isinstance(cols, list)
        assert len(cols) > 0

    def test_contains_expected_columns(self) -> None:
        """List must include representative columns from each feature group."""
        cols = get_advanced_feature_columns()
        for expected in [
            "winner_form_5",
            "h2h_win_rate",
            "winner_fatigue",
            "same_nationality",
            "winner_mov_elo",
        ]:
            assert expected in cols
