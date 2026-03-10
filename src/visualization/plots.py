"""
可视化模块

All charts share a unified editorial theme inspired by data journalism
(FiveThirtyEight / The Economist): clean white background, restrained palette,
no chart junk, strong typographic hierarchy.

Usage:
    from src.visualization.plots import create_all_visualizations
    create_all_visualizations(df, output_dir="docs/plots")
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
import loguru

logger = loguru.logger


# ── Palette ───────────────────────────────────────────────────────────────────

BG = "#FAFAF7"
CARD = "#FFFFFF"
TEXT = "#1A1A1A"
TEXT_MUTED = "#6B6B6B"
GRID = "#EBEBEB"
SPINE = "#D0D0D0"

# Named colours for charting
NAVY = "#1D3557"
RED = "#E63946"
TEAL = "#2A9D8F"
AMBER = "#F4A261"
BLUE = "#457B9D"
SKY = "#8ECAE6"
OLIVE = "#606C38"
WINE = "#9B2226"

CATEGORICAL = [NAVY, AMBER, RED, TEAL, BLUE, WINE, SKY, OLIVE]
POS_COLOR = TEAL  # positive / advantage
NEG_COLOR = RED  # negative / disadvantage
REF_COLOR = "#AAAAAA"


# ── Global rcParams ──────────────────────────────────────────────────────────


def _apply_theme():
    plt.rcParams.update(
        {
            # Font
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.titlepad": 12,
            "axes.labelsize": 11,
            "axes.labelweight": "normal",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            # Colours
            "figure.facecolor": BG,
            "axes.facecolor": CARD,
            "axes.edgecolor": SPINE,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
            "text.color": TEXT,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            # Spines – only left + bottom by default
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Grid
            "axes.grid": True,
            "axes.axisbelow": True,
            # Figure
            "figure.dpi": 100,
            "savefig.dpi": 180,
            "savefig.bbox": "tight",
            "savefig.facecolor": BG,
            # Padding
            "figure.autolayout": False,
            # Lines
            "lines.linewidth": 2.0,
            "patch.linewidth": 0.6,
            # Legend
            "legend.framealpha": 0.9,
            "legend.edgecolor": SPINE,
            "legend.fontsize": 10,
        }
    )


_apply_theme()


# ── Helpers ───────────────────────────────────────────────────────────────────

LEVEL_ORDER = ["OG", "WC", "WTF", "S1000", "S750", "S500", "S300", "S100", "IS", "IC"]


def _subtitle(ax, text: str):
    """Add a muted subtitle below the axis title."""
    ax.text(
        0.0,
        1.02,
        text,
        transform=ax.transAxes,
        fontsize=9,
        color=TEXT_MUTED,
        ha="left",
        va="bottom",
    )


def _bar_label(ax, bars, fmt="{:.1f}%", va="bottom", pad=1.5, fontsize=9):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + pad if va == "bottom" else h - pad,
            fmt.format(h),
            ha="center",
            va=va,
            fontsize=fontsize,
            color=TEXT,
        )


def _hbar_label(ax, bars, fmt="{:.1f}%", pad=0.4, fontsize=9):
    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + pad,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(w),
            ha="left",
            va="center",
            fontsize=fontsize,
            color=TEXT,
        )


def _ref_line(ax, val, label=None, axis="y", color=REF_COLOR):
    kw = dict(color=color, linestyle="--", linewidth=1.4, zorder=1)
    if axis == "y":
        ax.axhline(val, **kw)
        if label:
            ax.text(ax.get_xlim()[1], val, f" {label}", va="center", fontsize=9, color=color)
    else:
        ax.axvline(val, **kw)


# ── Visualizer ────────────────────────────────────────────────────────────────


class DataVisualizer:
    """Generates analysis charts for the Happy-Badminton dataset."""

    def __init__(self, output_dir: str = "docs/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, name: str):
        fig.savefig(self.output_dir / f"{name}.png")
        plt.close(fig)
        logger.info(f"  ✓ {name}.png")

    # ── 1. Home Advantage ────────────────────────────────────────────────────

    def plot_home_advantage_by_level(self, df: pd.DataFrame):
        home = (
            df.groupby("level")
            .apply(
                lambda x: pd.Series(
                    {"rate": (x["winner_assoc"] == x["country"]).mean() * 100, "n": len(x)}
                )
            )
            .reset_index()
        )
        home["level"] = pd.Categorical(home["level"], categories=LEVEL_ORDER, ordered=True)
        home = home.sort_values("level").dropna(subset=["level"])
        home = home[home["n"] >= 50]

        fig, ax = plt.subplots(figsize=(11, 5))

        colors = [POS_COLOR if r > 50 else NEG_COLOR for r in home["rate"]]
        bars = ax.bar(home["level"], home["rate"], color=colors, width=0.65, zorder=2)

        _ref_line(ax, 50, "50% (expected)")
        _bar_label(ax, bars, fmt="{:.1f}%")

        ax.set_title("Home Advantage by Tournament Level")
        _subtitle(ax, "Win rate when the match is played in the player's home country")
        ax.set_ylabel("Home Win Rate (%)")
        ax.set_ylim(0, max(home["rate"]) * 1.18)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)

        # Sample-size annotation
        for i, (bar, n) in enumerate(zip(bars, home["n"])):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                2,
                f"n={n:,}",
                ha="center",
                fontsize=7.5,
                color="white",
                fontweight="bold",
            )

        fig.tight_layout()
        self._save(fig, "home_advantage_by_level")
        return fig

    # ── 2. ELO Accuracy ──────────────────────────────────────────────────────

    def plot_elo_accuracy_by_level(self, df: pd.DataFrame):
        df = df.copy()
        df["high_elo_won"] = df["winner_elo"] > df["loser_elo"]

        stats = (
            df.groupby("level")
            .apply(lambda x: pd.Series({"acc": x["high_elo_won"].mean() * 100, "n": len(x)}))
            .reset_index()
        )
        stats["level"] = pd.Categorical(stats["level"], categories=LEVEL_ORDER, ordered=True)
        stats = stats.sort_values("level").dropna(subset=["level"])

        global_avg = df["high_elo_won"].mean() * 100

        fig, ax = plt.subplots(figsize=(11, 5))
        bars = ax.bar(stats["level"], stats["acc"], color=NAVY, width=0.65, zorder=2)
        _ref_line(ax, global_avg, f"Global avg {global_avg:.1f}%")
        _bar_label(ax, bars, fmt="{:.1f}%")

        ax.set_title("Elo Rating Prediction Accuracy by Tournament Level")
        _subtitle(ax, "How often the higher-rated player wins")
        ax.set_ylabel("Elo Accuracy (%)")
        ax.set_ylim(0, max(stats["acc"]) * 1.18)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)

        fig.tight_layout()
        self._save(fig, "elo_accuracy_by_level")
        return fig

    # ── 3. Duration by Level ─────────────────────────────────────────────────

    def plot_duration_by_level(self, df: pd.DataFrame):
        stats = df.groupby("level")["duration"].agg(["mean", "median"]).reset_index()
        stats["level"] = pd.Categorical(stats["level"], categories=LEVEL_ORDER, ordered=True)
        stats = stats.sort_values("level").dropna(subset=["level"])

        x = np.arange(len(stats))
        w = 0.38

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(x - w / 2, stats["mean"], w, label="Mean", color=NAVY, zorder=2)
        ax.bar(x + w / 2, stats["median"], w, label="Median", color=AMBER, zorder=2)

        ax.set_xticks(x)
        ax.set_xticklabels(stats["level"])
        ax.set_title("Match Duration by Tournament Level")
        _subtitle(ax, "Average vs median match length in minutes")
        ax.set_ylabel("Duration (minutes)")
        ax.legend()
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)

        fig.tight_layout()
        self._save(fig, "duration_by_level")
        return fig

    # ── 4. Association Bias ───────────────────────────────────────────────────

    def plot_association_bias(self, df: pd.DataFrame):
        df = df.copy()
        df["elo_diff_abs"] = (df["winner_elo"] - df["loser_elo"]).abs()
        close = df[df["elo_diff_abs"] < 50]

        TOP_ASSOC = [
            "China",
            "Japan",
            "India",
            "Chinese Taipei",
            "Indonesia",
            "Denmark",
            "Malaysia",
            "Korea",
        ]
        rows = []
        for a in TOP_ASSOC:
            m = close[(close["winner_assoc"] == a) | (close["loser_assoc"] == a)]
            if len(m) >= 100:
                win_rate = (m["winner_assoc"] == a).mean() * 100
                rows.append({"assoc": a, "win_rate": win_rate, "n": len(m)})

        assoc_df = pd.DataFrame(rows).sort_values("win_rate", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 5.5))

        colors = [POS_COLOR if r > 50 else NEG_COLOR for r in assoc_df["win_rate"]]
        bars = ax.barh(assoc_df["assoc"], assoc_df["win_rate"], color=colors, height=0.65, zorder=2)
        _ref_line(ax, 50, axis="x")
        _hbar_label(ax, bars, fmt="{:.1f}%")

        ax.set_xlabel("Win Rate (%) — matches where Elo diff < 50")
        ax.set_title("Association Bias: Who wins the close ones?")
        _subtitle(ax, "Win rate in closely-matched games (Elo differential < 50)")
        ax.set_xlim(0, max(assoc_df["win_rate"]) * 1.22)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)

        fig.tight_layout()
        self._save(fig, "association_bias")
        return fig

    # ── 5. Cold Start ────────────────────────────────────────────────────────

    def plot_cold_start_impact(self, df: pd.DataFrame):
        df = df.copy()
        df["high_elo_won"] = df["winner_elo"] > df["loser_elo"]

        all_ids = pd.concat([df["winner_id"], df["loser_id"]])
        player_counts = all_ids.value_counts()
        sparse_ids = set(player_counts[player_counts <= 5].index)

        df["has_sparse"] = df["winner_id"].isin(sparse_ids) | df["loser_id"].isin(sparse_ids)

        acc = df.groupby("has_sparse")["high_elo_won"].mean() * 100

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Bar comparison
        labels = ["Has sparse\nplayer", "Both data-rich"]
        vals = [acc.get(True, 0), acc.get(False, 0)]
        colors = [NEG_COLOR, POS_COLOR]
        bars = ax1.bar(labels, vals, color=colors, width=0.5, zorder=2)
        _bar_label(ax1, bars, fmt="{:.1f}%", fontsize=11)
        ax1.set_ylim(0, max(vals) * 1.22)
        ax1.set_ylabel("Elo Accuracy (%)")
        ax1.set_title("Cold-Start Impact on Elo Accuracy")
        _subtitle(ax1, "Accuracy gap when a player has ≤5 historical matches")
        ax1.grid(axis="y")
        ax1.grid(axis="x", visible=False)

        # Pie
        sparse_counts = df["has_sparse"].value_counts()
        rich = sparse_counts.get(False, 0)
        sparse = sparse_counts.get(True, 0)
        wedge_colors = [POS_COLOR, NEG_COLOR]
        ax2.pie(
            [rich, sparse],
            labels=["Both data-rich", "Has sparse player"],
            colors=wedge_colors,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 10},
        )
        ax2.set_title("Sample Composition")

        fig.tight_layout(pad=2.5)
        self._save(fig, "cold_start_impact")
        return fig

    # ── 6. Data Overview ─────────────────────────────────────────────────────

    def plot_data_overview(self, df: pd.DataFrame):
        fig = plt.figure(figsize=(16, 11))
        gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

        # 1. Match type pie
        ax1 = fig.add_subplot(gs[0, 0])
        tc = df["type"].value_counts()
        ax1.pie(
            tc.values,
            labels=tc.index,
            autopct="%1.1f%%",
            colors=[NAVY, AMBER],
            startangle=90,
            textprops={"fontsize": 10, "fontweight": "bold"},
        )
        ax1.set_title("Match Type")

        # 2. Sets distribution
        ax2 = fig.add_subplot(gs[0, 1])
        sc = df["sets_played"].value_counts().sort_index()
        ax2.bar(sc.index.astype(str), sc.values, color=BLUE, width=0.5, zorder=2)
        for xi, (idx, v) in enumerate(sc.items()):
            ax2.text(xi, v + sc.values.max() * 0.02, f"{v:,}", ha="center", fontsize=9, color=TEXT)
        ax2.set_xlabel("Sets Played")
        ax2.set_title("Sets Distribution")
        ax2.grid(axis="y")
        ax2.grid(axis="x", visible=False)

        # 3. Duration histogram
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(df["duration"].dropna(), bins=50, color=TEAL, edgecolor="none", zorder=2)
        _ref_line(ax3, df["duration"].mean(), f"μ {df['duration'].mean():.0f}′", color=NAVY)
        _ref_line(ax3, df["duration"].median(), f"med {df['duration'].median():.0f}′", color=AMBER)
        ax3.set_xlabel("Duration (min)")
        ax3.set_title("Match Duration")
        ax3.grid(axis="y")
        ax3.grid(axis="x", visible=False)

        # 4. Level distribution (full width)
        ax4 = fig.add_subplot(gs[1, :])
        lc = df["level"].value_counts().reindex(LEVEL_ORDER).fillna(0)
        bars4 = ax4.bar(range(len(lc)), lc.values, color=NAVY, width=0.7, zorder=2)
        ax4.set_xticks(range(len(lc)))
        ax4.set_xticklabels(lc.index)
        ax4.set_xlabel("Tournament Level")
        ax4.set_ylabel("Matches")
        ax4.set_title("Match Count by Tournament Level")
        for bar in bars4:
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 50,
                f"{int(bar.get_height()):,}",
                ha="center",
                fontsize=8.5,
                color=TEXT,
            )
        ax4.grid(axis="y")
        ax4.grid(axis="x", visible=False)

        # 5. Monthly trend
        ax5 = fig.add_subplot(gs[2, 0])
        df2 = df.copy()
        df2["month"] = pd.to_datetime(df2["match_date"]).dt.to_period("M")
        monthly = df2.groupby("month").size()
        ax5.plot(range(len(monthly)), monthly.values, color=BLUE, linewidth=2, zorder=2)
        ax5.fill_between(range(len(monthly)), monthly.values, alpha=0.12, color=BLUE)
        step = max(1, len(monthly) // 6)
        ax5.set_xticks(range(0, len(monthly), step))
        ax5.set_xticklabels(
            [str(monthly.index[i]) for i in range(0, len(monthly), step)], rotation=35, ha="right"
        )
        ax5.set_ylabel("Matches")
        ax5.set_title("Matches Over Time")

        # 6. ELO distribution
        ax6 = fig.add_subplot(gs[2, 1])
        all_elo = pd.concat([df["winner_elo"], df["loser_elo"]]).dropna()
        ax6.hist(all_elo, bins=60, color=AMBER, edgecolor="none", zorder=2)
        _ref_line(ax6, all_elo.mean(), f"μ {all_elo.mean():.0f}", color=NAVY)
        ax6.set_xlabel("Elo Rating")
        ax6.set_title("Elo Rating Distribution")
        ax6.grid(axis="y")
        ax6.grid(axis="x", visible=False)

        # 7. Rank distribution
        ax7 = fig.add_subplot(gs[2, 2])
        all_ranks = pd.concat([df["winner_rank"], df["loser_rank"]]).dropna()
        all_ranks = all_ranks[all_ranks < 500]
        ax7.hist(all_ranks, bins=60, color=TEAL, edgecolor="none", zorder=2)
        ax7.set_xlabel("World Ranking")
        ax7.set_title("World Ranking Distribution")
        ax7.grid(axis="y")
        ax7.grid(axis="x", visible=False)

        fig.suptitle("Dataset Overview", fontsize=16, fontweight="bold", y=1.01)
        fig.tight_layout()
        self._save(fig, "data_overview")
        return fig

    # ── 7. Feature Correlation ───────────────────────────────────────────────

    def plot_feature_correlation(self, df: pd.DataFrame):
        numeric_features = [
            "duration",
            "winner_elo",
            "loser_elo",
            "winner_rank",
            "loser_rank",
            "total_points",
            "sets_played",
            "point_diff_set1",
            "seconds_per_point",
            "elo_diff",
            "log_rank_diff",
        ]
        available = [f for f in numeric_features if f in df.columns]
        corr = df[available].corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            center=0,
            square=True,
            linewidths=0.5,
            annot_kws={"size": 9},
            cbar_kws={"shrink": 0.75, "label": "Pearson r"},
            ax=ax,
        )
        ax.set_title("Feature Correlation Matrix")
        _subtitle(ax, "Lower triangle only — redundant cells masked")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=40, ha="right")
        fig.tight_layout()
        self._save(fig, "feature_correlation")
        return fig

    # ── 8. Player Activity ───────────────────────────────────────────────────

    def plot_player_activity(self, df: pd.DataFrame):
        ids = pd.concat(
            [
                df[["winner_id", "match_date"]],
                df[["loser_id", "match_date"]].rename(columns={"loser_id": "winner_id"}),
            ]
        )
        pstats = (
            ids.groupby("winner_id")
            .agg(
                match_count=("match_date", "count"),
                first=("match_date", "min"),
                last=("match_date", "max"),
            )
            .reset_index()
        )
        pstats["career_days"] = (pstats["last"] - pstats["first"]).dt.days

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        # Match count histogram
        ax1.hist(pstats["match_count"], bins=60, color=NAVY, edgecolor="none", zorder=2)
        _ref_line(ax1, pstats["match_count"].mean(), f"μ {pstats['match_count'].mean():.1f}")
        _ref_line(
            ax1,
            pstats["match_count"].median(),
            f"med {pstats['match_count'].median():.0f}",
            color=AMBER,
        )
        ax1.set_xlim(0, 120)
        ax1.set_xlabel("Matches per Player")
        ax1.set_ylabel("Players")
        ax1.set_title("Player Activity Distribution")
        _subtitle(ax1, "Most players appear only a handful of times")
        ax1.grid(axis="y")
        ax1.grid(axis="x", visible=False)

        # Career length
        ax2.hist(pstats["career_days"].dropna(), bins=60, color=TEAL, edgecolor="none", zorder=2)
        _ref_line(ax2, pstats["career_days"].mean(), f"μ {pstats['career_days'].mean():.0f}d")
        ax2.set_xlim(0, 1300)
        ax2.set_xlabel("Career Span (days)")
        ax2.set_ylabel("Players")
        ax2.set_title("Player Career Span")
        ax2.grid(axis="y")
        ax2.grid(axis="x", visible=False)

        fig.tight_layout(pad=2.5)
        self._save(fig, "player_activity")
        return fig

    # ── 9. Win Rate by Rank ──────────────────────────────────────────────────

    def plot_win_rate_by_rank(self, df: pd.DataFrame):
        w = df[["winner_rank"]].assign(won=1).rename(columns={"winner_rank": "rank"})
        l = df[["loser_rank"]].assign(won=0).rename(columns={"loser_rank": "rank"})
        all_p = pd.concat([w, l])
        all_p = all_p[all_p["rank"] < 500]

        bins = [0, 10, 20, 30, 50, 100, 500]
        labels = ["1–10", "11–20", "21–30", "31–50", "51–100", "101–500"]
        all_p["group"] = pd.cut(all_p["rank"], bins=bins, labels=labels)
        wr = all_p.groupby("group", observed=True)["won"].mean() * 100

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = [POS_COLOR if v > 50 else NEG_COLOR for v in wr]
        bars = ax.bar(range(len(wr)), wr.values, color=colors, width=0.6, zorder=2)
        _ref_line(ax, 50, "50% (random)")
        _bar_label(ax, bars, fmt="{:.1f}%")

        ax.set_xticks(range(len(wr)))
        ax.set_xticklabels(wr.index)
        ax.set_xlabel("World Ranking Band")
        ax.set_ylabel("Win Rate (%)")
        ax.set_title("Win Rate by World Ranking")
        _subtitle(ax, "Top-ranked players win significantly more than chance")
        ax.set_ylim(0, max(wr.values) * 1.2)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)

        fig.tight_layout()
        self._save(fig, "win_rate_by_rank")
        return fig

    # ── 10. Association ELO Distribution ─────────────────────────────────────

    def plot_association_elo_distribution(self, df: pd.DataFrame):
        top_assoc = df["winner_assoc"].value_counts().head(10).index.tolist()

        data_to_plot = []
        for a in top_assoc:
            elo = pd.concat(
                [
                    df[df["winner_assoc"] == a]["winner_elo"],
                    df[df["loser_assoc"] == a]["loser_elo"],
                ]
            ).dropna()
            data_to_plot.append(elo.values)

        fig, ax = plt.subplots(figsize=(12, 5.5))
        bp = ax.boxplot(
            data_to_plot,
            labels=top_assoc,
            patch_artist=True,
            medianprops=dict(color=AMBER, linewidth=2),
            whiskerprops=dict(color=SPINE),
            capprops=dict(color=SPINE),
            flierprops=dict(marker="o", markersize=3, alpha=0.3, color=TEXT_MUTED),
        )

        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(CATEGORICAL[i % len(CATEGORICAL)])
            patch.set_alpha(0.8)

        ax.set_ylabel("Elo Rating")
        ax.set_title("Elo Rating Distribution by Association")
        _subtitle(ax, "Median, IQR, and outliers — top-10 associations by match count")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right")
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)

        fig.tight_layout()
        self._save(fig, "association_elo_distribution")
        return fig

    # ── 11. Three-Set Rate ────────────────────────────────────────────────────

    def plot_three_set_rate_by_level(self, df: pd.DataFrame):
        rate = (
            df.groupby("level")["sets_played"]
            .apply(lambda x: (x == 3).mean() * 100)
            .reset_index(name="three_set_rate")
        )
        rate["level"] = pd.Categorical(rate["level"], categories=LEVEL_ORDER, ordered=True)
        rate = rate.sort_values("level").dropna(subset=["level"])

        fig, ax = plt.subplots(figsize=(11, 5))
        bars = ax.bar(rate["level"], rate["three_set_rate"], color=AMBER, width=0.65, zorder=2)
        _bar_label(ax, bars, fmt="{:.1f}%")
        _ref_line(ax, rate["three_set_rate"].mean(), f"avg {rate['three_set_rate'].mean():.1f}%")

        ax.set_ylabel("3-Set Match Rate (%)")
        ax.set_title("Three-Set Match Rate by Tournament Level")
        _subtitle(ax, "Higher-tier events tend to produce more contested (3-set) matches")
        ax.set_ylim(0, rate["three_set_rate"].max() * 1.22)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)

        fig.tight_layout()
        self._save(fig, "three_set_rate_by_level")
        return fig

    # ── 12. MS vs WS Feature Distributions ───────────────────────────────────

    def plot_feature_distributions(self, df: pd.DataFrame):
        features = [
            f
            for f in ["duration", "seconds_per_point", "total_points", "sets_played"]
            if f in df.columns
        ]

        n = len(features)
        ncols = 2
        nrows = (n + 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(13, nrows * 4.5))
        axes = axes.flatten()

        for i, feat in enumerate(features):
            ax = axes[i]
            for mtype, color, lbl in [("MS", NAVY, "MS"), ("WS", AMBER, "WS")]:
                data = df[df["type"] == mtype][feat].dropna()
                ax.hist(
                    data, bins=50, alpha=0.55, color=color, label=lbl, edgecolor="none", zorder=2
                )
            ax.set_xlabel(feat.replace("_", " ").title())
            ax.set_ylabel("Frequency")
            ax.set_title(f"{feat.replace('_', ' ').title()} — MS vs WS")
            ax.legend()
            ax.grid(axis="y")
            ax.grid(axis="x", visible=False)

        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        fig.tight_layout(pad=2.5)
        self._save(fig, "feature_distributions_ms_ws")
        return fig

    # ── 13. Monthly Trends ────────────────────────────────────────────────────

    def plot_monthly_trends(self, df: pd.DataFrame):
        df = df.copy()
        df["month"] = pd.to_datetime(df["match_date"]).dt.to_period("M")

        monthly = (
            df.groupby("month")
            .agg(
                match_count=("winner_id", "count"),
                avg_duration=("duration", "mean"),
                three_set_rate=("sets_played", lambda x: (x == 3).mean() * 100),
            )
            .reset_index()
        )

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        colors = [NAVY, TEAL, AMBER]
        titles = ["Monthly Match Count", "Average Match Duration (min)", "Three-Set Rate (%)"]
        cols = ["match_count", "avg_duration", "three_set_rate"]
        ylabs = ["Matches", "Minutes", "%"]

        xs = range(len(monthly))
        for ax, col, title, color, ylab in zip(axes, cols, titles, colors, ylabs):
            ax.plot(xs, monthly[col], color=color, linewidth=2, zorder=2)
            ax.fill_between(xs, monthly[col], alpha=0.10, color=color)
            ax.set_ylabel(ylab)
            ax.set_title(title)
            ax.grid(axis="y")
            ax.grid(axis="x", visible=False)

        step = max(1, len(monthly) // 9)
        ticks = range(0, len(monthly), step)
        axes[-1].set_xticks(ticks)
        axes[-1].set_xticklabels(
            [str(monthly["month"].iloc[i]) for i in ticks], rotation=35, ha="right"
        )
        axes[-1].set_xlabel("Month")

        fig.suptitle("Monthly Trends", fontsize=15, fontweight="bold")
        fig.tight_layout(pad=2.5)
        self._save(fig, "monthly_trends")
        return fig

    # ── 14. Comprehensive Dashboard ───────────────────────────────────────────

    def plot_comprehensive_dashboard(self, df: pd.DataFrame):
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 4, hspace=0.45, wspace=0.38)

        df = df.copy()
        df["high_elo_won"] = df["winner_elo"] > df["loser_elo"]

        # ── Row 0 ──────────────────────────────────────────────────────────
        # 0a. Home advantage
        ax = fig.add_subplot(gs[0, 0])
        home = (
            df.groupby("level")
            .apply(lambda x: (x["winner_assoc"] == x["country"]).mean() * 100)
            .reset_index(name="rate")
        )
        home["level"] = pd.Categorical(home["level"], categories=LEVEL_ORDER, ordered=True)
        home = home.sort_values("level").dropna(subset=["level"])
        colors = [POS_COLOR if r > 50 else NEG_COLOR for r in home["rate"]]
        ax.bar(range(len(home)), home["rate"], color=colors, width=0.7, zorder=2)
        ax.axhline(50, color=REF_COLOR, linestyle="--", linewidth=1)
        ax.set_xticks(range(len(home)))
        ax.set_xticklabels(home["level"], fontsize=7.5, rotation=40, ha="right")
        ax.set_title("Home Win %", fontsize=10)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)

        # 0b. Match type
        ax = fig.add_subplot(gs[0, 1])
        tc = df["type"].value_counts()
        ax.pie(
            tc.values,
            labels=tc.index,
            autopct="%1.1f%%",
            colors=[NAVY, AMBER],
            startangle=90,
            textprops={"fontsize": 10},
        )
        ax.set_title("Match Type", fontsize=10)

        # 0c. Sets
        ax = fig.add_subplot(gs[0, 2])
        sc = df["sets_played"].value_counts().sort_index()
        ax.bar(sc.index.astype(str), sc.values, color=BLUE, width=0.5, zorder=2)
        ax.set_title("Sets Distribution", fontsize=10)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)

        # 0d. Duration
        ax = fig.add_subplot(gs[0, 3])
        ax.hist(df["duration"].dropna(), bins=30, color=TEAL, edgecolor="none", zorder=2)
        ax.axvline(df["duration"].mean(), color=NAVY, linestyle="--", linewidth=1.5)
        ax.set_title(f"Duration (μ={df['duration'].mean():.0f}′)", fontsize=10)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)

        # ── Row 1 ──────────────────────────────────────────────────────────
        # 1a. Elo accuracy by level
        ax = fig.add_subplot(gs[1, :2])
        eac = df.groupby("level")["high_elo_won"].mean().reset_index(name="acc")
        eac["level"] = pd.Categorical(eac["level"], categories=LEVEL_ORDER, ordered=True)
        eac = eac.sort_values("level").dropna(subset=["level"])
        ax.bar(eac["level"], eac["acc"] * 100, color=NAVY, width=0.65, zorder=2)
        ax.axhline(
            df["high_elo_won"].mean() * 100,
            color=AMBER,
            linestyle="--",
            linewidth=1.5,
            label=f"Global {df['high_elo_won'].mean() * 100:.1f}%",
        )
        ax.set_title("Elo Accuracy by Level", fontsize=11)
        ax.set_ylabel("%")
        ax.legend(fontsize=9)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)

        # 1b. Avg Elo by association
        ax = fig.add_subplot(gs[1, 2:])
        top8 = df["winner_assoc"].value_counts().head(8).index.tolist()
        avg_elo = [
            (
                a,
                pd.concat(
                    [
                        df[df["winner_assoc"] == a]["winner_elo"],
                        df[df["loser_assoc"] == a]["loser_elo"],
                    ]
                ).mean(),
            )
            for a in top8
        ]
        avg_elo.sort(key=lambda x: x[1], reverse=True)
        assocs, elos = zip(*avg_elo)
        bars = ax.barh(range(len(assocs)), elos, color=BLUE, zorder=2)
        ax.set_yticks(range(len(assocs)))
        ax.set_yticklabels(assocs, fontsize=9)
        ax.invert_yaxis()
        ax.set_title("Average Elo by Association", fontsize=11)
        ax.set_xlabel("Elo")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)

        # ── Row 2 ──────────────────────────────────────────────────────────
        # 2a. Cold start
        ax = fig.add_subplot(gs[2, :2])
        all_ids = pd.concat([df["winner_id"], df["loser_id"]])
        sparse_ids = set(all_ids.value_counts()[lambda x: x <= 5].index)
        df["has_sparse"] = df["winner_id"].isin(sparse_ids) | df["loser_id"].isin(sparse_ids)
        cold = df.groupby("has_sparse")["high_elo_won"].mean() * 100
        ax.bar(
            ["Has sparse", "Both rich"],
            [cold.get(True, 0), cold.get(False, 0)],
            color=[NEG_COLOR, POS_COLOR],
            width=0.5,
            zorder=2,
        )
        for i, v in enumerate([cold.get(True, 0), cold.get(False, 0)]):
            ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(cold.values) * 1.22)
        ax.set_ylabel("Elo Accuracy %")
        ax.set_title("Cold-Start Impact", fontsize=11)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)

        # 2b. Monthly trend
        ax = fig.add_subplot(gs[2, 2:])
        df["month"] = pd.to_datetime(df["match_date"]).dt.to_period("M")
        monthly = df.groupby("month").size()
        ax.plot(range(len(monthly)), monthly.values, color=NAVY, linewidth=2, zorder=2)
        ax.fill_between(range(len(monthly)), monthly.values, alpha=0.12, color=NAVY)
        ax.set_title("Matches Over Time", fontsize=11)
        ax.set_ylabel("Matches")
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)

        # ── Row 3: Stats summary ───────────────────────────────────────────
        ax = fig.add_subplot(gs[3, :])
        ax.axis("off")
        three_set = (df["sets_played"] == 3).mean() * 100 if "sets_played" in df.columns else 0
        total_pts = df["total_points"].mean() if "total_points" in df.columns else 0
        summary = (
            f"  DATASET                                                   "
            f"ELO STATS                                                   "
            f"MATCH STATS\n"
            f"  Total matches : {len(df):>8,}                              "
            f"  Winner avg Elo : {df['winner_elo'].mean():>7.0f}                             "
            f"  Avg duration  : {df['duration'].mean():>6.1f} min\n"
            f"  Players       : {df['winner_id'].nunique() + df['loser_id'].nunique():>8,}                              "
            f"  Loser avg Elo  : {df['loser_elo'].mean():>7.0f}                             "
            f"  3-set rate    : {three_set:>6.2f}%\n"
            f"  Date range    : {str(df['match_date'].min())[:10]} → {str(df['match_date'].max())[:10]}        "
            f"  Elo accuracy   : {df['high_elo_won'].mean() * 100:>6.2f}%                             "
            f"  Avg points    : {total_pts:>6.1f}\n"
        )
        ax.text(
            0.02,
            0.5,
            summary,
            transform=ax.transAxes,
            fontsize=10,
            fontfamily="monospace",
            va="center",
            color=TEXT,
            bbox=dict(facecolor=GRID, edgecolor=SPINE, boxstyle="round,pad=0.6"),
        )

        fig.suptitle(
            "Happy-Badminton — Comprehensive Data Dashboard", fontsize=16, fontweight="bold", y=1.01
        )
        fig.tight_layout()
        self._save(fig, "comprehensive_dashboard")
        return fig

    # ── Model evaluation plots (optional) ────────────────────────────────────

    def plot_calibration_curve(self, y_true, y_prob, n_bins: int = 10):
        from sklearn.calibration import calibration_curve

        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(
            [0, 1],
            [0, 1],
            color=REF_COLOR,
            linestyle="--",
            linewidth=1.5,
            label="Perfect calibration",
        )
        ax.plot(prob_pred, prob_true, "o-", color=NAVY, linewidth=2, markersize=7, label="Model")
        ax.fill_between(prob_pred, prob_true, prob_pred, alpha=0.1, color=NAVY)
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Empirical Probability")
        ax.set_title("Probability Calibration Curve")
        _subtitle(ax, "How accurately the model's confidence maps to real outcomes")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        self._save(fig, "calibration_curve")
        return fig

    def plot_roc_curve(self, y_true, y_score):
        from sklearn.metrics import roc_curve, auc

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot([0, 1], [0, 1], color=REF_COLOR, linestyle="--", linewidth=1.5)
        ax.plot(fpr, tpr, color=NAVY, linewidth=2, label=f"ROC curve  (AUC = {roc_auc:.3f})")
        ax.fill_between(fpr, tpr, alpha=0.08, color=NAVY)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        self._save(fig, "roc_curve")
        return fig


# ── Public API ────────────────────────────────────────────────────────────────


def create_all_visualizations(
    df: pd.DataFrame,
    output_dir: str = "docs/plots",
    include_model_plots: bool = False,
    y_true=None,
    y_pred=None,
) -> DataVisualizer:
    """
    Generate all data-exploration charts.

    Args:
        df:                  Preprocessed match DataFrame (output of preprocess_pipeline).
        output_dir:          Directory to save PNG files.
        include_model_plots: If True, also generate calibration + ROC curves.
        y_true:              Ground-truth labels (required when include_model_plots=True).
        y_pred:              Predicted probabilities (required when include_model_plots=True).

    Returns:
        DataVisualizer instance.
    """
    viz = DataVisualizer(output_dir)

    logger.info(f"Generating charts → {output_dir}")

    viz.plot_home_advantage_by_level(df)
    viz.plot_elo_accuracy_by_level(df)
    viz.plot_duration_by_level(df)
    viz.plot_association_bias(df)
    viz.plot_cold_start_impact(df)
    viz.plot_data_overview(df)
    viz.plot_feature_correlation(df)
    viz.plot_player_activity(df)
    viz.plot_win_rate_by_rank(df)
    viz.plot_association_elo_distribution(df)
    viz.plot_three_set_rate_by_level(df)
    viz.plot_feature_distributions(df)
    viz.plot_monthly_trends(df)
    viz.plot_comprehensive_dashboard(df)

    if include_model_plots and y_true is not None and y_pred is not None:
        viz.plot_calibration_curve(y_true, y_pred)
        viz.plot_roc_curve(y_true, y_pred)

    logger.info(f"Done — {len(list(Path(output_dir).glob('*.png')))} charts in {output_dir}")
    return viz
