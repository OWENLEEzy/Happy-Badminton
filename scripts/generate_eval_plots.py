"""
Generate model evaluation plots for SimplifiedEnsemble.

Produces:
  docs/plots/eval_roc_curve.png          — ROC curve with AUC
  docs/plots/eval_pr_curve.png           — Precision-Recall curve
  docs/plots/eval_confusion_matrix.png   — Confusion matrix (counts + %)
  docs/plots/eval_calibration.png        — Calibration / reliability diagram
  docs/plots/eval_score_distribution.png — Predicted probability histogram by class
  docs/plots/eval_feature_importance.png — Top-20 LGBM feature importances
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.validate_model import load_config, prepare_simplified_data  # noqa: E402
from src.utils.logger import setup_logger

logger = setup_logger()

PLOT_DIR = project_root / "docs" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

STYLE = {
    "primary": "#1a1a2e",
    "accent": "#e94560",
    "positive": "#16213e",
    "neutral": "#0f3460",
    "bg": "#f5f5f5",
    "grid": "#dddddd",
}


def _base_fig(title: str, figsize: tuple = (7, 5)):
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor(STYLE["bg"])
    ax.set_facecolor(STYLE["bg"])
    ax.set_title(title, fontsize=13, fontweight="bold", color=STYLE["primary"], pad=12)
    ax.grid(True, color=STYLE["grid"], linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = _base_fig(f"ROC Curve  (AUC = {auc:.4f})")
    ax.plot(fpr, tpr, color=STYLE["accent"], lw=2, label=f"SimplifiedEnsemble (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], color=STYLE["grid"], lw=1.5, linestyle="--", label="Random (AUC=0.50)")
    ax.fill_between(fpr, tpr, alpha=0.08, color=STYLE["accent"])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.legend(fontsize=10, framealpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "eval_roc_curve.png")
    plt.close(fig)
    logger.info("  ✓ eval_roc_curve.png")


def plot_pr(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    baseline = y_true.mean()

    fig, ax = _base_fig(f"Precision-Recall Curve  (AP = {ap:.4f})")
    ax.plot(
        recall, precision, color=STYLE["neutral"], lw=2, label=f"SimplifiedEnsemble (AP={ap:.4f})"
    )
    ax.axhline(
        baseline, color=STYLE["grid"], lw=1.5, linestyle="--", label=f"Baseline (={baseline:.2f})"
    )
    ax.fill_between(recall, precision, alpha=0.08, color=STYLE["neutral"])
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.legend(fontsize=10, framealpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "eval_pr_curve.png")
    plt.close(fig)
    logger.info("  ✓ eval_pr_curve.png")


def plot_confusion(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    y_pred = (y_prob >= 0.5).astype(int)
    labels = ["Loser (0)", "Winner (1)"]
    cm = np.array(
        [
            [(y_pred[y_true == 0] == 0).sum(), (y_pred[y_true == 0] == 1).sum()],
            [(y_pred[y_true == 1] == 0).sum(), (y_pred[y_true == 1] == 1).sum()],
        ]
    )
    cm_pct = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = _base_fig("Confusion Matrix", figsize=(6, 5))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(2):
        for j in range(2):
            color = "white" if cm_pct[i, j] > 0.6 else STYLE["primary"]
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}\n({cm_pct[i, j]:.1%})",
                ha="center",
                va="center",
                fontsize=12,
                color=color,
                fontweight="bold",
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted " + l for l in labels], fontsize=9)
    ax.set_yticklabels(["Actual " + l for l in labels], fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "eval_confusion_matrix.png")
    plt.close(fig)
    logger.info("  ✓ eval_confusion_matrix.png")


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> None:
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers, actual_freq, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() > 0:
            bin_centers.append(y_prob[mask].mean())
            actual_freq.append(y_true[mask].mean())
            counts.append(mask.sum())

    bin_centers = np.array(bin_centers)
    actual_freq = np.array(actual_freq)

    fig, ax = _base_fig("Calibration Curve (Reliability Diagram)")
    ax.plot(
        [0, 1], [0, 1], color=STYLE["grid"], lw=1.5, linestyle="--", label="Perfect calibration"
    )
    scatter = ax.scatter(
        bin_centers,
        actual_freq,
        c=counts,
        cmap="Oranges",
        s=80,
        zorder=5,
        edgecolors=STYLE["primary"],
        linewidth=0.5,
    )
    ax.plot(bin_centers, actual_freq, color=STYLE["accent"], lw=2, label="SimplifiedEnsemble")
    plt.colorbar(scatter, ax=ax, label="Sample count", fraction=0.046, pad=0.04)
    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Fraction of Positives", fontsize=11)
    ax.legend(fontsize=10, framealpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "eval_calibration.png")
    plt.close(fig)
    logger.info("  ✓ eval_calibration.png")


def plot_score_distribution(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    fig, ax = _base_fig("Predicted Probability Distribution by Class")
    bins: list[float] = np.linspace(0, 1, 40).tolist()
    ax.hist(
        y_prob[y_true == 1],
        bins=bins,
        alpha=0.6,
        color=STYLE["accent"],
        label=f"Winner (n={int((y_true == 1).sum()):,})",
        density=True,
    )
    ax.hist(
        y_prob[y_true == 0],
        bins=bins,
        alpha=0.6,
        color=STYLE["neutral"],
        label=f"Loser (n={int((y_true == 0).sum()):,})",
        density=True,
    )
    ax.axvline(0.5, color=STYLE["primary"], lw=1.5, linestyle="--", label="Decision boundary")
    ax.set_xlabel("Predicted Win Probability", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=10, framealpha=0.7)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "eval_score_distribution.png")
    plt.close(fig)
    logger.info("  ✓ eval_score_distribution.png")


def plot_feature_importance(top_n: int = 20) -> None:
    importance_path = project_root / "models" / "simplified_feature_importance.json"
    if not importance_path.exists():
        logger.warning("  ⚠ simplified_feature_importance.json not found, skipping")
        return

    with open(importance_path) as f:
        raw: dict = json.load(f)

    total = sum(raw.values())
    items = sorted(raw.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features = [k for k, _ in items]
    values = [v / total * 100 for _, v in items]

    fig, ax = _base_fig(f"Top-{top_n} Feature Importances (LGBM)", figsize=(8, 6))
    colors = [STYLE["accent"] if v >= values[2] else STYLE["neutral"] for v in values]
    bars = ax.barh(
        features[::-1], values[::-1], color=colors[::-1], edgecolor="white", linewidth=0.4
    )
    for bar, val in zip(bars, values[::-1]):
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            fontsize=8,
            color=STYLE["primary"],
        )
    ax.set_xlabel("Relative Importance (%)", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "eval_feature_importance.png")
    plt.close(fig)
    logger.info("  ✓ eval_feature_importance.png")


def main() -> None:
    logger.info("Loading model and test data...")
    model_path = project_root / "models" / "simplified_ensemble.pkl"
    results_path = project_root / "models" / "simplified_results.json"
    if not model_path.exists():
        logger.error("  ✗ Model not found. Run: uv run python scripts/train_simplified.py")
        sys.exit(1)

    model = joblib.load(model_path)
    config = load_config()
    with open(results_path) as f:
        feature_cols: list[str] = json.load(f)["features"]
    X_test, y_test = prepare_simplified_data(config, feature_cols)

    logger.info("Running predictions...")
    y_prob = model.predict_proba_calibrated(X_test)
    y_true = np.asarray(y_test)

    logger.info("Generating plots...")
    plot_roc(y_true, y_prob)
    plot_pr(y_true, y_prob)
    plot_confusion(y_true, y_prob)
    plot_calibration(y_true, y_prob)
    plot_score_distribution(y_true, y_prob)
    plot_feature_importance()

    logger.info(f"\nAll plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
