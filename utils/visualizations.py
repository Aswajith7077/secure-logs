# utils/visualizations.py
"""
Generate and save all evaluation visualizations for LogSentry.

Output files (all PNG, saved to `save_dir`):
  roc_curve.png          — ROC curve with AUC annotation
  pr_curve.png           — Precision-Recall curve with AP annotation
  confusion_matrix.png   — Colour-mapped confusion matrix with counts
  score_distribution.png — KDE of hybrid scores for Normal vs Anomaly
  knn_vs_model_scores.png — Scatter: KNN score vs Model score, coloured by label
  novel_log_bar.png      — Stacked bar: novel vs known sessions by true label
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless rendering — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap


# ── Colour palette ─────────────────────────────────────────────────
NORMAL_COLOR = "#4CAF50"  # green
ANOMALY_COLOR = "#F44336"  # red
NOVEL_COLOR = "#FF9800"  # orange
KNOWN_COLOR = "#2196F3"  # blue
BG_COLOR = "#0F1117"  # dark background
GRID_COLOR = "#2A2D35"
TEXT_COLOR = "#E0E0E0"

STYLE = {
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": "#1A1D27",
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.titlecolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "grid.color": GRID_COLOR,
    "text.color": TEXT_COLOR,
    "legend.facecolor": "#1A1D27",
    "legend.edgecolor": GRID_COLOR,
    "legend.labelcolor": TEXT_COLOR,
}


def _apply_style():
    plt.rcParams.update(STYLE)
    plt.rcParams["font.family"] = "DejaVu Sans"


def _save(fig: plt.Figure, path: str):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\tSaved to the path -> {path}")


# ── 1. ROC Curve ──────────────────────────────────────────────────


def plot_roc_curve(metrics: dict, save_dir: str):
    _apply_style()
    fpr = metrics["fpr_curve"]
    tpr = metrics["tpr_curve"]
    auc = metrics["roc_auc"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        fpr, tpr, color="#7C4DFF", linewidth=2.5, label=f"LogSentry (AUC = {auc:.4f})"
    )
    ax.plot(
        [0, 1],
        [0, 1],
        color=GRID_COLOR,
        linestyle="--",
        linewidth=1.2,
        label="Random classifier",
    )
    ax.fill_between(fpr, tpr, alpha=0.15, color="#7C4DFF")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve — LogSentry Anomaly Detection", fontsize=13, pad=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linewidth=0.5)

    _save(fig, os.path.join(save_dir, "roc_curve.png"))


# ── 2. Precision-Recall Curve ─────────────────────────────────────


def plot_pr_curve(metrics: dict, save_dir: str):
    _apply_style()
    prec = metrics["prec_curve"]
    rec = metrics["rec_curve"]
    ap = metrics["pr_auc"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.step(
        rec,
        prec,
        color="#00BCD4",
        linewidth=2.5,
        where="post",
        label=f"LogSentry (AP = {ap:.4f})",
    )
    ax.fill_between(rec, prec, alpha=0.15, color="#00BCD4", step="post")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curve", fontsize=13, pad=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, linewidth=0.5)

    _save(fig, os.path.join(save_dir, "pr_curve.png"))


# ── 3. Confusion Matrix ────────────────────────────────────────────


def plot_confusion_matrix(metrics: dict, save_dir: str):
    _apply_style()
    cm = metrics["confusion_matrix"]
    labels = ["Normal (0)", "Anomaly (1)"]

    cmap = LinearSegmentedColormap.from_list("logsentry", ["#1A1D27", "#7C4DFF"], N=256)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=13, pad=14)

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color="white" if cm[i, j] < thresh else "#0F1117",
            )

    _save(fig, os.path.join(save_dir, "confusion_matrix.png"))


# ── 4. Score Distribution ─────────────────────────────────────────


def plot_score_distribution(
    metrics: dict,
    save_dir: str,
    prob_model: np.ndarray = None,
    prob_knn: np.ndarray = None,
):
    _apply_style()
    y_true = metrics["y_true"]
    y_score = metrics["y_score"]

    fig, ax = plt.subplots(figsize=(7, 5))

    for label, color, name in [
        (0, NORMAL_COLOR, "Normal"),
        (1, ANOMALY_COLOR, "Anomaly"),
    ]:
        mask = y_true == label
        scores = y_score[mask]
        if scores.size == 0:
            continue
        ax.hist(
            scores,
            bins=40,
            density=True,
            alpha=0.55,
            color=color,
            edgecolor="none",
            label=f"{name} (n={mask.sum()})",
        )
        # Smooth KDE overlay
        from scipy.stats import gaussian_kde

        if scores.std() > 1e-6:
            xs = np.linspace(0, 1, 300)
            kde = gaussian_kde(scores, bw_method=0.15)
            ax.plot(xs, kde(xs), color=color, linewidth=2)

    ax.axvline(
        0.5,
        color="white",
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        label="Decision threshold (0.5)",
    )
    ax.set_xlabel("Hybrid Score  (β·p_model + (1-β)·p_knn)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Anomaly Score Distribution", fontsize=13, pad=14)
    ax.legend(fontsize=10)
    ax.grid(True, linewidth=0.5)

    _save(fig, os.path.join(save_dir, "score_distribution.png"))


# ── 5. KNN vs Model Score Scatter ─────────────────────────────────


def plot_knn_vs_model_scores(
    y_true: np.ndarray, prob_model: np.ndarray, prob_knn: np.ndarray, save_dir: str
):
    _apply_style()
    y_true = np.array(y_true, dtype=int)
    prob_model = np.array(prob_model)
    prob_knn = np.array(prob_knn)

    fig, ax = plt.subplots(figsize=(6, 5))

    for label, color, name in [
        (0, NORMAL_COLOR, "Normal"),
        (1, ANOMALY_COLOR, "Anomaly"),
    ]:
        mask = y_true == label
        ax.scatter(
            prob_knn[mask],
            prob_model[mask],
            c=color,
            alpha=0.55,
            s=18,
            label=f"{name} (n={mask.sum()})",
            edgecolors="none",
        )

    ax.axhline(0.5, color="white", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(0.5, color="white", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("KNN Score  (p_knn)", fontsize=11)
    ax.set_ylabel("Model Score  (p_model)", fontsize=11)
    ax.set_title("KNN vs Model Score per Session", fontsize=13, pad=14)
    ax.legend(fontsize=10)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, linewidth=0.5)

    _save(fig, os.path.join(save_dir, "knn_vs_model_scores.png"))


# ── 6. Novel-log Bar Chart ────────────────────────────────────────


# def plot_novel_log_bar(metrics: dict, save_dir: str):
#     _apply_style()

#     categories = ["Known Sessions", "Novel Sessions"]
#     anomaly_counts = [metrics["known_anomaly"], metrics["novel_anomaly"]]
#     normal_counts = [metrics["known_normal"], metrics["novel_normal"]]

#     x = np.arange(len(categories))
#     width = 0.45

#     fig, ax = plt.subplots(figsize=(6, 5))
#     bars_n = ax.bar(
#         x,
#         normal_counts,
#         width,
#         label="Normal",
#         color=NORMAL_COLOR,
#         alpha=0.85,
#         edgecolor="none",
#     )
#     bars_a = ax.bar(
#         x,
#         anomaly_counts,
#         width,
#         bottom=normal_counts,
#         label="Anomaly",
#         color=ANOMALY_COLOR,
#         alpha=0.85,
#         edgecolor="none",
#     )

#     ax.set_xticks(x)
#     ax.set_xticklabels(categories, fontsize=11)
#     ax.set_ylabel("Session Count", fontsize=11)
#     ax.set_title(
#         "Novel vs Known Log Sessions\n(KNN-RAG Detection)", fontsize=13, pad=14
#     )
#     ax.legend(fontsize=10)
#     ax.grid(True, axis="y", linewidth=0.5)

#     for bar, val in zip(list(bars_n) + list(bars_a), normal_counts + anomaly_counts):
#         if val > 0:
#             ax.text(
#                 bar.get_x() + bar.get_width() / 2,
#                 bar.get_y() + bar.get_height() / 2,
#                 str(val),
#                 ha="center",
#                 va="center",
#                 fontsize=11,
#                 fontweight="bold",
#                 color="white",
#             )

#     _save(fig, os.path.join(save_dir, "novel_log_bar.png"))


# ── Master generate function ──────────────────────────────────────


def generate_all(
    metrics: dict,
    save_dir: str,
    prob_model: np.ndarray = None,
    prob_knn: np.ndarray = None,
):
    """
    Generate and save all 6 visualization files to `save_dir`.

    Args:
        metrics   : dict from utils.metrics.compute_metrics()
        save_dir  : directory to write PNG files (created if missing)
        prob_model: per-sample BERT classifier probabilities [N]
        prob_knn  : per-sample KNN vote probabilities       [N]
    """
    os.makedirs(save_dir, exist_ok=True)

    import math

    auc_ok = not math.isnan(metrics.get("roc_auc", float("nan")))

    if auc_ok:
        plot_roc_curve(metrics, save_dir)
        plot_pr_curve(metrics, save_dir)
    else:
        print("\tSkipping ROC/PR curves — only one class present in labels.")

    plot_confusion_matrix(metrics, save_dir)
    plot_score_distribution(metrics, save_dir, prob_model, prob_knn)

    if prob_model is not None and prob_knn is not None:
        plot_knn_vs_model_scores(metrics["y_true"], prob_model, prob_knn, save_dir)

    # plot_novel_log_bar(metrics, save_dir)
