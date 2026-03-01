# utils/metrics.py
"""
Paper-aligned evaluation metrics for LogSentry anomaly detection.

Computes:
  - Accuracy, Precision, Recall, F1-score (binary, anomaly-positive)
  - Matthews Correlation Coefficient (MCC)
  - ROC-AUC and PR-AUC
  - False Positive Rate (FPR) and False Negative Rate (FNR)
  - Novel-log category statistics (sessions beyond KNN distance threshold)
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[float],
    novel_flags: Sequence[bool],
) -> dict:
    """
    Compute all evaluation metrics.

    Args:
        y_true      : Ground-truth binary labels  (0=Normal, 1=Anomaly)
        y_pred      : Predicted binary labels     (0/1)
        y_score     : Hybrid detector final scores (float in [0,1])
        novel_flags : Boolean per-sample — True if KNN distance > threshold

    Returns:
        Dictionary with all metric values and curve data for plotting.
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    y_score = np.array(y_score, dtype=float)
    novel_flags = np.array(novel_flags, dtype=bool)

    # ── Core classification metrics ────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    # ── Confusion matrix ───────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # ── Curve-based metrics ────────────────────────────────────────
    # Safely handle edge case where all labels are the same class
    n_classes = len(np.unique(y_true))
    if n_classes > 1:
        roc_auc = roc_auc_score(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)
        fpr_curve, tpr_curve, roc_thresholds = roc_curve(y_true, y_score)
        prec_curve, rec_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")
        fpr_curve = tpr_curve = roc_thresholds = np.array([])
        prec_curve = rec_curve = pr_thresholds = np.array([])

    # ── Novel-log statistics ───────────────────────────────────────
    n_novel = int(novel_flags.sum())
    n_known = int((~novel_flags).sum())
    novel_anomaly = int((novel_flags & (y_true == 1)).sum())
    novel_normal = int((novel_flags & (y_true == 0)).sum())
    known_anomaly = int((~novel_flags & (y_true == 1)).sum())
    known_normal = int((~novel_flags & (y_true == 0)).sum())

    return {
        # Scalar metrics
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "fpr": fpr,
        "fnr": fnr,
        # Confusion matrix
        "confusion_matrix": cm,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        # Curve data (for plots)
        "fpr_curve": fpr_curve,
        "tpr_curve": tpr_curve,
        "prec_curve": prec_curve,
        "rec_curve": rec_curve,
        # Novel-log stats
        "n_novel": n_novel,
        "n_known": n_known,
        "novel_anomaly": novel_anomaly,
        "novel_normal": novel_normal,
        "known_anomaly": known_anomaly,
        "known_normal": known_normal,
        # Raw arrays for distribution plots
        "y_true": y_true,
        "y_score": y_score,
        "novel_flags": novel_flags,
    }


def format_metrics_report(metrics: dict, model_info: dict | None = None) -> str:
    """Format a human-readable metrics report for writing to metrics.txt."""

    def _fmt(v):
        if math.isnan(v):
            return "N/A (single-class dataset)"
        return f"{v:.4f}"

    lines = [
        "=" * 60,
        "  LogSentry — Anomaly Detection Evaluation Report",
        "=" * 60,
        "",
    ]

    if model_info:
        lines += [
            "📁 Model Artefacts",
            f"  Encoder    : {model_info.get('encoder_path', '?')}",
            f"  Classifier : {model_info.get('classifier_path', '?')}",
            f"  KNN index  : {model_info.get('knn_path', '?')}",
            f"  Beta (β)   : {model_info.get('beta', 0.68)}",
            f"  KNN-k      : {model_info.get('k', 10)}",
            f"  Novel thresh: {model_info.get('novel_thresh', '?')}",
            "",
        ]

    lines += [
        "📊 Dataset",
        f"  Total sessions : {metrics['tp'] + metrics['tn'] + metrics['fp'] + metrics['fn']}",
        f"  Anomalies (GT) : {metrics['tp'] + metrics['fn']}",
        f"  Normals   (GT) : {metrics['tn'] + metrics['fp']}",
        "",
        "📈 Classification Metrics",
        f"  Accuracy   : {_fmt(metrics['accuracy'])}",
        f"  Precision  : {_fmt(metrics['precision'])}",
        f"  Recall     : {_fmt(metrics['recall'])}",
        f"  F1-Score   : {_fmt(metrics['f1'])}",
        f"  MCC        : {_fmt(metrics['mcc'])}",
        f"  FPR        : {_fmt(metrics['fpr'])}",
        f"  FNR        : {_fmt(metrics['fnr'])}",
        "",
        "📉 Curve-based Metrics",
        f"  ROC-AUC    : {_fmt(metrics['roc_auc'])}",
        f"  PR-AUC     : {_fmt(metrics['pr_auc'])}",
        "",
        "🔲 Confusion Matrix  (rows=GT, cols=Pred)",
        "             Pred=Normal  Pred=Anomaly",
        f"  GT=Normal  {metrics['tn']:>11d}  {metrics['fp']:>12d}",
        f"  GT=Anomaly {metrics['fn']:>11d}  {metrics['tp']:>12d}",
        "",
        "🆕 KNN-RAG: Novel Log Category Detection",
        f"  Novel sessions : {metrics['n_novel']}",
        f"    ↳ Anomaly    : {metrics['novel_anomaly']}",
        f"    ↳ Normal     : {metrics['novel_normal']}",
        f"  Known sessions : {metrics['n_known']}",
        f"    ↳ Anomaly    : {metrics['known_anomaly']}",
        f"    ↳ Normal     : {metrics['known_normal']}",
        "",
        "=" * 60,
    ]

    return "\n".join(lines)
