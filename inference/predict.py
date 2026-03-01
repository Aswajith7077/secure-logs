# inference/predict.py
"""
LogSentry — Inference, KNN-RAG novel detection, metrics & visualisations.

Usage (from the project root):
    python inference/predict.py

Loads trained artefacts from ai-models/, runs the hybrid LogSentry detector
over the full HDFSFinetuneDataset, and writes:
    result/metrics.txt
    result/visualizations/*.png
"""

from __future__ import annotations

import os
import sys

# Allow running from project root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import config_service as cfg
from data.dataset import HDFSFinetuneDataset
from models.bert_encoder import LogBERTEncoder
from models.classifier import LogClassifier
from retrieval.knn_index import KNNRetriever
from utils.metrics import compute_metrics, format_metrics_report
from utils.visualizations import generate_all
from services.logger import get_logger
from utils.optimal_threshold import optimal_threshold

log = get_logger(__name__)

# ── Tunable constants ──────────────────────────────────────────────
BETA = 0.68  # weight of BERT classifier vs KNN in hybrid score
KNN_K = 10  # neighbours to retrieve
# Novel-log detection: threshold is computed adaptively during a calibration
# pass as  mean(NN-distance) + NOVEL_SIGMA × std(NN-distance).
# Sessions whose nearest-neighbour distance exceeds this are flagged as
# belonging to a previously unseen log category.
NOVEL_SIGMA = 2.0  # how many std-devs above mean → "novel"
NOVEL_THRESH = 1.0  # hard-coded fallback when KNN index is empty

VIZ_DIR = os.path.join(cfg.PREDICT_DIR, "visualizations")
METRICS_FILE = os.path.join(cfg.PREDICT_DIR, "metrics.txt")


# ── Model loading ──────────────────────────────────────────────────


def load_models(device: str):
    """Load encoder, classifier and KNN retriever from ai-models/."""
    save_dir = cfg.SAVE_DIR  # "ai-models"

    encoder_path = os.path.join(save_dir, "bert_encoder.pt")
    classifier_path = os.path.join(save_dir, "log_classifier.pt")
    knn_path = save_dir  # directory that contains knn.index + knn_labels.npy

    log.info("[Load] Encoder -> %s", encoder_path)
    encoder = LogBERTEncoder(model_name=cfg.MODEL_NAME, freeze_bert=True).to(device)
    encoder.load_state_dict(
        torch.load(encoder_path, map_location=device, weights_only=True)
    )
    encoder.eval()

    log.info("[Load] Classifier -> %s", classifier_path)
    classifier = LogClassifier(encoder=encoder).to(device)
    classifier.load_state_dict(
        torch.load(classifier_path, map_location=device, weights_only=True)
    )
    classifier.eval()

    log.info("[Load] KNN index -> %s/knn.index", knn_path)
    retriever = KNNRetriever.load(knn_path, dim=768, k=KNN_K)

    return encoder, classifier, retriever


# ── Adaptive novel-log threshold calibration ───────────────────────


def calibrate_novel_threshold(
    classifier: LogClassifier,
    retriever: KNNRetriever,
    dataloader: DataLoader,
    device: str,
) -> float:
    """
    Compute an adaptive novel-log threshold as:
        mean(nearest-neighbour distance) + NOVEL_SIGMA × std(NN distance)
    by doing a single pass over the dataloader.
    """
    log.info("[Calibrate] Computing KNN distance distribution for novel threshold")
    all_min_dists: list[float] = []

    with torch.no_grad():
        for input_ids, attention_mask, _ in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            for i in range(input_ids.size(0)):
                emb = classifier.encoder(
                    input_ids[i].unsqueeze(0), attention_mask[i].unsqueeze(0)
                )
                emb_np = emb.cpu().numpy().astype(np.float32)
                distances, _ = retriever.index.search(emb_np, 1)
                all_min_dists.append(float(distances[0][0]))

    dists = np.array(all_min_dists)
    threshold = float(dists.mean() + NOVEL_SIGMA * dists.std())
    log.info(
        "[Calibrate] NN-dist  mean=%.4f  std=%.4f  ->  novel_thresh=%.4f  (mean + %.1fσ)",
        dists.mean(),
        dists.std(),
        threshold,
        NOVEL_SIGMA,
    )
    return threshold


# ── Inference ──────────────────────────────────────────────────────


def run_inference(
    classifier: LogClassifier,
    retriever: KNNRetriever,
    dataloader: DataLoader,
    device: str,
    novel_thresh: float,
):
    """
    Run the hybrid LogSentry detector over every session in `dataloader`.

    Returns:
        y_true       — ground-truth labels (list[int])
        y_pred       — predicted labels    (list[int])
        y_score      — hybrid final scores (list[float])
        prob_model   — BERT classifier probabilities (np.ndarray)
        prob_knn     — KNN vote probabilities        (np.ndarray)
        novel_flags  — True if session is novel  (list[bool])
    """
    classifier.eval()

    y_true, y_pred, y_score = [], [], []
    prob_model_all, prob_knn_all = [], []
    novel_flags: list[bool] = []

    log.info("[Predict] Running inference on %d sessions …", len(dataloader.dataset))

    # threshold = dataloader.dataset.ANOMALY_COUNT / (dataloader.dataset.ANOMALY_COUNT + dataloader.dataset.NORMAL_COUNT)
    # log.info("[Predict] Using threshold: %.4f", threshold)

    with torch.no_grad():
        pbar = tqdm(
            dataloader,
            desc="[Predict] Inference",
            unit="batch",
            total=len(dataloader),
        )
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(pbar):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            for i in range(input_ids.size(0)):
                ids_i = input_ids[i].unsqueeze(0)
                mask_i = attention_mask[i].unsqueeze(0)

                # ── BERT classifier score ──────────────────────────
                logit = classifier(ids_i, mask_i)
                p_model = float(torch.sigmoid(logit).item())

                # ── KNN score + novel detection ────────────────────
                emb = classifier.encoder(ids_i, mask_i)
                emb_np = emb.cpu().numpy().astype(np.float32)

                distances, indices = retriever.index.search(emb_np, KNN_K)
                neighbour_labels = [retriever.labels[j] for j in indices[0] if j >= 0]
                weights = np.exp(-distances[0][: len(neighbour_labels)])
                p_knn = (
                    float(np.average(neighbour_labels, weights=weights))
                    if weights.sum() > 0
                    else 0.0
                )

                min_dist = float(distances[0][0]) if len(distances[0]) > 0 else 0.0
                is_novel = min_dist > novel_thresh

                # ── Hybrid decision ────────────────────────────────
                final_score = BETA * p_model + (1 - BETA) * p_knn
                # threshold = optimal_threshold(y_true, y_score)

                # Arrived using PR-curve optimal threshold
                predicted = int(final_score > 0.475853842187741)

                gt = int(labels[i].item())
                y_true.append(gt)
                y_pred.append(predicted)
                y_score.append(final_score)
                prob_model_all.append(p_model)
                prob_knn_all.append(p_knn)
                novel_flags.append(is_novel)

            if (batch_idx + 1) % 50 == 0:
                log.debug(
                    "[Predict] Processed %d / %d sessions",
                    (batch_idx + 1) * dataloader.batch_size,
                    len(dataloader.dataset),
                )

    return (
        y_true,
        y_pred,
        y_score,
        np.array(prob_model_all, dtype=np.float32),
        np.array(prob_knn_all, dtype=np.float32),
        novel_flags,
    )


def main():
    log.info("=" * 60)
    log.info("  LogSentry — Prediction & Evaluation")
    log.info("=" * 60)
    log.info("Device : %s", cfg.DEVICE)

    # ── Load data ──────────────────────────────────────────────────
    log.info("[Data] Loading HDFSFinetuneDataset …")
    dataset = HDFSFinetuneDataset(
        csv_path=cfg.DATA_PATH,
        tokenizer_name=cfg.MODEL_NAME,
        max_len=cfg.MAX_LEN,
        label_path=cfg.LABEL_PATH,
    )
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    log.info("[Data] %d sessions loaded.", len(dataset))

    # ── Load models ────────────────────────────────────────────────
    encoder, classifier, retriever = load_models(cfg.DEVICE)  # noqa: F841

    # ── Calibrate novel-log threshold (adaptive, data-driven) ──────
    novel_thresh = calibrate_novel_threshold(classifier, retriever, loader, cfg.DEVICE)

    model_info = {
        "encoder_path": os.path.join(cfg.SAVE_DIR, "bert_encoder.pt"),
        "classifier_path": os.path.join(cfg.SAVE_DIR, "log_classifier.pt"),
        "knn_path": cfg.SAVE_DIR,
        "beta": BETA,
        "k": KNN_K,
        "novel_thresh": round(novel_thresh, 4),
        "novel_sigma": NOVEL_SIGMA,
    }

    # ── Run inference ──────────────────────────────────────────────
    y_true, y_pred, y_score, prob_model, prob_knn, novel_flags = run_inference(
        classifier, retriever, loader, cfg.DEVICE, novel_thresh=novel_thresh
    )

    # ── Calibrate decision threshold via PR curve (post-inference) ──
    # optimal_threshold needs the FULL score list — must be called here,
    # not inside the per-sample loop where y_true/y_score are still empty.
    best_threshold = optimal_threshold(y_true, y_score)
    log.info("[Threshold] PR-curve optimal threshold → %.4f", best_threshold)
    model_info["decision_threshold"] = round(float(best_threshold), 4)
    y_pred = [int(s >= best_threshold) for s in y_score]

    # ── Compute metrics ────────────────────────────────────────────
    log.info("[Metrics] Computing evaluation metrics …")
    metrics = compute_metrics(y_true, y_pred, y_score, novel_flags)
    report = format_metrics_report(metrics, model_info)

    log.info("\n%s", report)

    # ── Save metrics.txt ───────────────────────────────────────────
    os.makedirs(cfg.PREDICT_DIR, exist_ok=True)
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    log.info("Metrics saved → %s", METRICS_FILE)

    # ── Generate visualizations ────────────────────────────────────
    log.info("[Visualizations] Generating plots → %s", VIZ_DIR)
    generate_all(metrics, VIZ_DIR, prob_model=prob_model, prob_knn=prob_knn)

    # ── Summary ────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("  Evaluation complete.")
    log.info("  Metrics  : %s", METRICS_FILE)
    log.info("  Plots    : %s/", VIZ_DIR)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
