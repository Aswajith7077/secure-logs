# Secure Logs — BERT-Based Log Anomaly Detection

> A research implementation of a BERT-powered two-stage pipeline for detecting anomalies in distributed system logs (HDFS), combining contrastive pre-training with supervised fine-tuning and FAISS-backed KNN retrieval.

---

## Problem Statement

Large-scale distributed systems like Hadoop generate enormous volumes of log data. Identifying anomalous behaviour — hardware failures, software exceptions, or security incidents — by manually inspecting these logs is practically infeasible. Existing rule-based approaches break down as log formats evolve, and classical ML methods require hand-crafted features that don't generalise well across systems.

The research challenge is to learn **semantic representations of log sequences** that can reliably distinguish normal execution patterns from anomalous ones, without relying on brittle regex rules or domain-specific feature engineering.

---

## Proposed Solution

This project implements the approach from the **LogBERT / LogADT** family of papers, which applies BERT-style transformer models to the log anomaly detection problem through two stages:

1. **Contrastive Pre-training** — The encoder is trained to produce similar embeddings for log sequences belonging to the same execution block and dissimilar embeddings for sequences from different blocks (InfoNCE + Matching loss). No anomaly labels are required for this stage.

2. **Supervised Fine-tuning** — The pre-trained encoder is frozen. A lightweight classifier head is trained on top using ground-truth block-level anomaly labels, learning to output an anomaly probability for each log session.

3. **KNN Retrieval Augmentation** — After fine-tuning, all training session embeddings are indexed in a FAISS vector store. At inference time, a new log session is classified both by the classifier head and by weighted-vote over its k nearest stored neighbours, combining parametric and non-parametric signals.

---

## What We Implemented

| Component | File(s) | Description |
|---|---|---|
| **Config** | `config/config.py` | Centralised hyperparameters and file paths |
| **Data Loading** | `data/dataset.py` | HDFS session grouping by block ID; contrastive pair generation; ground-truth label join from `anomaly_label.csv` |
| **BERT Encoder** | `models/bert_encoder.py` | `LogBERTEncoder` — mean-pooled BERT with optional backbone freezing |
| **Contrastive Model** | `models/contrastive_model.py` | `LogContrastiveModel` — siamese-style head over the shared encoder |
| **Classifier** | `models/classifier.py` | `LogClassifier` — 2-layer MLP head for binary anomaly classification |
| **Pre-training Loop** | `training/pretrain.py` | Epoch loop with joint InfoNCE + Matching loss |
| **Fine-tuning Loop** | `training/finetune.py` | Epoch loop with Binary Cross-Entropy loss |
| **KNN Vector Store** | `retrieval/knn_index.py` | FAISS HNSW index: build, save, load, and weighted-vote query |
| **Pipeline Orchestrator** | `main.py` | End-to-end: data → pretrain → finetune → save models → build FAISS index |

**Saved artefacts** (written to `ai-models/`, excluded from git):

| File | Contents |
|---|---|
| `bert_encoder.pt` | Encoder state dict |
| `log_classifier.pt` | Classifier head state dict |
| `knn.index` | FAISS HNSW vector store |
| `knn_labels.npy` | Parallel label array for KNN |

---

## Prerequisites

- Python 3.9 or higher
- pip
- *(Optional but recommended)* CUDA-capable GPU for full fine-tuning

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd secure-logs
```

### 2. Create and activate a virtual environment

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the HDFS dataset files

Place the following files inside `data/HDFS/`:

| File | Source |
|---|---|
| `2k-structured.csv` | Parsed HDFS log lines (LogPAI format) |
| `anomaly_label.csv` | Block-level ground-truth labels (`BlockId`, `Label`) |

The HDFS_1 dataset is publicly available from the [LogPAI loghub repository](https://github.com/logpai/loghub).

### 5. Create the model output directory

```bash
mkdir ai-models
```

> [!IMPORTANT]
> The pipeline saves all trained artefacts to `ai-models/`. This directory **must exist** before running `main.py`, otherwise the save step will fail.
> It is intentionally excluded from git (via `.gitignore`) because model weights and FAISS indexes are large binary files that should not be committed.
>
> Files written here after a full run:
> | File | Contents |
> |---|---|
> | `bert_encoder.pt` | Trained BERT encoder weights |
> | `log_classifier.pt` | Anomaly classifier head weights |
> | `knn.index` | FAISS HNSW vector store (all session embeddings) |
> | `knn_labels.npy` | Parallel ground-truth label array for KNN voting |

### 6. Verify installation

```bash
python -c "import torch; import transformers; import faiss; print('All dependencies OK')"
```

---

## Running the Pipeline

```bash
python main.py
```

The pipeline will execute in five steps and print progress at each stage:

```
[Data]     Loading pre-training dataset...   (contrastive pairs)
[Data]     Loading fine-tuning dataset...    (labelled sessions)
[Pretrain] Epoch 1/1 — Loss: X.XXXX
[Finetune] Epoch 1/1 — Loss: X.XXXX
✓ Saved encoder    → ai-models/bert_encoder.pt
✓ Saved classifier → ai-models/log_classifier.pt
✓ Saved FAISS index → ai-models/knn.index
```

### Configuration

Edit `config/config.py` to tune the pipeline:

```python
PRETRAIN_PAIRS  = 200     # increase for better pre-training (needs GPU)
PRETRAIN_EPOCHS = 1       # recommended: 3–5 on GPU
FINETUNE_EPOCHS = 1       # recommended: 3–5 on GPU
BATCH_SIZE      = 8       # increase to 32–64 on GPU
```

To enable full BERT fine-tuning (instead of frozen feature extraction), set `freeze_bert=False` in `main.py`.

---

## Project Structure

```
secure-logs/
├── config/
│   └── config.py           # Hyperparameters and paths
├── data/
│   ├── dataset.py          # Dataset classes
│   └── HDFS/               # Raw data files (not committed)
├── models/
│   ├── bert_encoder.py     # LogBERTEncoder
│   ├── contrastive_model.py
│   └── classifier.py       # LogClassifier
├── training/
│   ├── pretrain.py         # Contrastive pre-training loop
│   └── finetune.py         # Supervised fine-tuning loop
├── retrieval/
│   └── knn_index.py        # FAISS KNN vector store
├── utils/
│   └── loss.py             # InfoNCE + Matching + Joint loss
├── main.py                 # Pipeline entry point
├── Understanding.md        # Detailed technical explanation
└── requirements.txt
```

---

## Deactivating the Virtual Environment

```bash
deactivate
```
