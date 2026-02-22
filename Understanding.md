# Understanding the Secure-Logs BERT Pipeline

## How anomaly detection works (one-line summary)

> Each block of HDFS log lines is converted into a single text sequence, encoded by BERT into a 768-dimensional embedding, and then classified as **normal (0)** or **anomalous (1)** by a small feed-forward head.

---

## File-by-file Changes

### `config/config.py` — *new*
Central place for all hyperparameters.

| Setting | Value | Purpose |
|---|---|---|
| `MODEL_NAME` | `bert-base-uncased` | Which BERT checkpoint to load |
| `MAX_LEN` | 64 | Token budget per sequence |
| `BATCH_SIZE` | 8 | Samples per gradient step |
| `PRETRAIN_EPOCHS` | 1 | Contrastive pre-training passes |
| `FINETUNE_EPOCHS` | 1 | Supervised fine-tuning passes |
| `LR` | 2e-5 | AdamW learning rate |
| `PRETRAIN_PAIRS` | 200 | Number of contrastive pairs (CPU budget) |
| `DEVICE` | auto | `cuda` if GPU available, else `cpu` |

---

### `data/dataset.py` — *new*

#### Session building (`_build_sessions`)
The HDFS dataset has one log line per row. Lines are **grouped by block ID** (`blk_XXXXXXX`) extracted from the `Content` column with a regex. Each group is called a *session*.

```
Session = {
    "templates": ["PacketResponder <*> terminating", "Received block blk_<*> ...", ...],
    "has_anomaly": True / False
}
```

#### `HDFSPretrainDataset` — contrastive pairs
Used during **pre-training** only.

| Item | Detail |
|---|---|
| **Positive pair (label=1)** | Two random sub-sequences sampled from the **same** block |
| **Negative pair (label=0)** | Sub-sequences from **two different** blocks |
| **Feature** | Tokenized event-template strings joined with `[SEP]` |
| **Label** | `1` = same block, `0` = different blocks |

Goal: teach BERT to produce *similar* embeddings for log lines from the same block and *dissimilar* ones for different blocks.

#### `HDFSFinetuneDataset` — anomaly classification
Used during **fine-tuning** and ultimately for inference.

| Item | Detail |
|---|---|
| **Feature (X)** | All event templates of a block joined: `"PacketResponder <*> terminating [SEP] Received block blk_<*> ..."` |
| **Label (y)** | `1` if **any** log line in the block has `Level == WARN`, else `0` |
| **Anomaly signal** | WARN-level events correspond to exceptions (e.g. E3: *"Got exception while serving block"*) |

---

### `models/bert_encoder.py` — *modified*

**Added: `freeze_bert=True`**

```python
if freeze_bert:
    for param in self.bert.parameters():
        param.requires_grad = False
```

BERT's 110M parameters are frozen. It acts as a **feature extractor** — it converts token sequences into rich contextual embeddings without updating its own weights. Only the small task heads on top are trained.

- Frozen BERT still runs on the **forward pass** (produces embeddings).
- No gradients flow *through* BERT → training is fast on CPU.
- Set `freeze_bert=False` on a GPU to do full end-to-end fine-tuning.

---

### `models/contrastive_model.py` — *unchanged*
Used during pre-training. Takes two sequences, encodes each with the shared encoder, and:
1. Computes `diff = |emb_a − emb_b|`
2. Concatenates `[emb_a; diff]` → shape `[B, 1536]`
3. Linear head outputs 2-class logits (same / different block)

---

### `models/classifier.py` — *unchanged*
Used during fine-tuning. Takes one sequence, encodes it, then:
```
embedding [B, 768]
    → Linear(768, 768) → ReLU
    → Linear(768, 1)
    → scalar logit
```
Output is a single logit per session. `sigmoid(logit) > 0.5` → anomaly.

---

### `training/pretrain.py` — *modified*
Added an epoch loop and per-epoch loss reporting.

**Loss function: `joint_loss`**
```
joint_loss = matching_loss + α × info_nce_loss
```
- `matching_loss`: cross-entropy on same/different classification.
- `info_nce_loss`: contrastive loss that pulls same-block embeddings together in embedding space.
- `α = 1.0` by default.

---

### `training/finetune.py` — *modified*
Added an epoch loop and per-epoch loss reporting.

**Loss function: `BCEWithLogitsLoss`**
```python
F.binary_cross_entropy_with_logits(logits, labels.float())
```
Binary cross-entropy with numerical stability. Labels are 0 or 1.

---

### `main.py` — *complete rewrite*

Full pipeline in order:

```
1. Load HDFSPretrainDataset  (contrastive pairs)
2. Load HDFSFinetuneDataset  (session → anomaly label)
3. Build LogBERTEncoder      (BERT frozen)
4. Wrap in LogContrastiveModel
5. Run pretrain()            → teaches BERT-head to distinguish blocks
6. Wrap encoder in LogClassifier
7. Run finetune()            → teaches classifier to flag WARN sessions
8. Save encoder  → ai-models/bert_encoder.pt
9. Save classifier → ai-models/log_classifier.pt
```

Only **trainable** (non-frozen) parameters are passed to each optimizer:
```python
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(trainable, lr=cfg.LR)
```

---

## Supervised Fine-tuning: Features & Labels

| | Description |
|---|---|
| **Input (X)** | All event templates of an HDFS block, concatenated with `[SEP]` tokens, then tokenized to `MAX_LEN=64` tokens |
| **Encoding** | BERT mean-pools the contextual token embeddings → one vector of size 768 per session |
| **Output (ŷ)** | Single scalar logit → `sigmoid(ŷ) → probability of anomaly` |
| **Label (y)** | `1` if any line in the block is `Level=WARN`, else `0` |
| **Decision rule** | `sigmoid(logit) > 0.5` → **Anomalous**, else **Normal** |

### Why WARN = anomaly?
HDFS WARN-level events correspond to exceptions during block serving (EventId E3: *"Got exception while serving blk_<\*> to /<\*>"*). These are the standard ground-truth anomaly markers used in the HDFS log anomaly detection literature.

---

## Running Inference on a New Log Block

```python
import torch
from transformers import BertTokenizer
from models.bert_encoder import LogBERTEncoder
from models.classifier import LogClassifier

encoder = LogBERTEncoder(freeze_bert=False)
encoder.load_state_dict(torch.load("ai-models/bert_encoder.pt"))

classifier = LogClassifier(encoder=encoder)
classifier.load_state_dict(torch.load("ai-models/log_classifier.pt"))
classifier.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
templates = ["PacketResponder <*> terminating", "Received block blk_<*> of size <*>"]
text = " [SEP] ".join(templates)

enc = tokenizer(text, return_tensors="pt", max_length=64, padding="max_length", truncation=True)
with torch.no_grad():
    logit = classifier(enc["input_ids"], enc["attention_mask"])
    prob = torch.sigmoid(logit).item()

print("Anomaly" if prob > 0.5 else "Normal", f"(confidence: {prob:.2%})")
```
