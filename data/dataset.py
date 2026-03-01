# data/dataset.py
import re
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


def _extract_block_id(content: str) -> str:
    """Extract block ID from HDFS log content string."""
    match = re.search(r"blk_-?\d+", content)
    return match.group(0) if match else "unknown"


def _build_sessions(df: pd.DataFrame):
    """
    Group log lines by block ID, building a dict:
      { block_id: {"templates": [str, ...], "has_anomaly": bool} }
    A block is anomalous if any of its lines has Level == WARN.
    """
    sessions = {}
    for _, row in df.iterrows():
        block_id = _extract_block_id(str(row["Content"]))
        if block_id not in sessions:
            sessions[block_id] = {"templates": [], "has_anomaly": False}
        sessions[block_id]["templates"].append(str(row["EventTemplate"]))
        if str(row["Level"]).upper() == "WARN":
            sessions[block_id]["has_anomaly"] = True
    return sessions


class HDFSPretrainDataset(Dataset):
    """
    Contrastive pre-training dataset.
    Positive pair  (label=1): two random sub-sequences from the SAME block.
    Negative pair  (label=0): sub-sequences from DIFFERENT blocks.
    We generate `num_pairs` samples total, half positive / half negative.
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer_name: str = "bert-base-uncased",
        max_len: int = 64,
        num_pairs: int = 2000,
    ):
        df = pd.read_csv(csv_path)
        sessions = _build_sessions(df)
        self.block_ids = list(sessions.keys())
        self.templates = {block_id: info["templates"] for block_id, info in sessions.items()}
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

        # Pre-generate pairs at construction time for reproducibility
        self.pairs = []
        n = num_pairs // 2
        # Positive pairs
        self.__pair_positives(n)
        self.__pair_negatives(n,sessions)        

    def __pair_positives(self,n):
        for _ in range(n):
            block_id = random.choice(self.block_ids)
            tmpl = self.templates[block_id]
            seq_a = self._sample_seq(tmpl)
            seq_b = self._sample_seq(tmpl)
            self.pairs.append((seq_a, seq_b, 0))

    def __pair_negatives(self,n,sessions):

        # This is to pair each dissimilar pair with 1
        normal_blocks = [block_id for block_id, info in sessions.items() if not info["has_anomaly"]]
        anomaly_blocks = [block_id for block_id, info in sessions.items() if info["has_anomaly"]]

        for _ in range(n):
            block_id_a = random.choice(normal_blocks)
            block_id_b = random.choice(anomaly_blocks)
            seq_a = self._sample_seq(self.templates[block_id_a])
            seq_b = self._sample_seq(self.templates[block_id_b])
            self.pairs.append((seq_a, seq_b, 1))
        random.shuffle(self.pairs)

    def _sample_seq(self, templates):
        k = max(1, len(templates) // 2)
        return " [SEP] ".join(random.sample(templates, min(k, len(templates))))

    def _tokenize(self, text):
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        seq_a, seq_b, label = self.pairs[idx]
        ids_a, mask_a = self._tokenize(seq_a)
        ids_b, mask_b = self._tokenize(seq_b)
        return (ids_a, mask_a), (ids_b, mask_b), torch.tensor(label, dtype=torch.long)


class HDFSFinetuneDataset(Dataset):
    """
    Supervised fine-tuning dataset using ground-truth labels from anomaly_label.csv.

    Each sample is a full HDFS block represented as a joined event-template string.
    Label source: anomaly_label.csv  (BlockId, Label) where Label is 'Anomaly' or 'Normal'.
      -> 1 = Anomaly,  0 = Normal

    If label_path is None, falls back to the WARN-level heuristic (not recommended).
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer_name: str = "bert-base-uncased",
        max_len: int = 64,
        label_path: str = None,
    ):
        df = pd.read_csv(csv_path)
        sessions = _build_sessions(df)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

        label_map,use_heuristic = self.__handle_label(label_path)

        # ── Build samples ──────────────────────────────────────────
        self.samples = []
        self.__build_samples(sessions,label_map,use_heuristic)

        # Label distribution summary
        n_anomaly = sum(1 for _, lbl in self.samples if lbl == 1)
        n_normal = len(self.samples) - n_anomaly
        print(
            f"[Finetune] Label distribution — Normal: {n_normal}, Anomaly: {n_anomaly}"
        )

    def __handle_label(self,label_path):

        if label_path is not None:
            label_df = pd.read_csv(label_path)
            # Normalise column names defensively
            label_df.columns = [c.strip() for c in label_df.columns]
            label_map = {
                row["BlockId"]: 1 if str(row["Label"]).strip() == "Anomaly" else 0
                for _, row in label_df.iterrows()
            }
            use_heuristic = False
        else:
            # Fallback: derive label from WARN lines already stored in sessions
            label_map = {}
            use_heuristic = True
            print(
                "[Warning] No label_path provided — using WARN-level heuristic for labels."
            )

        return label_map,use_heuristic

    def __build_samples(self,sessions,label_map,use_heuristic):
        skipped = 0
        for block_id, info in sessions.items():
            text = " [SEP] ".join(info["templates"])
            if use_heuristic:
                label = int(info["has_anomaly"])
            else:
                if block_id not in label_map:
                    skipped += 1
                    continue  # drop sessions with no ground-truth label
                label = label_map[block_id]
            self.samples.append((text, label))

        if skipped:
            print(
                f"[Warning] {skipped} sessions had no matching label in anomaly_label.csv and were skipped."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return (
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            torch.tensor(label, dtype=torch.float),
        )
