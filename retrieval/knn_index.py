# retrieval/knn_index.py
import os
import faiss
import numpy as np
import torch


class KNNRetriever:
    """
    FAISS-backed vector store for KNN anomaly retrieval.

    Flow:
      1. After fine-tuning, call build_index() to encode every training
         session with the frozen BERT encoder and add the resulting
         768-dim vectors (+ their labels) into this index.
      2. Save the index to disk with save().
      3. At inference time, load() the index, encode a new log session,
         call query() to find k nearest neighbours, and take a weighted
         vote on their labels → anomaly probability.

    Vector store: FAISS IndexHNSWFlat (in-memory approximate nearest-
    neighbour index; persisted to disk as a single binary file).
    """

    def __init__(self, dim: int = 768, k: int = 10):
        self.dim = dim
        self.k = k
        # HNSW: fast approximate NN with good recall
        self.index = faiss.IndexHNSWFlat(dim, 32)
        self.labels: list[int] = []  # parallel list of 0/1 labels

    # ── Populate ──────────────────────────────────────────────────

    def add(self, embeddings: np.ndarray, labels: list[int]):
        """Add a batch of embeddings and their ground-truth labels."""
        self.index.add(embeddings.astype(np.float32))
        self.labels.extend(labels)

    # ── Persist ───────────────────────────────────────────────────

    def save(self, directory: str):
        """Save FAISS index + labels to `directory`."""
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "knn.index"))
        np.save(os.path.join(directory, "knn_labels.npy"), np.array(self.labels))
        print(
            f"✓ Saved FAISS index  → {directory}/knn.index  ({len(self.labels)} vectors)"
        )

    @classmethod
    def load(cls, directory: str, dim: int = 768, k: int = 10) -> "KNNRetriever":
        """Load a previously saved index from `directory`."""
        retriever = cls(dim=dim, k=k)
        retriever.index = faiss.read_index(os.path.join(directory, "knn.index"))
        retriever.labels = np.load(os.path.join(directory, "knn_labels.npy")).tolist()
        print(f"Loaded FAISS index from {directory}  ({len(retriever.labels)} vectors)")
        return retriever

    # ── Query ─────────────────────────────────────────────────────

    def query(self, embedding: np.ndarray) -> float:
        """
        Given a single query embedding (shape [dim] or [1, dim]),
        return a weighted-vote anomaly score in [0, 1].

        Weighting: exp(-distance) so closer neighbours count more.
        Score > 0.5  →  predicted Anomaly.
        """
        vec = embedding.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(vec, self.k)
        neighbour_labels = [self.labels[i] for i in indices[0] if i >= 0]
        weights = np.exp(-distances[0][: len(neighbour_labels)])
        if weights.sum() == 0:
            return 0.0
        return float(np.average(neighbour_labels, weights=weights))


# ── Index builder ─────────────────────────────────────────────────


def build_index(
    encoder, dataloader, device: str, save_dir: str, k: int = 10
) -> KNNRetriever:
    """
    Encode every session in `dataloader` with `encoder`, collect the
    resulting 768-dim vectors and their labels, build a FAISS index,
    and save it to `save_dir`.

    Args:
        encoder    : trained LogBERTEncoder (weights already loaded)
        dataloader : HDFSFinetuneDataset DataLoader
                     yields (input_ids, attention_mask, label)
        device     : 'cpu' or 'cuda'
        save_dir   : directory to write knn.index + knn_labels.npy
        k          : number of neighbours to use at query time

    Returns:
        Populated KNNRetriever ready for querying.
    """
    encoder.eval()
    retriever = KNNRetriever(dim=768, k=k)

    all_embeddings = []
    all_labels = []

    print("[KNN] Encoding training sessions to build vector store...")
    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            emb = encoder(input_ids, attention_mask)  # [B, 768]
            all_embeddings.append(emb.cpu().numpy())
            all_labels.extend(labels.int().tolist())

    embeddings_np = np.vstack(all_embeddings)  # [N, 768]
    retriever.add(embeddings_np, all_labels)
    retriever.save(save_dir)
    return retriever
