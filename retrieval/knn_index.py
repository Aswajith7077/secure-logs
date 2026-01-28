# retrieval/knn_index.py
import faiss
import numpy as np


class KNNRetriever:
    def __init__(self, dim, k=100):
        self.k = k
        self.index = faiss.IndexHNSWFlat(dim, 32)
        self.labels = []

    def add(self, embeddings, labels):
        self.index.add(embeddings.astype(np.float32))
        self.labels.extend(labels)

    def query(self, embedding):
        D, I = self.index.search(embedding.reshape(1, -1), self.k)
        retrieved_labels = [self.labels[i] for i in I[0]]
        weights = np.exp(-D[0])
        return np.average(retrieved_labels, weights=weights)
