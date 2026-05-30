from typing import List, Dict
import faiss
import numpy as np


class FaissIndex:
    def __init__(self, embedding_dim: int):
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunks: List[Dict] = []

    def add(self, embeddings: np.ndarray, chunks: List[Dict]):
        assert len(embeddings) == len(chunks)

        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, k: int = 5):
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            results.append({
                "chunk": self.chunks[idx],
                "score": float(score)
            })

        return results
