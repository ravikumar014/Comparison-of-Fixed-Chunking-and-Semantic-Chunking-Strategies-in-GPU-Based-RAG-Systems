from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.device import get_device, device_name


class Embedder:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.device = get_device()
        print(f"[Embedder] Using device: {device_name(self.device)}")

        self.model = SentenceTransformer(
            model_name,
            device=str(self.device)
        )

    def embed_chunks(self, chunks: List[Dict]) -> np.ndarray:
        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            batch_size=16,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        return embeddings
