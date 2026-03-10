"""SentenceTransformer embedding wrapper."""

from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    """Lazy-load and cache embedding model across app sessions."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class EmbeddingModel:
    """Small wrapper to keep embedding calls consistent."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = _load_model(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts into normalized vectors."""
        if not texts:
            return np.empty((0, 0), dtype="float32")

        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype("float32")

    def embed_query(self, question: str) -> np.ndarray:
        """Encode a single query into a normalized vector."""
        vectors = self.model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype("float32")[0]
