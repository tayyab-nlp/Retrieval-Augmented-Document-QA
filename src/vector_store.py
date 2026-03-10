"""FAISS vector store helpers."""

from __future__ import annotations

from typing import Dict, List

import faiss
import numpy as np


class FaissVectorStore:
    """Simple FAISS cosine-similarity index (using normalized embeddings)."""

    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatIP(dimension)
        self.metadatas: List[Dict[str, str]] = []

    @classmethod
    def from_embeddings(
        cls,
        embeddings: np.ndarray,
        metadatas: List[Dict[str, str]],
    ) -> "FaissVectorStore":
        """Create and populate a FAISS index from embeddings."""
        if embeddings.size == 0:
            raise ValueError("Cannot build FAISS index from empty embeddings.")

        store = cls(embeddings.shape[1])
        store.index.add(embeddings)
        store.metadatas = list(metadatas)
        return store

    def size(self) -> int:
        """Return number of indexed chunks."""
        return len(self.metadatas)

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Dict[str, str]]:
        """Run similarity search and return metadata with scores."""
        if self.size() == 0:
            return []

        query_array = np.asarray([query_vector], dtype="float32")
        k = min(top_k, self.size())
        scores, indices = self.index.search(query_array, k)

        results: List[Dict[str, str]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = dict(self.metadatas[idx])
            item["score"] = float(score)
            results.append(item)

        return results
