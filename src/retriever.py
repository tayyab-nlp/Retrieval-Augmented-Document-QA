"""Retrieval helpers."""

from __future__ import annotations

from typing import Dict, List

from .embeddings import EmbeddingModel
from .vector_store import FaissVectorStore


def retrieve_chunks(
    question: str,
    embedder: EmbeddingModel,
    vector_store: FaissVectorStore,
    top_k: int,
) -> List[Dict[str, str]]:
    """Retrieve top-k most similar chunks for the question."""
    query_embedding = embedder.embed_query(question)
    return vector_store.search(query_embedding, top_k=top_k)
