"""Fast character-based chunking utilities."""

from __future__ import annotations

from typing import Dict, List

def chunk_documents(
    documents: List[Dict[str, str]],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Dict[str, str]]:
    """Split each document into overlapping chunks for retrieval."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    step = max(chunk_size - chunk_overlap, 1)

    chunks: List[Dict[str, str]] = []
    for document in documents:
        text = document["text"].replace("\r\n", "\n")
        chunk_idx = 0
        for start in range(0, len(text), step):
            window = text[start : start + chunk_size]
            clean_text = window.strip()
            if not clean_text:
                continue
            chunk_idx += 1
            chunks.append(
                {
                    "chunk_id": f"{document['source']}#{chunk_idx}",
                    "source": document["source"],
                    "chunk_index": chunk_idx,
                    "text": clean_text,
                }
            )
            if start + chunk_size >= len(text):
                break

    if not chunks:
        raise ValueError("No chunks were created from the uploaded files.")

    return chunks
