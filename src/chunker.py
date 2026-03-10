"""Chunking utilities built on LangChain text splitters."""

from __future__ import annotations

from typing import Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: List[Dict[str, str]],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Dict[str, str]]:
    """Split each document into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks: List[Dict[str, str]] = []
    for document in documents:
        split_texts = splitter.split_text(document["text"])
        for i, text in enumerate(split_texts, start=1):
            clean_text = text.strip()
            if not clean_text:
                continue
            chunks.append(
                {
                    "chunk_id": f"{document['source']}#{i}",
                    "source": document["source"],
                    "chunk_index": i,
                    "text": clean_text,
                }
            )

    if not chunks:
        raise ValueError("No chunks were created from the uploaded files.")

    return chunks
