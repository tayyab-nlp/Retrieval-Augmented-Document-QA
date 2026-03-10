"""Main RAG pipeline orchestration."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional

from .chunker import chunk_documents
from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    MAX_TOP_K,
    MODEL_ID,
)
from .document_loader import load_documents
from .embeddings import EmbeddingModel
from .gemini_client import generate_text
from .retriever import retrieve_chunks
from .vector_store import FaissVectorStore


class RAGPipeline:
    """Holds in-memory index state and question-answering logic."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        embedding_model_name: str = EMBEDDING_MODEL,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        default_top_k: int = DEFAULT_TOP_K,
    ):
        self.model_id = model_id
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.default_top_k = min(max(default_top_k, 1), MAX_TOP_K)

        self._embedder: Optional[EmbeddingModel] = None
        self.documents: List[Dict[str, str]] = []
        self.chunks: List[Dict[str, str]] = []
        self.vector_store: Optional[FaissVectorStore] = None
        self.last_index_summary: Dict[str, str] = {}

    def _get_embedder(self) -> EmbeddingModel:
        """Lazy-load embedding model once per app session."""
        if self._embedder is None:
            self._embedder = EmbeddingModel(self.embedding_model_name)
        return self._embedder

    def has_index(self) -> bool:
        """Return True when FAISS index is ready."""
        return self.vector_store is not None and self.vector_store.size() > 0

    def process_documents(self, file_paths: List[str]) -> Dict[str, str]:
        """Load, chunk, embed, and index uploaded documents."""
        documents = load_documents(file_paths)
        chunks = chunk_documents(documents, self.chunk_size, self.chunk_overlap)
        texts = [chunk["text"] for chunk in chunks]

        embedder = self._get_embedder()
        vectors = embedder.embed_texts(texts)
        vector_store = FaissVectorStore.from_embeddings(vectors, chunks)

        self.documents = documents
        self.chunks = chunks
        self.vector_store = vector_store

        summary = {
            "Documents processed": str(len(documents)),
            "Chunks created": str(len(chunks)),
            "Embedding model": self.embedding_model_name,
            "Vector store": "FAISS",
            "Chunk size": str(self.chunk_size),
            "Chunk overlap": str(self.chunk_overlap),
            "Indexed files": ", ".join(doc["source"] for doc in documents),
        }
        self.last_index_summary = summary
        return summary

    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[Dict[str, str]]:
        """Retrieve relevant chunks for a question."""
        if not self.has_index():
            raise ValueError("No FAISS index found. Process documents first.")
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        k = min(max(top_k or self.default_top_k, 1), MAX_TOP_K)
        return retrieve_chunks(
            question=question.strip(),
            embedder=self._get_embedder(),
            vector_store=self.vector_store,  # type: ignore[arg-type]
            top_k=k,
        )

    @staticmethod
    def build_prompt(question: str, retrieved_chunks: List[Dict[str, str]]) -> str:
        """Create grounded prompt with explicit context usage rules."""
        context_blocks = []
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            context_blocks.append(
                (
                    f"[{idx}] source={chunk['source']} chunk={chunk['chunk_index']}\n"
                    f"{chunk['text']}"
                )
            )

        context = "\n\n".join(context_blocks) if context_blocks else "No retrieved context."

        return (
            "You are a precise document QA assistant.\n"
            "Rules:\n"
            "1) Use only the provided context.\n"
            "2) Do not invent details.\n"
            "3) Cite supporting evidence inline using bracket citations like [1], [2].\n"
            "4) Keep markdown clean: no code fences and no triple quotes.\n"
            "5) End with a heading `### References` and list cited ids.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question.strip()}\n\n"
            "Return a concise, direct answer to the question with good structure."
        )

    @staticmethod
    def build_no_context_prompt(question: str) -> str:
        """Create fallback prompt when no indexed documents are available."""
        return (
            "You are a helpful assistant.\n"
            "No uploaded document context is available for this question.\n"
            "Give a concise, structured answer using your general knowledge.\n"
            "Do not include fake document citations.\n\n"
            f"Question:\n{question.strip()}"
        )

    def generate_answer(
        self,
        api_key: str,
        question: str,
        retrieved_chunks: List[Dict[str, str]],
    ) -> Dict[str, object]:
        """Generate final answer from retrieved context with Gemini."""
        prompt = self.build_prompt(question=question, retrieved_chunks=retrieved_chunks)
        answer = generate_text(api_key=api_key, prompt=prompt, model_id=self.model_id)

        # Ordered unique source lines for clear provenance display.
        source_map = OrderedDict()
        for chunk in retrieved_chunks:
            key = f"{chunk['source']} — chunk {chunk['chunk_index']}"
            source_map[key] = None

        return {
            "answer": answer,
            "prompt": prompt,
            "sources": list(source_map.keys()),
        }

    def generate_answer_without_retrieval(
        self,
        api_key: str,
        question: str,
    ) -> Dict[str, object]:
        """Generate answer directly from Gemini when no index is available."""
        prompt = self.build_no_context_prompt(question)
        answer = generate_text(api_key=api_key, prompt=prompt, model_id=self.model_id)
        return {
            "answer": answer,
            "prompt": prompt,
            "sources": [],
        }
