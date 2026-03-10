"""Document loading utilities for PDF/TXT/Markdown files."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader

from .config import MAX_FILES, SUPPORTED_EXTENSIONS


def load_pdf(file_path: str) -> str:
    """Extract text from a PDF file path."""
    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def load_text(file_path: str) -> str:
    """Load plain text from a UTF-8 compatible file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        return file.read().strip()


def load_documents(file_paths: List[str]) -> List[Dict[str, str]]:
    """Load up to MAX_FILES supported documents into in-memory records."""
    if not file_paths:
        raise ValueError("Please upload at least one document.")
    if len(file_paths) > MAX_FILES:
        raise ValueError(f"Maximum {MAX_FILES} files allowed for this demo.")

    documents: List[Dict[str, str]] = []
    for idx, file_path in enumerate(file_paths, start=1):
        path = Path(file_path)
        extension = path.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(f"Unsupported file type: {path.name}. Use {supported}.")

        text = load_pdf(file_path) if extension == ".pdf" else load_text(file_path)
        text = text.replace("\x00", "").strip()
        if not text:
            raise ValueError(f"No readable text found in {path.name}.")

        documents.append(
            {
                "doc_id": str(idx),
                "source": path.name,
                "text": text,
            }
        )

    if not documents:
        raise ValueError("No valid documents were loaded.")

    return documents
