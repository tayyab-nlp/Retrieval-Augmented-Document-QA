---
title: Retrieval-Augmented Document QA
sdk: gradio
python_version: "3.10"
app_file: app.py
pinned: false
---

# Retrieval-Augmented Document QA

A Gradio application that lets users upload documents, build a semantic FAISS index, and ask grounded questions answered by Gemini with retrieved context.

## Features

| Feature | Details |
|---|---|
| Document upload | Supports PDF, TXT, and Markdown (`max 3` files for fast demos). |
| RAG pipeline | Text extraction -> chunking -> embeddings -> FAISS indexing -> retrieval -> Gemini answering. |
| Embeddings | `all-MiniLM-L6-v2` (SentenceTransformers). |
| LLM | `gemini-3.1-flash-lite-preview` via HTTP API. |
| Retrieval transparency | Shows retrieved chunks, source references, and pipeline trace. |
| Security | API key is entered in the UI, used per run, and not stored/logged. |
| Space speed controls | File count limit, PDF page limit, text-size cap, lazy model loading, and shared embedding cache. |

## Project Structure

```text
retrieval-augmented-document-qa/
├── app.py
├── requirements.txt
├── README.md
└── src/
    ├── __init__.py
    ├── config.py
    ├── document_loader.py
    ├── chunker.py
    ├── embeddings.py
    ├── vector_store.py
    ├── retriever.py
    ├── rag_pipeline.py
    └── gemini_client.py
```

## How It Works

1. Upload up to 3 files (`.pdf`, `.txt`, `.md`).
2. Click **Process Documents** to build the knowledge base.
3. Enter your Gemini API key and question.
4. Click **Ask Question**.
5. Review:
   - final answer
   - retrieved chunks
   - sources used
   - execution trace

## Local Setup

```bash
cd retrieval-augmented-document-qa
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:7860`.

## Notes

- This app runs fully on CPU and is designed for Hugging Face Spaces demos.
- Retrieval controls answer grounding; if context is missing, the app prompts Gemini to say so.
