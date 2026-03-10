---
title: Retrieval-Augmented Document QA
sdk: gradio
python_version: "3.10"
app_file: app.py
pinned: false
---

# Retrieval-Augmented Document QA

Space Demo Link: [https://vtayyab6-retrieval-augmented-document-qa.hf.space](https://vtayyab6-retrieval-augmented-document-qa.hf.space)

A clean Gradio RAG demo where you upload files, build a FAISS index, ask questions, and get grounded Gemini answers with visible chunk-level citations and source references.

## Features

| Area | Details |
|---|---|
| Document Support | Upload up to 3 files (`.pdf`, `.txt`, `.md`) for fast CPU demos. |
| RAG Pipeline | Load text -> chunk -> embed (`all-MiniLM-L6-v2`) -> FAISS retrieval -> Gemini answer. |
| Grounded Output | Final answer includes inline citations (`[1]`, `[2]`) and a references section. |
| Transparency | Separate tabs for progress, index summary, retrieved chunks, sources, and trace. |
| API Handling | Gemini API key is entered in the UI, used per run, and not stored/logged. |
| Fast on Spaces | Lightweight defaults, text limits, and cached embedding model loading. |

## Screenshots

Initial screen before running:

![Before running](screenshots/1.%20before%20running.png)

Documents processed and chunking complete:

![Chunking finished](screenshots/2.%20chunking%20finished.png)

Grounded final answer with citations:

![Sample answer](screenshots/3.%20sample%20answer%20rag%20grounded.png)

Retrieved chunks used for answer generation:

![Retrieved chunks](screenshots/4.%20retrieved%20chunks.png)

Source mapping and citation references:

![Sources](screenshots/5.%20sources.png)

Step-by-step pipeline execution trace:

![Pipeline trace](screenshots/6.%20pipeline%20traces.png)

## Project Structure

```text
retrieval-augmented-document-qa/
├── app.py
├── requirements.txt
├── README.md
├── screenshots/
└── src/
    ├── __init__.py
    ├── chunker.py
    ├── config.py
    ├── document_loader.py
    ├── embeddings.py
    ├── gemini_client.py
    ├── rag_pipeline.py
    ├── retriever.py
    └── vector_store.py
```

## How It Works

1. Upload documents (optional for general Q&A mode).
2. Click `Process Documents` to build the FAISS knowledge base.
3. Add your Gemini API key in the `API` tab.
4. Ask a question in the `Task` tab.
5. Review answer, citations, retrieved chunks, sources, and pipeline trace.

## Local Setup

```bash
cd retrieval-augmented-document-qa
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 app.py
```
