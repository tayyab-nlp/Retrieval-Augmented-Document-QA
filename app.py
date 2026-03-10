"""Gradio app for Retrieval-Augmented Document QA."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import gradio as gr

from src.config import (
    APP_DESCRIPTION,
    APP_TITLE,
    DEFAULT_TOP_K,
    EXAMPLE_QUESTIONS,
    MAX_TOP_K,
    MAX_FILES,
    MODEL_OPTIONS,
)
from src.rag_pipeline import RAGPipeline

APP_CSS = """
#app-shell {
  max-width: 1320px;
  margin: 0 auto;
}
.panel-card {
  border: 1px solid #d8e0ef;
  border-radius: 14px;
  padding: 12px;
}
.output-card {
  border: 1px solid #e1e8f5;
  border-radius: 12px;
  padding: 12px;
}
.section-label {
  margin-bottom: 8px;
}
#process-btn, #ask-btn {
  min-height: 44px;
}
"""


def _new_pipeline() -> RAGPipeline:
    """Factory used by Gradio state initialization."""
    return RAGPipeline()


def _format_status(message: str, level: str = "info") -> str:
    """Short, readable status line."""
    prefix = {
        "success": "Success",
        "error": "Error",
        "running": "Running",
        "info": "Info",
    }.get(level, "Info")
    return f"### {prefix}: {message}"


def _format_index_summary(summary: Dict[str, str]) -> str:
    """Render index metadata in a concise table."""
    if not summary:
        return "_No index summary yet. Upload documents and click **Process Documents**._"

    rows = "\n".join(f"| {key} | {value} |" for key, value in summary.items())
    return "\n".join(
        [
            "### Index Summary",
            "| Metric | Value |",
            "|---|---|",
            rows,
        ]
    )


def _format_chunks(chunks: List[Dict[str, object]]) -> str:
    """Render retrieved chunks with clear source metadata."""
    if not chunks:
        return "_No chunks retrieved yet._"

    blocks = ["### Retrieved Chunks"]
    for idx, chunk in enumerate(chunks, start=1):
        blocks.append(
            (
                f"#### Chunk {idx} — {chunk['source']} (chunk {chunk['chunk_index']}, "
                f"score {float(chunk['score']):.3f})\n"
                f"{chunk['text']}"
            )
        )
    return "\n\n".join(blocks)


def _format_sources(sources: List[str]) -> str:
    """Render source list."""
    if not sources:
        return "_No sources available yet._"

    lines = "\n".join(f"- {source}" for source in sources)
    return f"### Sources Used\n{lines}"


def _format_trace(steps: List[str]) -> str:
    """Render pipeline trace as numbered timeline."""
    if not steps:
        return "_No trace yet._"

    lines = "\n".join(f"{i}. {step}" for i, step in enumerate(steps, start=1))
    return f"### Pipeline Trace\n{lines}"


def _format_live(steps: List[str]) -> str:
    """Render short live progress panel."""
    if not steps:
        return "### Live Progress\n- Waiting for a run."

    recent = steps[-5:]
    lines = "\n".join(f"- {item}" for item in recent)
    return f"### Live Progress\n- Steps completed: {len(steps)}\n{lines}"


def _extract_file_paths(files: Optional[List[object]]) -> List[str]:
    """Normalize Gradio file objects to paths."""
    if not files:
        return []

    paths: List[str] = []
    for file in files:
        file_path = getattr(file, "name", None)
        if isinstance(file_path, str):
            paths.append(file_path)
    return paths


def process_documents(
    files: Optional[List[object]],
    model_id: str,
    state: Optional[RAGPipeline],
) -> Tuple[RAGPipeline, str, str, str, str, str, str, str]:
    """Build FAISS index from uploaded files."""
    pipeline = state or _new_pipeline()
    pipeline.model_id = model_id

    file_paths = _extract_file_paths(files)
    if not file_paths:
        message = "Please upload 1 to 3 files before processing."
        return (
            pipeline,
            _format_status(message, "error"),
            _format_live([message]),
            _format_index_summary({}),
            "_No answer yet._",
            "_No chunks retrieved yet._",
            "_No sources yet._",
            "_No trace yet._",
        )

    if len(file_paths) > MAX_FILES:
        message = f"Please upload at most {MAX_FILES} files."
        return (
            pipeline,
            _format_status(message, "error"),
            _format_live([message]),
            _format_index_summary({}),
            "_No answer yet._",
            "_No chunks retrieved yet._",
            "_No sources yet._",
            "_No trace yet._",
        )

    try:
        summary = pipeline.process_documents(file_paths)
        steps = [
            "Loaded uploaded documents.",
            "Split documents into overlapping chunks.",
            "Generated embeddings with all-MiniLM-L6-v2.",
            "Built FAISS vector index.",
        ]
        return (
            pipeline,
            _format_status("Knowledge base is ready.", "success"),
            _format_live(steps),
            _format_index_summary(summary),
            "_No answer yet. Ask a question in the Task tab._",
            "_No chunks retrieved yet._",
            "_No sources yet._",
            _format_trace(steps),
        )
    except Exception as exc:  # noqa: BLE001
        message = f"Document processing failed: {exc}"
        return (
            pipeline,
            _format_status(message, "error"),
            _format_live([message]),
            _format_index_summary({}),
            "_No answer yet._",
            "_No chunks retrieved yet._",
            "_No sources yet._",
            "_No trace yet._",
        )


def ask_question(
    api_key: str,
    question: str,
    top_k: int,
    state: Optional[RAGPipeline],
):
    """Retrieve context + generate answer with live updates."""
    pipeline = state or _new_pipeline()

    if not pipeline.has_index():
        message = "Please process documents first."
        yield (
            pipeline,
            _format_status(message, "error"),
            _format_live([message]),
            _format_index_summary(pipeline.last_index_summary),
            "_No answer yet._",
            "_No chunks retrieved yet._",
            "_No sources yet._",
            _format_trace([message]),
        )
        return

    if not api_key or not api_key.strip():
        message = "Gemini API key is required."
        yield (
            pipeline,
            _format_status(message, "error"),
            _format_live([message]),
            _format_index_summary(pipeline.last_index_summary),
            "_No answer yet._",
            "_No chunks retrieved yet._",
            "_No sources yet._",
            _format_trace([message]),
        )
        return

    if not question or not question.strip():
        message = "Please enter a question."
        yield (
            pipeline,
            _format_status(message, "error"),
            _format_live([message]),
            _format_index_summary(pipeline.last_index_summary),
            "_No answer yet._",
            "_No chunks retrieved yet._",
            "_No sources yet._",
            _format_trace([message]),
        )
        return

    steps: List[str] = ["Question received."]
    yield (
        pipeline,
        _format_status("Retrieving relevant chunks from FAISS...", "running"),
        _format_live(steps),
        _format_index_summary(pipeline.last_index_summary),
        "_Generating answer..._",
        "_Retrieval in progress..._",
        "_Sources will appear after retrieval._",
        _format_trace(steps),
    )

    try:
        retrieved_chunks = pipeline.retrieve(question=question, top_k=top_k)
        steps.append(f"Retrieved top-{min(top_k, len(retrieved_chunks))} chunks.")

        sources = []
        for chunk in retrieved_chunks:
            sources.append(f"{chunk['source']} — chunk {chunk['chunk_index']}")
        unique_sources = sorted(set(sources))

        yield (
            pipeline,
            _format_status("Generating grounded answer with Gemini...", "running"),
            _format_live(steps),
            _format_index_summary(pipeline.last_index_summary),
            "_Generating answer..._",
            _format_chunks(retrieved_chunks),
            _format_sources(unique_sources),
            _format_trace(steps),
        )

        generation = pipeline.generate_answer(
            api_key=api_key,
            question=question,
            retrieved_chunks=retrieved_chunks,
        )
        steps.append("Generated final answer from retrieved context.")

        answer = str(generation["answer"]).strip()
        if not answer:
            answer = "No answer returned by Gemini."

        final_answer = "\n".join(
            [
                "### Final Answer",
                answer,
            ]
        )

        yield (
            pipeline,
            _format_status("Done. Retrieval-Augmented answer is ready.", "success"),
            _format_live(steps),
            _format_index_summary(pipeline.last_index_summary),
            final_answer,
            _format_chunks(retrieved_chunks),
            _format_sources(list(generation["sources"])),
            _format_trace(steps),
        )
    except Exception as exc:  # noqa: BLE001
        steps.append(f"Run failed: {exc}")
        yield (
            pipeline,
            _format_status(f"Question answering failed: {exc}", "error"),
            _format_live(steps),
            _format_index_summary(pipeline.last_index_summary),
            "_No answer produced due to an error._",
            "_No chunks to display._",
            "_No sources to display._",
            _format_trace(steps),
        )


with gr.Blocks(title=APP_TITLE) as demo:
    pipeline_state = gr.State(_new_pipeline())

    with gr.Column(elem_id="app-shell"):
        gr.Markdown(f"# {APP_TITLE}")
        gr.Markdown(APP_DESCRIPTION)

        with gr.Row(equal_height=False):
            with gr.Column(scale=4, elem_classes=["panel-card"]):
                with gr.Tabs():
                    with gr.Tab("Documents"):
                        gr.Markdown("### Upload Documents")
                        files_input = gr.File(
                            file_count="multiple",
                            file_types=[".pdf", ".txt", ".md"],
                            label=f"Files (max {MAX_FILES})",
                        )
                        process_button = gr.Button("Process Documents", variant="primary", elem_id="process-btn")

                    with gr.Tab("Task"):
                        gr.Markdown("### Ask a Question")
                        question_input = gr.Textbox(
                            label="Question",
                            lines=5,
                            placeholder="Example: What are the main contributions in these documents?",
                        )
                        gr.Examples(
                            examples=[[item] for item in EXAMPLE_QUESTIONS],
                            inputs=[question_input],
                            label="Example Questions",
                        )
                        top_k_input = gr.Slider(
                            minimum=1,
                            maximum=MAX_TOP_K,
                            step=1,
                            value=DEFAULT_TOP_K,
                            label="Top-K chunks to retrieve",
                        )
                        ask_button = gr.Button("Ask Question", variant="primary", elem_id="ask-btn")

                    with gr.Tab("API & Model"):
                        gr.Markdown("### Gemini Configuration")
                        api_key_input = gr.Textbox(
                            label="Gemini API Key",
                            type="password",
                            placeholder="Paste Gemini API key",
                        )
                        model_input = gr.Dropdown(
                            choices=MODEL_OPTIONS,
                            value=MODEL_OPTIONS[0],
                            label="Gemini Model",
                        )
                        gr.Markdown("API key is used only for this session run and is never stored or logged.")

            with gr.Column(scale=6, elem_classes=["panel-card"]):
                gr.Markdown("## Results")
                status_output = gr.Markdown(_format_status("Upload documents and process them to start."))
                with gr.Tabs():
                    with gr.Tab("Live Progress"):
                        live_output = gr.Markdown(_format_live([]), elem_classes=["output-card"])

                    with gr.Tab("Index Summary"):
                        index_summary_output = gr.Markdown(_format_index_summary({}), elem_classes=["output-card"])

                    with gr.Tab("Final Answer"):
                        answer_output = gr.Markdown("_No answer yet._", elem_classes=["output-card"])

                    with gr.Tab("Retrieved Chunks"):
                        chunks_output = gr.Markdown("_No chunks retrieved yet._", elem_classes=["output-card"])

                    with gr.Tab("Sources"):
                        sources_output = gr.Markdown("_No sources yet._", elem_classes=["output-card"])

                    with gr.Tab("Pipeline Trace"):
                        trace_output = gr.Markdown("_No trace yet._", elem_classes=["output-card"])

    process_button.click(
        fn=process_documents,
        inputs=[files_input, model_input, pipeline_state],
        outputs=[
            pipeline_state,
            status_output,
            live_output,
            index_summary_output,
            answer_output,
            chunks_output,
            sources_output,
            trace_output,
        ],
    )

    ask_button.click(
        fn=ask_question,
        inputs=[api_key_input, question_input, top_k_input, pipeline_state],
        outputs=[
            pipeline_state,
            status_output,
            live_output,
            index_summary_output,
            answer_output,
            chunks_output,
            sources_output,
            trace_output,
        ],
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.queue(default_concurrency_limit=2, max_size=24).launch(
        server_name="0.0.0.0",
        server_port=port,
        css=APP_CSS,
        theme=gr.themes.Default(),
    )
