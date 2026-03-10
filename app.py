"""Gradio app for Retrieval-Augmented Document QA."""

from __future__ import annotations

import os
import re
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
        snippet = str(chunk["text"]).strip()
        blocks.append(
            (
                f"#### [{idx}] {chunk['source']} (chunk {chunk['chunk_index']}, "
                f"score {float(chunk['score']):.3f})\n"
                f"{snippet}"
            )
        )
    return "\n\n".join(blocks)


def _format_sources(chunks: List[Dict[str, object]]) -> str:
    """Render source list."""
    if not chunks:
        return "_No sources available yet._"

    rows = []
    for idx, chunk in enumerate(chunks, start=1):
        rows.append(f"| [{idx}] | {chunk['source']} | {chunk['chunk_index']} |")
    table = "\n".join(rows)
    return "\n".join(
        [
            "### Sources Used",
            "| Citation | Document | Chunk |",
            "|---|---|---|",
            table,
        ]
    )


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


def _reflow_markdown_paragraphs(text: str) -> str:
    """Collapse accidental single line breaks while preserving markdown structure."""
    lines = text.splitlines()
    out: List[str] = []
    paragraph: List[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            out.append(" ".join(part.strip() for part in paragraph if part.strip()))
            paragraph.clear()

    for raw in lines:
        line = raw.strip()
        if not line:
            flush_paragraph()
            if out and out[-1] != "":
                out.append("")
            continue

        is_structured = bool(
            re.match(r"^(#{1,6}\s|[-*+]\s|\d+\.\s|>\s|\|\s*.+\s*\||\[?\d+\]?\s)", line)
        )
        if is_structured:
            flush_paragraph()
            out.append(line)
            continue

        paragraph.append(line)

    flush_paragraph()
    return "\n".join(out).strip()


def _clean_model_markdown(text: str) -> str:
    """Normalize model output into clean markdown."""
    cleaned = (text or "").replace("\r\n", "\n").strip()

    for fence in ("```markdown", "```md", "```text", "```"):
        if cleaned.lower().startswith(fence):
            cleaned = cleaned[len(fence) :].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    if cleaned.startswith('"""') and cleaned.endswith('"""') and len(cleaned) > 6:
        cleaned = cleaned[3:-3].strip()
    if cleaned.startswith('"') and cleaned.endswith('"') and cleaned.count('"') <= 2:
        cleaned = cleaned[1:-1].strip()

    cleaned = re.split(
        r"\n#{1,6}\s*references\b|\nreferences\s*:",
        cleaned,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    cleaned = re.sub(r"\[chunk\s*(\d+)\]", r"[\1]", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = _reflow_markdown_paragraphs(cleaned)
    return cleaned


def _ensure_inline_citations(answer: str, max_refs: int) -> str:
    """Ensure answer contains at least one inline citation marker."""
    if max_refs <= 0:
        return answer
    if re.search(r"\[\d+\]", answer):
        return answer

    citation_span = " ".join(f"[{i}]" for i in range(1, min(max_refs, 3) + 1))
    lines = answer.splitlines()
    for idx in range(len(lines) - 1, -1, -1):
        stripped = lines[idx].strip()
        if stripped and not stripped.startswith("#"):
            lines[idx] = f"{stripped.rstrip('.')} {citation_span}"
            return "\n".join(lines)
    return f"{answer}\n\n{citation_span}"


def _format_final_answer(answer: str, chunks: List[Dict[str, object]]) -> str:
    """Render final answer with inline citations and reference list."""
    normalized = _ensure_inline_citations(_clean_model_markdown(answer), len(chunks))
    lines = ["### Final Answer", normalized, "", "### References"]
    if not chunks:
        lines.append("- No document references (documents were not indexed).")
        return "\n".join(lines)

    for idx, chunk in enumerate(chunks, start=1):
        snippet = str(chunk["text"]).replace("\n", " ").strip()
        if len(snippet) > 180:
            snippet = f"{snippet[:177]}..."
        lines.append(
            f"- [{idx}] **{chunk['source']}** chunk {chunk['chunk_index']}: {snippet}"
        )
    return "\n".join(lines)


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
) -> Tuple[RAGPipeline, str, str, str, str, str, str, str, gr.Tabs]:
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
            gr.Tabs(selected="live-progress"),
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
            gr.Tabs(selected="live-progress"),
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
            gr.Tabs(selected="index-summary"),
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
            gr.Tabs(selected="live-progress"),
        )


def ask_question(
    api_key: str,
    question: str,
    top_k: int,
    state: Optional[RAGPipeline],
):
    """Retrieve context + generate answer with live updates."""
    pipeline = state or _new_pipeline()

    if not api_key or not api_key.strip():
        message = "Please add your Gemini API key in the API tab."
        yield (
            pipeline,
            _format_status(message, "error"),
            _format_live([message]),
            _format_index_summary(pipeline.last_index_summary),
            "_No answer yet._",
            "_No chunks retrieved yet._",
            "_No sources yet._",
            _format_trace([message]),
            gr.Tabs(selected="live-progress"),
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
            gr.Tabs(selected="live-progress"),
        )
        return

    steps: List[str] = ["Question received."]

    if not pipeline.has_index():
        steps.append("No documents indexed. Running general answer mode.")
        yield (
            pipeline,
            _format_status("No index found. Generating answer without document retrieval...", "running"),
            _format_live(steps),
            _format_index_summary(pipeline.last_index_summary),
            "_Generating answer..._",
            "_No retrieved chunks (documents not indexed)._",
            "_No sources available in general mode._",
            _format_trace(steps),
            gr.Tabs(selected="live-progress"),
        )

        try:
            generation = pipeline.generate_answer_without_retrieval(
                api_key=api_key,
                question=question,
            )
            steps.append("Generated final answer without retrieval.")
            answer = str(generation["answer"]).strip() or "No answer returned by Gemini."
            final_answer = _format_final_answer(answer, [])
            yield (
                pipeline,
                _format_status(
                    "Done. Answer is ready (no documents indexed, so no citations).",
                    "success",
                ),
                _format_live(steps),
                _format_index_summary(pipeline.last_index_summary),
                final_answer,
                "_No retrieved chunks (documents not indexed)._",
                "_No sources available in general mode._",
                _format_trace(steps),
                gr.Tabs(selected="final-answer"),
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
                gr.Tabs(selected="live-progress"),
            )
        return

    yield (
        pipeline,
        _format_status("Retrieving relevant chunks from FAISS...", "running"),
        _format_live(steps),
        _format_index_summary(pipeline.last_index_summary),
        "_Generating answer..._",
        "_Retrieval in progress..._",
        "_Sources will appear after retrieval._",
        _format_trace(steps),
        gr.Tabs(selected="live-progress"),
    )

    try:
        retrieved_chunks = pipeline.retrieve(question=question, top_k=top_k)
        steps.append(f"Retrieved top-{min(top_k, len(retrieved_chunks))} chunks.")

        yield (
            pipeline,
            _format_status("Generating grounded answer with Gemini...", "running"),
            _format_live(steps),
            _format_index_summary(pipeline.last_index_summary),
            "_Generating answer..._",
            _format_chunks(retrieved_chunks),
            _format_sources(retrieved_chunks),
            _format_trace(steps),
            gr.Tabs(selected="retrieved-chunks"),
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
        final_answer = _format_final_answer(answer, retrieved_chunks)

        yield (
            pipeline,
            _format_status("Done. Retrieval-Augmented answer is ready.", "success"),
            _format_live(steps),
            _format_index_summary(pipeline.last_index_summary),
            final_answer,
            _format_chunks(retrieved_chunks),
            _format_sources(retrieved_chunks),
            _format_trace(steps),
            gr.Tabs(selected="final-answer"),
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
            gr.Tabs(selected="live-progress"),
        )


with gr.Blocks(title=APP_TITLE) as demo:
    pipeline_state = gr.State(_new_pipeline())

    with gr.Column(elem_id="app-shell"):
        gr.Markdown(f"# {APP_TITLE}")
        gr.Markdown(APP_DESCRIPTION)

        with gr.Row(equal_height=False):
            with gr.Column(scale=4, elem_classes=["panel-card"]):
                with gr.Tabs():
                    with gr.Tab("Task"):
                        gr.Markdown("### Ask a Question")
                        question_input = gr.Textbox(
                            label="Question",
                            lines=5,
                            placeholder="Ask about uploaded documents, or ask a general question.",
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

                    with gr.Tab("Documents"):
                        gr.Markdown("### Upload Documents")
                        files_input = gr.File(
                            file_count="multiple",
                            file_types=[".pdf", ".txt", ".md"],
                            label=f"Files (max {MAX_FILES})",
                        )
                        process_button = gr.Button(
                            "Process Documents",
                            variant="primary",
                            elem_id="process-btn",
                        )

                    with gr.Tab("API"):
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
                with gr.Tabs(selected="live-progress") as right_tabs:
                    with gr.Tab("Live Progress", id="live-progress"):
                        live_output = gr.Markdown(
                            _format_live([]),
                            elem_classes=["output-card"],
                        )

                    with gr.Tab("Index Summary", id="index-summary"):
                        index_summary_output = gr.Markdown(
                            _format_index_summary({}),
                            elem_classes=["output-card"],
                        )

                    with gr.Tab("Retrieved Chunks", id="retrieved-chunks"):
                        chunks_output = gr.Markdown(
                            "_No chunks retrieved yet._",
                            elem_classes=["output-card"],
                        )

                    with gr.Tab("Sources", id="sources"):
                        sources_output = gr.Markdown(
                            "_No sources yet._",
                            elem_classes=["output-card"],
                        )

                    with gr.Tab("Pipeline Trace", id="pipeline-trace"):
                        trace_output = gr.Markdown(
                            "_No trace yet._",
                            elem_classes=["output-card"],
                        )

                    with gr.Tab("Final Answer", id="final-answer"):
                        answer_output = gr.Markdown(
                            "_No answer yet._",
                            elem_classes=["output-card"],
                            buttons=["copy"],
                        )

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
            right_tabs,
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
            right_tabs,
        ],
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.queue(default_concurrency_limit=2, max_size=24).launch(
        server_name="0.0.0.0",
        server_port=port,
        css=APP_CSS,
    )
