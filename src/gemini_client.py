"""Gemini API client (HTTP requests only)."""

from __future__ import annotations

from typing import Any, Dict, List

import requests

from .config import GENERATE_CONTENT_API, MODEL_ID, TIMEOUT_SECONDS


def _extract_text_from_payload(payload: Any) -> str:
    """Extract generated text from dict/list Gemini responses."""
    containers: List[Dict[str, Any]] = []
    if isinstance(payload, dict):
        containers = [payload]
    elif isinstance(payload, list):
        containers = [item for item in payload if isinstance(item, dict)]

    parts: List[str] = []
    for container in containers:
        candidates = container.get("candidates", []) or []
        for candidate in candidates:
            content = candidate.get("content", {}) or {}
            content_parts = content.get("parts", []) or []
            for part in content_parts:
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)

    return "\n".join(parts).strip()


def generate_text(
    api_key: str,
    prompt: str,
    model_id: str = MODEL_ID,
    timeout_seconds: int = TIMEOUT_SECONDS,
) -> str:
    """Generate text with Gemini using streamGenerateContent endpoint."""
    if not api_key or not api_key.strip():
        raise ValueError("Missing Gemini API key.")
    if not prompt or not prompt.strip():
        raise ValueError("Prompt is empty.")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_id}:{GENERATE_CONTENT_API}?key={api_key.strip()}"
    )
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "thinkingConfig": {
                "thinkingLevel": "MINIMAL",
            }
        },
    }

    try:
        response = requests.post(url, json=body, timeout=timeout_seconds)
    except requests.Timeout as exc:
        raise RuntimeError(
            f"Gemini request timed out after {timeout_seconds} seconds."
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Gemini request failed: {exc}") from exc

    if response.status_code != 200:
        error_message = ""
        try:
            error_payload = response.json()
            error_message = (
                error_payload.get("error", {}).get("message")
                if isinstance(error_payload, dict)
                else ""
            )
        except ValueError:
            error_message = response.text.strip()

        detail = error_message or response.text.strip() or "Unknown error"
        raise RuntimeError(
            f"Gemini API error ({response.status_code}): {detail}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("Gemini returned non-JSON response.") from exc

    text = _extract_text_from_payload(payload)
    if text:
        return text

    if isinstance(payload, dict):
        error_info = payload.get("error", {}).get("message")
        if error_info:
            raise RuntimeError(f"Gemini error: {error_info}")

    raise RuntimeError(
        "Gemini returned an empty or unexpected response. "
        "Try again or reduce question complexity."
    )
