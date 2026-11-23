"""Grounded answers from retrieved chunks.

By default answers are extractive: the best-matching sections are returned
verbatim with their source, so nothing is made up. If an Ollama server is
running and ``SAHAAYAK_LLM=ollama`` is set, a local model writes a short answer
using only those sections; any error falls back to the extractive answer.
"""
from __future__ import annotations

import os

import httpx

from .retrieval import Hit

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

PROMPT = """You are Sahaayak, an assistant that explains Indian government schemes in simple words.
Answer the question using ONLY the context below. If the context does not contain the answer,
say you are not sure and suggest checking the official portal. Mention scheme names.
Keep it under 120 words.

Context:
{context}

Question: {question}
Answer:"""


def build_context(hits: list[Hit]) -> str:
    return "\n".join(f"[{i + 1}] {h.chunk.scheme_name} - {h.chunk.title}: {h.chunk.text}"
                     for i, h in enumerate(hits))


def extractive_answer(hits: list[Hit]) -> str:
    if not hits:
        return "I could not find a matching scheme. Try describing your situation in different words."
    lines, seen = [], set()
    for h in hits:
        key = (h.chunk.scheme_id, h.chunk.section)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{h.chunk.scheme_name} - {h.chunk.title}: {h.chunk.text}")
        if len(lines) == 3:
            break
    return "\n\n".join(lines)


def llm_answer(question: str, hits: list[Hit], timeout: float = 60.0) -> str | None:
    if os.getenv("SAHAAYAK_LLM", "").lower() != "ollama" or not hits:
        return None
    try:
        resp = httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "stream": False,
                  "prompt": PROMPT.format(context=build_context(hits), question=question)},
            timeout=timeout,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        return text or None
    except Exception:
        return None


def answer(question: str, hits: list[Hit]) -> tuple[str, str]:
    """Return (answer_text, mode) where mode is 'llm' or 'extractive'."""
    generated = llm_answer(question, hits)
    if generated:
        return generated, "llm"
    return extractive_answer(hits), "extractive"
