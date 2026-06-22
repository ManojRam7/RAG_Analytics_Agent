"""Prompt templates and context formatting for source-grounded answers."""
from __future__ import annotations

from typing import Dict, List, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

SYSTEM_INSTRUCTIONS = (
    "You are a meticulous business analytics assistant. Answer the user's "
    "question using ONLY the information in the provided report excerpts.\n"
    "- Ground every claim in the excerpts and cite them inline using their tags, "
    "e.g. [S1], [S2].\n"
    "- Prefer concrete numbers, dates and names taken directly from the excerpts.\n"
    "- Do NOT use outside knowledge or invent figures.\n"
    '- If the answer is not in the excerpts, reply exactly: '
    '"I could not find this in the provided reports."'
)

RAG_PROMPT = PromptTemplate.from_template(
    "{system}\n\n"
    "# Report excerpts\n{context}\n\n"
    "# Question\n{question}\n\n"
    "# Source-grounded answer (cite excerpts as [S#])\n"
)


def format_context(scored_docs: List[Tuple[Document, float]]) -> Tuple[str, List[Dict]]:
    """Turn retrieved (doc, score) pairs into a prompt context string + source list.

    Returns ``(context_str, sources)`` where each source dict holds the citation
    tag, originating file, page, raw similarity score, the full chunk text (for
    the extractive engine) and a short snippet (for display).
    """
    lines: List[str] = []
    sources: List[Dict] = []
    for i, (doc, score) in enumerate(scored_docs, start=1):
        tag = f"S{i}"
        src = doc.metadata.get("source", "?")
        page = doc.metadata.get("page")
        loc = src + (f" (p.{page + 1})" if isinstance(page, int) else "")
        text = doc.page_content.strip()
        lines.append(f"[{tag}] from {loc}\n{text}")
        sources.append(
            {
                "tag": tag,
                "source": src,
                "page": page,
                "score": float(score),
                "text": text,
                "snippet": text[:320] + ("…" if len(text) > 320 else ""),
            }
        )
    return "\n\n".join(lines), sources
