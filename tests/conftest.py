"""Shared pytest fixtures.

All fixtures force the offline backends (hashing embeddings + extractive
generator) so the suite is fast, deterministic, and needs no API keys or model
downloads.
"""
from __future__ import annotations

import pytest

from rag_assistant.config import (
    AppConfig,
    EmbeddingConfig,
    LLMConfig,
    PathConfig,
    RetrievalConfig,
)
from rag_assistant.pipeline import RAGPipeline

REPORTS = {
    "alpha.md": (
        "# Alpha Financials\n\n"
        "Revenue was $48.2M, up 12% year over year. "
        "Cloud subscriptions grew 24% while legacy licenses fell 9%."
    ),
    "beta.txt": (
        "Voluntary attrition was 14.2%. Engineering had the highest attrition at 18%. "
        "APAC was the fastest growing region at 31% year over year."
    ),
}


@pytest.fixture
def cfg(tmp_path):
    reports = tmp_path / "reports"
    index = tmp_path / "index"
    reports.mkdir()
    for name, content in REPORTS.items():
        (reports / name).write_text(content, encoding="utf-8")
    return AppConfig(
        embeddings=EmbeddingConfig(provider="hashing", hashing_dim=128),
        llm=LLMConfig(provider="extractive"),
        retrieval=RetrievalConfig(k=3, chunk_size=200, chunk_overlap=40),
        paths=PathConfig(reports_dir=str(reports), index_dir=str(index)),
    )


@pytest.fixture
def pipeline(cfg):
    rag = RAGPipeline(cfg)
    rag.build_index()
    return rag
