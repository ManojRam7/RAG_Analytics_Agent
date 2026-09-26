"""Embedding providers with graceful fallback.

Cascade for provider == "auto":
    sentence-transformers (if installed)  ->  Gemini (if GOOGLE_API_KEY)
    ->  OpenAI (if OPENAI_API_KEY)  ->  hashing.

``HashingEmbeddings`` is a stateless, dependency-light fallback (sklearn
HashingVectorizer). It needs no model download, so the whole pipeline runs
offline out of the box. It implements the LangChain ``Embeddings`` interface,
so it plugs straight into the FAISS vector store.
"""
from __future__ import annotations

import os
from typing import List, Tuple

from langchain_core.embeddings import Embeddings

from .config import AppConfig
from .utils import get_logger

logger = get_logger("embeddings")


class HashingEmbeddings(Embeddings):
    """Deterministic TF-style embeddings via sklearn's HashingVectorizer."""

    def __init__(self, n_features: int = 1024):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.n_features = n_features
        self._vec = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
            stop_words="english",
        )

    def _embed(self, texts: List[str]) -> List[List[float]]:
        matrix = self._vec.transform(texts)
        return matrix.toarray().astype("float32").tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(list(texts))

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]


def get_embeddings(cfg: AppConfig) -> Tuple[Embeddings, str]:
    """Return ``(embeddings, provider_name)`` based on config + availability."""
    provider = (cfg.embeddings.provider or "auto").lower()

    if provider == "hashing":
        return HashingEmbeddings(cfg.embeddings.hashing_dim), "hashing"

    if provider in ("auto", "huggingface"):
        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            emb = HuggingFaceEmbeddings(model_name=cfg.embeddings.model)
            logger.info("Using sentence-transformers embeddings: %s", cfg.embeddings.model)
            return emb, f"huggingface:{cfg.embeddings.model}"
        except Exception as exc:  # noqa: BLE001
            if provider == "huggingface":
                raise
            logger.info("sentence-transformers unavailable (%s); trying next option.", exc.__class__.__name__)

    if provider in ("auto", "gemini") and os.getenv("GOOGLE_API_KEY"):
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            model = "models/text-embedding-004"
            logger.info("Using Gemini embeddings: %s", model)
            return GoogleGenerativeAIEmbeddings(model=model), f"gemini:{model}"
        except Exception:  # noqa: BLE001
            if provider == "gemini":
                raise

    if provider in ("auto", "openai") and os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import OpenAIEmbeddings

            logger.info("Using OpenAI embeddings.")
            return OpenAIEmbeddings(), "openai"
        except Exception:  # noqa: BLE001
            if provider == "openai":
                raise

    logger.info("Falling back to dependency-free hashing embeddings.")
    return HashingEmbeddings(cfg.embeddings.hashing_dim), "hashing"
