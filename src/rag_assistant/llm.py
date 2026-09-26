"""Answer generators with a graceful cascade.

Provider == "auto" resolves to the best available backend:
    Gemini (if GOOGLE_API_KEY)  ->  OpenAI (if OPENAI_API_KEY)
    ->  local flan-t5 (if transformers installed)  ->  extractive (always works).

Every generator exposes the same tiny interface::

    generate(question: str, context_str: str, sources: list[dict]) -> str

so the pipeline does not care which backend is active.
"""
from __future__ import annotations

import os
from typing import Dict, List

from .config import AppConfig
from .prompts import RAG_PROMPT, SYSTEM_INSTRUCTIONS
from .utils import get_logger, split_sentences

logger = get_logger("llm")

ABSTAIN = "I could not find this in the provided reports."


class BaseGenerator:
    name = "base"

    def generate(self, question: str, context_str: str, sources: List[Dict]) -> str:
        raise NotImplementedError


class LLMGenerator(BaseGenerator):
    """Wraps any LangChain chat/LLM runnable behind the common interface."""

    def __init__(self, llm, name: str):
        self.llm = llm
        self.name = name

    def generate(self, question: str, context_str: str, sources: List[Dict]) -> str:
        prompt = RAG_PROMPT.format(
            system=SYSTEM_INSTRUCTIONS, context=context_str, question=question
        )
        resp = self.llm.invoke(prompt)
        text = getattr(resp, "content", resp)
        return str(text).strip()


class ExtractiveGenerator(BaseGenerator):
    """LLM-free fallback: builds a source-grounded summary by ranking the report
    sentences most relevant to the question (TF-IDF cosine) and citing each one.

    It means the assistant still returns a grounded, cited answer with no API
    key and no model downloads.
    """

    name = "extractive"

    def __init__(self, max_sentences: int = 5, min_score: float = 0.06):
        self.max_sentences = max_sentences
        self.min_score = min_score

    def generate(self, question: str, context_str: str, sources: List[Dict]) -> str:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        candidates = []  # (sentence, tag)
        for s in sources:
            for sent in split_sentences(s.get("text", "")):
                if len(sent.split()) >= 4:
                    candidates.append((sent, s["tag"]))
        if not candidates:
            return ABSTAIN

        sentences = [c[0] for c in candidates]
        try:
            vec = TfidfVectorizer(stop_words="english").fit(sentences + [question])
            sims = cosine_similarity(vec.transform([question]), vec.transform(sentences))[0]
        except ValueError:
            return ABSTAIN

        ranked = [i for i in np.argsort(-sims) if sims[i] >= self.min_score][: self.max_sentences]
        if not ranked:
            return ABSTAIN

        # keep original reading order for a coherent summary
        chosen = sorted(ranked)
        parts = [f"{candidates[i][0]} [{candidates[i][1]}]" for i in chosen]
        return "Based on the retrieved reports: " + " ".join(parts)


def _build_local_hf(cfg: AppConfig) -> LLMGenerator:
    from langchain_huggingface import HuggingFacePipeline
    from transformers import pipeline

    model = cfg.llm.model or "google/flan-t5-base"
    logger.info("Loading local HF model %s (first run downloads weights)…", model)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        max_new_tokens=cfg.llm.max_tokens,
        temperature=max(cfg.llm.temperature, 1e-3),
        do_sample=cfg.llm.temperature > 0,
    )
    return LLMGenerator(HuggingFacePipeline(pipeline=pipe), f"huggingface:{model}")


def get_generator(cfg: AppConfig) -> BaseGenerator:
    provider = (cfg.llm.provider or "auto").lower()

    if provider == "extractive":
        return ExtractiveGenerator()

    if provider in ("auto", "gemini") and os.getenv("GOOGLE_API_KEY"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            model = cfg.llm.model or "gemini-2.0-flash"
            llm = ChatGoogleGenerativeAI(
                model=model, temperature=cfg.llm.temperature, max_output_tokens=cfg.llm.max_tokens
            )
            logger.info("Using Gemini backend: %s", model)
            return LLMGenerator(llm, f"gemini:{model}")
        except Exception:  # noqa: BLE001
            if provider == "gemini":
                raise

    if provider in ("auto", "openai") and os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI

            model = cfg.llm.model or "gpt-4o-mini"
            llm = ChatOpenAI(
                model=model, temperature=cfg.llm.temperature, max_tokens=cfg.llm.max_tokens
            )
            logger.info("Using OpenAI backend: %s", model)
            return LLMGenerator(llm, f"openai:{model}")
        except Exception:  # noqa: BLE001
            if provider == "openai":
                raise

    if provider in ("auto", "huggingface"):
        try:
            return _build_local_hf(cfg)
        except Exception as exc:  # noqa: BLE001
            if provider == "huggingface":
                raise
            logger.info("Local HF model unavailable (%s); using extractive fallback.", exc.__class__.__name__)

    logger.info("Using extractive (LLM-free) answer engine.")
    return ExtractiveGenerator()
