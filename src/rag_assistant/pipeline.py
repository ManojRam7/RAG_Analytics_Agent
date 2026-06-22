"""The end-to-end RAG pipeline: ingest -> embed -> index -> retrieve -> generate."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from . import vectorstore as vs_mod
from .config import AppConfig, load_config
from .embeddings import get_embeddings
from .ingestion import chunk_documents, load_documents
from .llm import get_generator
from .prompts import format_context
from .utils import get_logger

logger = get_logger("pipeline")


class RAGPipeline:
    """High-level orchestrator tying every component together.

    Typical use::

        rag = RAGPipeline()
        rag.build_index()                 # one-off, persists to data/index
        result = rag.query("What drove the revenue change?")
        print(result["answer"])           # cites [S1], [S2], …
        print(result["sources"])          # where each citation came from
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.cfg = config or load_config()
        self.embeddings, self.embeddings_name = get_embeddings(self.cfg)
        self.generator = get_generator(self.cfg)
        self.vs = None  # lazily built/loaded FAISS store

    # ------------------------------------------------------------------ index
    def build_index(self, reports_dir: Optional[str | Path] = None, persist: bool = True) -> Dict[str, Any]:
        reports_dir = Path(reports_dir or self.cfg.reports_path)
        docs = load_documents(reports_dir)
        if not docs:
            raise FileNotFoundError(
                f"No supported documents found in {reports_dir}. "
                "Generate samples first: python scripts/generate_sample_reports.py"
            )
        chunks = chunk_documents(
            docs, self.cfg.retrieval.chunk_size, self.cfg.retrieval.chunk_overlap
        )
        self.vs = vs_mod.build_vectorstore(chunks, self.embeddings)
        if persist:
            vs_mod.save_vectorstore(self.vs, self.cfg.index_path)
            logger.info("Saved index to %s", self.cfg.index_path)
        return {
            "documents": len({d.metadata.get("source") for d in docs}),
            "sections": len(docs),
            "chunks": len(chunks),
            "index_dir": str(self.cfg.index_path),
        }

    def load_index(self) -> "RAGPipeline":
        if not vs_mod.index_exists(self.cfg.index_path):
            raise FileNotFoundError(
                f"No index at {self.cfg.index_path}. Build it first: python scripts/ingest.py"
            )
        self.vs = vs_mod.load_vectorstore(self.cfg.index_path, self.embeddings)
        return self

    def ensure_index(self) -> "RAGPipeline":
        if self.vs is None:
            if vs_mod.index_exists(self.cfg.index_path):
                self.load_index()
            else:
                logger.info("No index found; building from %s", self.cfg.reports_path)
                self.build_index()
        return self

    # ------------------------------------------------------------------ query
    def retrieve(self, question: str, k: Optional[int] = None) -> List[Tuple[Document, float]]:
        self.ensure_index()
        k = k or self.cfg.retrieval.k
        return self.vs.similarity_search_with_score(question, k=k)

    def query(self, question: str, k: Optional[int] = None) -> Dict[str, Any]:
        scored = self.retrieve(question, k=k)
        context_str, sources = format_context(scored)
        answer = self.generator.generate(question, context_str, sources)
        # strip the heavy full-text field from what we return to callers/UI
        public_sources = [{kk: vv for kk, vv in s.items() if kk != "text"} for s in sources]
        return {
            "question": question,
            "answer": answer,
            "sources": public_sources,
            "backend": {"llm": self.generator.name, "embeddings": self.embeddings_name},
        }
