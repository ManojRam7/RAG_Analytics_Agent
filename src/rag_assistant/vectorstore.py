"""FAISS vector-store helpers (build / save / load)."""
from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def build_vectorstore(chunks: List[Document], embeddings: Embeddings) -> FAISS:
    return FAISS.from_documents(chunks, embeddings)


def save_vectorstore(vs: FAISS, index_dir: str | Path) -> None:
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_dir))


def load_vectorstore(index_dir: str | Path, embeddings: Embeddings) -> FAISS:
    # allow_dangerous_deserialization is required for local pickle load; safe here
    # because we created the index ourselves.
    return FAISS.load_local(
        str(index_dir), embeddings, allow_dangerous_deserialization=True
    )


def index_exists(index_dir: str | Path) -> bool:
    p = Path(index_dir)
    return (p / "index.faiss").exists() and (p / "index.pkl").exists()
