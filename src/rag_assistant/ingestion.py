"""Document loading and chunking.

Loads .txt/.md/.pdf/.docx from a directory into LangChain ``Document`` objects,
then splits them into overlapping chunks for retrieval. Every chunk keeps
``source`` (file name) and (for PDFs) ``page`` metadata so answers can cite them.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .utils import get_logger

logger = get_logger("ingestion")

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


def _load_one(path: Path) -> List[Document]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        from langchain_community.document_loaders import TextLoader

        return TextLoader(str(path), encoding="utf-8").load()
    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        return PyPDFLoader(str(path)).load()
    if suffix == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader

        return Docx2txtLoader(str(path)).load()
    return []


def load_documents(reports_dir: str | Path) -> List[Document]:
    """Recursively load every supported document under ``reports_dir``."""
    reports_dir = Path(reports_dir)
    docs: List[Document] = []
    for path in sorted(reports_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            loaded = _load_one(path)
        except Exception as exc:  # noqa: BLE001 - skip unreadable files, keep going
            logger.warning("Could not load %s: %s", path.name, exc)
            continue
        for d in loaded:
            d.metadata["source"] = path.name
            d.metadata.setdefault("path", str(path))
        docs.extend(loaded)
    logger.info("Loaded %d document section(s) from %s", len(docs), reports_dir)
    return docs


def chunk_documents(
    docs: List[Document], chunk_size: int = 800, chunk_overlap: int = 120
) -> List[Document]:
    """Split documents into overlapping chunks and tag each with a ``chunk_id``."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    logger.info("Split into %d chunk(s)", len(chunks))
    return chunks
