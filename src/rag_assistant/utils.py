"""Small shared helpers: sentence splitting, cosine similarity, logging."""
from __future__ import annotations

import logging
import re
from typing import List

import numpy as np

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    """Naive but dependency-free sentence splitter."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def cosine(a, b) -> float:
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


_LOGGER_CONFIGURED = False


def get_logger(name: str = "rag") -> logging.Logger:
    global _LOGGER_CONFIGURED
    if not _LOGGER_CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        _LOGGER_CONFIGURED = True
    return logging.getLogger(name)
