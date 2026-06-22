"""Configuration loading.

Defaults live here, can be overridden by ``config.yaml`` at the project root,
and a few provider switches can be overridden again by environment variables.
Precedence (low -> high): dataclass defaults < config.yaml < environment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:  # optional, only used to load a local .env file
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

# repo root = two levels up from this file (src/rag_assistant/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


@dataclass
class EmbeddingConfig:
    provider: str = "auto"  # auto | huggingface | openai | hashing
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hashing_dim: int = 1024


@dataclass
class LLMConfig:
    provider: str = "auto"  # auto | anthropic | openai | huggingface | extractive
    model: str = ""
    temperature: float = 0.1
    max_tokens: int = 512


@dataclass
class RetrievalConfig:
    k: int = 4
    chunk_size: int = 800
    chunk_overlap: int = 120


@dataclass
class PathConfig:
    reports_dir: str = "data/reports"
    index_dir: str = "data/index"


@dataclass
class AppConfig:
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    @property
    def reports_path(self) -> Path:
        return _resolve(self.paths.reports_dir)

    @property
    def index_path(self) -> Path:
        return _resolve(self.paths.index_dir)


def _apply_env(cfg: AppConfig) -> AppConfig:
    if os.getenv("RAG_LLM_PROVIDER"):
        cfg.llm.provider = os.environ["RAG_LLM_PROVIDER"].strip().lower()
    if os.getenv("RAG_EMBEDDINGS_PROVIDER"):
        cfg.embeddings.provider = os.environ["RAG_EMBEDDINGS_PROVIDER"].strip().lower()
    if os.getenv("RAG_LLM_MODEL"):
        cfg.llm.model = os.environ["RAG_LLM_MODEL"].strip()
    return cfg


def load_config(path: Optional[str | Path] = None) -> AppConfig:
    """Build an :class:`AppConfig`, merging config.yaml and env overrides."""
    cfg = AppConfig()
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if yaml is not None and cfg_path.exists():
        data = yaml.safe_load(cfg_path.read_text()) or {}
        if "embeddings" in data:
            cfg.embeddings = EmbeddingConfig(**{**cfg.embeddings.__dict__, **data["embeddings"]})
        if "llm" in data:
            cfg.llm = LLMConfig(**{**cfg.llm.__dict__, **data["llm"]})
        if "retrieval" in data:
            cfg.retrieval = RetrievalConfig(**{**cfg.retrieval.__dict__, **data["retrieval"]})
        if "paths" in data:
            cfg.paths = PathConfig(**{**cfg.paths.__dict__, **data["paths"]})
    return _apply_env(cfg)
