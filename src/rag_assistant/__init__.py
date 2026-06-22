"""RAG Analytics Assistant for Business Reports.

A retrieval-augmented generation pipeline that answers questions about a corpus
of business reports and returns *source-grounded* summaries with inline [S#]
citations.

Design goals:
  * Runs end-to-end for free, offline, with zero API keys (graceful fallbacks).
  * Upgrades transparently to local LLMs or cloud APIs when available.
  * Small, readable, well-separated modules.
"""

from .config import AppConfig, load_config
from .pipeline import RAGPipeline

__all__ = ["AppConfig", "load_config", "RAGPipeline"]
__version__ = "0.1.0"
