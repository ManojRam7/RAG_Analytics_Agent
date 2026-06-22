"""Build (or rebuild) the FAISS index from the reports in data/reports."""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rag_assistant.pipeline import RAGPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the FAISS index from business reports.")
    parser.add_argument("--reports-dir", default=None, help="Override reports directory.")
    args = parser.parse_args()

    rag = RAGPipeline()
    print(f"Embeddings backend: {rag.embeddings_name}")
    stats = rag.build_index(reports_dir=args.reports_dir)
    print(
        f"Indexed {stats['documents']} document(s) -> {stats['chunks']} chunks. "
        f"Saved to {stats['index_dir']}"
    )


if __name__ == "__main__":
    main()
