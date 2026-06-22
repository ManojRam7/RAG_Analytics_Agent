"""Ask the assistant a single question from the command line.

    python scripts/query.py "What drove revenue growth in Q3?"
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rag_assistant.pipeline import RAGPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Query business reports with source-grounded answers.")
    parser.add_argument("question", nargs="+", help="The question to ask.")
    parser.add_argument("-k", type=int, default=None, help="Number of chunks to retrieve.")
    args = parser.parse_args()
    question = " ".join(args.question)

    rag = RAGPipeline()
    result = rag.query(question, k=args.k)

    print("\n" + "=" * 70)
    print(f"Q: {question}")
    print("=" * 70)
    print(f"\n{result['answer']}\n")
    print("-" * 70)
    print("Sources")
    for s in result["sources"]:
        page = f" p.{s['page'] + 1}" if isinstance(s.get("page"), int) else ""
        print(f"  [{s['tag']}] {s['source']}{page}  (score={s['score']:.3f})")
        print(f"        {s['snippet']}")
    print("-" * 70)
    print(f"backend: llm={result['backend']['llm']} | embeddings={result['backend']['embeddings']}")


if __name__ == "__main__":
    main()
