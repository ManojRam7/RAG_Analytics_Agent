"""Run the offline RAG evaluation and write a results report.

    python scripts/evaluate.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rag_assistant.evaluation import evaluate  # noqa: E402
from rag_assistant.pipeline import RAGPipeline  # noqa: E402

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval + grounding on the eval set.")
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "eval" / "eval_dataset.json"))
    parser.add_argument("-k", type=int, default=None)
    args = parser.parse_args()

    rag = RAGPipeline()
    report = evaluate(rag, args.dataset, k=args.k)
    summary = report["summary"]

    print("\n=== RAG Evaluation ===")
    print(f"backend: llm={summary['backend']['llm']} | embeddings={summary['backend']['embeddings']}")
    print(f"questions evaluated : {summary['n']}")
    print(f"recall@k            : {summary['recall@k']}")
    print(f"groundedness        : {summary['groundedness']}")
    print(f"answer_relevance    : {summary['answer_relevance']}")
    print(f"abstention_rate     : {summary['abstention_rate']}")

    out_json = PROJECT_ROOT / "eval" / "results.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"\nDetailed results written to {out_json}")

    try:
        import pandas as pd

        out_csv = PROJECT_ROOT / "eval" / "results.csv"
        pd.DataFrame(report["rows"]).to_csv(out_csv, index=False)
        print(f"Per-question table written to {out_csv}")
    except Exception:  # pragma: no cover - pandas optional for this step
        pass


if __name__ == "__main__":
    main()
