"""Lightweight, API-free RAG evaluation.

Three signals, all computable locally with the active embedding model:

* **recall@k**         did retrieval surface the expected source document?
* **groundedness**     fraction of answer sentences that are semantically close
                       to at least one retrieved chunk (a faithfulness proxy).
* **answer_relevance** cosine similarity between the question and the answer.

It also reports the **abstention rate** (how often the assistant correctly says
it cannot find the answer), which matters for a source-grounded system.

For heavier, LLM-judged metrics, RAGAS can be dropped in later — see the README.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .llm import ABSTAIN
from .pipeline import RAGPipeline
from .utils import cosine, split_sentences


def recall_at_k(sources: List[Dict], expected_source: Optional[str]) -> float:
    if not expected_source:
        return float("nan")
    return 1.0 if any(s.get("source") == expected_source for s in sources) else 0.0


def groundedness(embeddings, answer: str, snippets: List[str], threshold: float = 0.35) -> float:
    sents = [s for s in split_sentences(answer) if len(s.split()) >= 4]
    if not sents or not snippets or answer.startswith(ABSTAIN):
        return float("nan")
    ctx = [np.asarray(v, dtype="float32") for v in embeddings.embed_documents(snippets)]
    grounded = 0
    for s in sents:
        sv = np.asarray(embeddings.embed_query(s), dtype="float32")
        if max(cosine(sv, c) for c in ctx) >= threshold:
            grounded += 1
    return grounded / len(sents)


def answer_relevance(embeddings, question: str, answer: str) -> float:
    if not answer.strip() or answer.startswith(ABSTAIN):
        return float("nan")
    qv = embeddings.embed_query(question)
    av = embeddings.embed_query(answer)
    return cosine(qv, av)


def _mean(rows: List[Dict], key: str) -> float:
    vals = [r[key] for r in rows if isinstance(r[key], (int, float)) and r[key] == r[key]]
    return round(sum(vals) / len(vals), 3) if vals else float("nan")


def evaluate(pipeline: RAGPipeline, dataset_path: str | Path, k: Optional[int] = None) -> Dict[str, Any]:
    data = json.loads(Path(dataset_path).read_text())
    rows: List[Dict] = []
    for item in data:
        res = pipeline.query(item["question"], k=k)
        snippets = [s["snippet"] for s in res["sources"]]
        rows.append(
            {
                "question": item["question"],
                "answer": res["answer"],
                "expected_source": item.get("expected_source"),
                "recall@k": recall_at_k(res["sources"], item.get("expected_source")),
                "groundedness": groundedness(pipeline.embeddings, res["answer"], snippets),
                "answer_relevance": answer_relevance(pipeline.embeddings, item["question"], res["answer"]),
                "abstained": res["answer"].startswith(ABSTAIN),
            }
        )
    summary = {
        "n": len(rows),
        "recall@k": _mean(rows, "recall@k"),
        "groundedness": _mean(rows, "groundedness"),
        "answer_relevance": _mean(rows, "answer_relevance"),
        "abstention_rate": round(sum(r["abstained"] for r in rows) / len(rows), 3) if rows else float("nan"),
        "backend": {"llm": pipeline.generator.name, "embeddings": pipeline.embeddings_name},
    }
    return {"summary": summary, "rows": rows}
