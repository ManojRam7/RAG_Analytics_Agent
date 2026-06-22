import json

from rag_assistant.evaluation import evaluate


def test_evaluate_produces_summary(pipeline, tmp_path):
    dataset = [
        {"question": "What was revenue?", "expected_source": "alpha.md"},
        {"question": "What is the CEO's home address?", "expected_source": None},
    ]
    ds_path = tmp_path / "eval.json"
    ds_path.write_text(json.dumps(dataset))

    report = evaluate(pipeline, ds_path)
    summary = report["summary"]

    assert summary["n"] == 2
    assert len(report["rows"]) == 2
    for key in ("recall@k", "groundedness", "answer_relevance", "abstention_rate"):
        assert key in summary
    # recall@k is a proportion when defined
    r = summary["recall@k"]
    assert (r != r) or (0.0 <= r <= 1.0)  # nan or in [0,1]
