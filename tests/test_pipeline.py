from rag_assistant.pipeline import RAGPipeline


def test_build_index_reports_stats(pipeline):
    # pipeline fixture already built the index
    assert pipeline.vs is not None


def test_query_returns_grounded_answer_with_citations(pipeline):
    result = pipeline.query("What was revenue and how did it change?")

    assert result["answer"]
    assert "[S" in result["answer"]  # contains an inline citation tag
    assert 1 <= len(result["sources"]) <= 3
    assert result["backend"]["llm"] == "extractive"
    assert result["backend"]["embeddings"] == "hashing"

    src = result["sources"][0]
    assert {"tag", "source", "score", "snippet"} <= set(src)
    assert "text" not in src  # heavy field stripped from public output


def test_index_persists_and_reloads(cfg):
    RAGPipeline(cfg).build_index()  # persist to disk
    reloaded = RAGPipeline(cfg).load_index()
    result = reloaded.query("Which region grew fastest?")
    assert result["sources"]
