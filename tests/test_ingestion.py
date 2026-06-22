from rag_assistant.ingestion import chunk_documents, load_documents


def test_load_documents_reads_supported_files(tmp_path):
    (tmp_path / "a.md").write_text("# Title\n\nSome content about revenue.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Plain text about attrition.", encoding="utf-8")
    (tmp_path / "ignore.csv").write_text("x,y\n1,2", encoding="utf-8")  # unsupported

    docs = load_documents(tmp_path)

    sources = {d.metadata["source"] for d in docs}
    assert sources == {"a.md", "b.txt"}
    assert all("source" in d.metadata for d in docs)


def test_chunk_documents_adds_ids_and_preserves_source(tmp_path):
    (tmp_path / "long.md").write_text("Sentence one. " * 200, encoding="utf-8")
    docs = load_documents(tmp_path)

    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(c.metadata["source"] == "long.md" for c in chunks)
    assert [c.metadata["chunk_id"] for c in chunks] == list(range(len(chunks)))
