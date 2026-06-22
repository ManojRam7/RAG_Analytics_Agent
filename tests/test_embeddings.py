from rag_assistant.config import AppConfig, EmbeddingConfig
from rag_assistant.embeddings import HashingEmbeddings, get_embeddings


def test_hashing_embeddings_shape_and_determinism():
    emb = HashingEmbeddings(n_features=64)
    vecs = emb.embed_documents(["hello world revenue", "attrition report"])
    assert len(vecs) == 2
    assert all(len(v) == 64 for v in vecs)

    q1 = emb.embed_query("hello world revenue")
    q2 = emb.embed_query("hello world revenue")
    assert len(q1) == 64
    assert q1 == q2  # deterministic


def test_get_embeddings_hashing_provider():
    cfg = AppConfig(embeddings=EmbeddingConfig(provider="hashing", hashing_dim=32))
    emb, name = get_embeddings(cfg)
    assert name == "hashing"
    assert len(emb.embed_query("test")) == 32
