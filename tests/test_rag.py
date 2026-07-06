from pathlib import Path

from src.rag.simple_rag import SimplePolicyRetriever


def test_retriever_returns_context(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "policy.md").write_text("High-priority products require high service level.", encoding="utf-8")
    retriever = SimplePolicyRetriever(docs)
    results = retriever.retrieve("high priority service level", top_k=1)
    assert len(results) == 1
    assert "service level" in results[0]["text"]
