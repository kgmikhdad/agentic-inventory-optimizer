from __future__ import annotations

from pathlib import Path
from typing import List, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimplePolicyRetriever:
    """Lightweight local RAG retriever using TF-IDF.

    This keeps the project runnable without paid APIs. Later, this can be replaced
    by LlamaIndex, ChromaDB, or embedding-based retrieval.
    """

    def __init__(self, docs_dir: str | Path = "docs") -> None:
        self.docs_dir = Path(docs_dir)
        self.documents = self._load_documents()
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([doc["text"] for doc in self.documents]) if self.documents else None

    def _load_documents(self) -> List[Dict[str, str]]:
        docs = []
        if not self.docs_dir.exists():
            return docs
        for path in sorted(self.docs_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
            for idx, chunk in enumerate(chunks):
                docs.append({"source": f"{path.name}#chunk-{idx+1}", "text": chunk})
        return docs

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        if not self.documents or self.matrix is None:
            return []
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).ravel()
        ranked = scores.argsort()[::-1][:top_k]
        results = []
        for idx in ranked:
            doc = self.documents[int(idx)].copy()
            doc["score"] = float(scores[int(idx)])
            results.append(doc)
        return results
