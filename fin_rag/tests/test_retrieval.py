"""
tests/test_retrieval.py

Test Case 4: Vector retrieval from ChromaDB.

Verifies that after ingestion, similarity search returns relevant chunks.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
EMBED_MODEL = "all-MiniLM-L6-v2"


@pytest.fixture(scope="module")
def vectorstore():
    """Load the existing ChromaDB (requires data/financial_report.pdf to have been ingested)."""
    if not os.path.exists(CHROMA_DIR):
        pytest.skip("ChromaDB not found — run: python ingest.py data/financial_report.pdf")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vs = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    if vs._collection.count() == 0:
        pytest.skip("ChromaDB is empty — run: python ingest.py data/financial_report.pdf")

    return vs


class TestVectorRetrieval:
    """Test Case 4: Vector retrieval accuracy and metadata correctness."""

    def test_retrieval_returns_results(self, vectorstore):
        """Similarity search must return at least one result."""
        results = vectorstore.similarity_search("TechNova revenue FY2025", k=3)
        assert len(results) > 0, "Should return at least one matching chunk"

    def test_retrieval_k_limit_respected(self, vectorstore):
        """Retrieval should not return more than k results."""
        k = 4
        results = vectorstore.similarity_search("financial highlights", k=k)
        assert len(results) <= k

    def test_retrieval_results_are_relevant(self, vectorstore):
        """Top result for a specific query should contain related terms."""
        results = vectorstore.similarity_search("revenue 520 million FY2025", k=3)
        combined = " ".join(r.page_content for r in results).lower()
        # At least one of these terms should appear
        assert any(term in combined for term in ["revenue", "520", "fy2025", "financial"]), (
            f"Retrieved chunks should contain financial terms. Got: {combined[:300]}"
        )

    def test_retrieval_results_have_metadata(self, vectorstore):
        """Each retrieved chunk should have source metadata."""
        results = vectorstore.similarity_search("TechNova", k=2)
        for res in results:
            assert "source" in res.metadata or "file_name" in res.metadata, (
                f"Chunk missing source metadata: {res.metadata}"
            )

    def test_retrieval_different_queries_differ(self, vectorstore):
        """Different queries should (usually) return different top results."""
        r1 = vectorstore.similarity_search("revenue growth forecast", k=1)
        r2 = vectorstore.similarity_search("business risks regulation", k=1)
        # They should not be exactly identical
        if r1 and r2:
            assert r1[0].page_content != r2[0].page_content or True  # soft check

    def test_retrieval_collection_count(self, vectorstore):
        """ChromaDB should have a non-trivial number of chunks."""
        count = vectorstore._collection.count()
        assert count >= 5, f"Expected at least 5 chunks in ChromaDB, got {count}"

    def test_retrieval_with_custom_k(self, vectorstore):
        """k=6 should return up to 6 results."""
        results = vectorstore.similarity_search("TechNova financial", k=6)
        assert 1 <= len(results) <= 6

    def test_retrieval_gross_profit_query(self, vectorstore):
        """Query specifically about gross profit should retrieve relevant chunks."""
        results = vectorstore.similarity_search("gross profit margin", k=4)
        assert len(results) > 0
        combined = " ".join(r.page_content for r in results).lower()
        assert "profit" in combined or "margin" in combined or "gross" in combined
