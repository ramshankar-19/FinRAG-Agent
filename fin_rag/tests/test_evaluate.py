"""
tests/test_evaluate.py

Test Case 11: RAGAs evaluation pipeline correctness.

Tests that:
  - Evaluation dataset is structured correctly.
  - run_rag_for_question retrieves contexts and generates an answer.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluate import EVAL_QUESTIONS, run_rag_for_question

class TestEvaluatePipeline:
    """Test Case 11: RAGAs evaluation pipeline."""

    def test_eval_dataset_has_questions(self):
        """Evaluation dataset must have questions and ground truth."""
        assert len(EVAL_QUESTIONS) > 0
        for item in EVAL_QUESTIONS:
            assert "question" in item
            assert "ground_truth" in item

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")),
        reason="ChromaDB not found — run: python ingest.py data/financial_report.pdf"
    )
    def test_run_rag_for_question(self):
        """run_rag_for_question should return an answer and a list of contexts."""
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_groq import ChatGroq

        db_dir = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory=db_dir, embedding_function=embeddings)
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

        answer, contexts = run_rag_for_question(
            "What was TechNova's total revenue for FY2025?",
            vectorstore,
            llm,
            k=2
        )

        assert isinstance(answer, str)
        assert len(answer) > 0
        assert isinstance(contexts, list)
        assert len(contexts) > 0
