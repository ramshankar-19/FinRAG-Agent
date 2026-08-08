"""
evaluate.py — Real RAGAs evaluation pipeline for Finance RAG.

This script:
  1. Defines a small golden evaluation dataset (7 questions from TechNova FY2025 PDF)
  2. Runs ACTUAL vector retrieval for each question → real contexts
  3. Generates ACTUAL answers from the LLM using retrieved context → real answers
  4. Feeds everything into ragas.evaluate() with 4 metrics
  5. Prints per-question and aggregate scores
  6. Saves results to data/ragas_results.csv

If aggregate score < 0.85:
  - Shows per-metric breakdown
  - Identifies the weakest metric
  - Suggests next improvement

NOTE: QUALITY_THRESHOLD = 0.85 is the INTERNAL self-critique gate in agent.py.
      The RAGAs score here is calculated independently from actual retrieval output.
      Do not conflate the two numbers.
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage

# ─────────────────────────────────────────────
# Golden evaluation dataset
# (all questions answerable from TechNova FY2025 PDF)
# ─────────────────────────────────────────────

EVAL_QUESTIONS = [
    {
        "question":    "What was TechNova's total revenue for FY2025?",
        "ground_truth": "TechNova's total revenue for FY2025 was $520 million.",
    },
    {
        "question":    "What was TechNova's gross profit for FY2025?",
        "ground_truth": "TechNova's gross profit for FY2025 was $315 million.",
    },
    {
        "question":    "What is TechNova's projected revenue growth for FY2026?",
        "ground_truth": (
            "Management expects TechNova's FY2026 revenue to grow between 14% and 18%."
        ),
    },
    {
        "question":    "What were TechNova's total operating expenses in FY2025?",
        "ground_truth": "TechNova's total operating expenses for FY2025 were $210 million.",
    },
    {
        "question":    "What was TechNova's net income for FY2025?",
        "ground_truth": "TechNova's net income for FY2025 was $105 million.",
    },
    {
        "question":    "What are TechNova's primary business risks?",
        "ground_truth": (
            "TechNova's primary business risks include intense market competition, "
            "regulatory changes, and talent retention challenges."
        ),
    },
    {
        "question":    "What is TechNova's primary industry?",
        "ground_truth": "TechNova operates in the Artificial Intelligence and Cloud Software industry.",
    },
]


# ─────────────────────────────────────────────
# RAG pipeline  (retrieval + generation)
# ─────────────────────────────────────────────

def run_rag_for_question(
    question: str,
    vectorstore: Chroma,
    llm: ChatGroq,
    k: int = 5,
) -> tuple[str, list[str]]:
    """
    Run the RAG pipeline for a single question.

    Returns:
        (answer, contexts)
        answer   — LLM-generated answer grounded in retrieved context
        contexts — list of raw retrieved chunk texts (used as RAGAs context field)
    """
    docs     = vectorstore.similarity_search(question, k=k)
    contexts = [doc.page_content for doc in docs]

    if not contexts:
        return "No relevant information found in the document.", []

    context_text = "\n\n".join(contexts)
    prompt = (
        f"You are a financial analyst. Answer the question using ONLY the context below.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Provide a specific, factual answer. "
        "If the context does not contain enough information, say so explicitly."
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content, contexts


# ─────────────────────────────────────────────
# Main evaluation function
# ─────────────────────────────────────────────

def run_evaluation(
    persist_directory: str = "data/chroma_db",
    output_csv: str = "data/ragas_results.csv",
    retrieval_k: int = 5,
) -> dict:
    """
    Run the full RAGAs evaluation pipeline.

    Returns a dict of metric_name → score.
    """
    print("=" * 55)
    print(" FinRAG - RAGAs Evaluation Pipeline")
    print("=" * 55)

    # ── Initialise models ─────────────────────────────
    print("\nInitialising models …")
    evaluator_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    embeddings    = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )
    doc_count = vectorstore._collection.count()
    print(f"  ChromaDB: {doc_count} chunks available")

    if doc_count == 0:
        raise RuntimeError(
            "ChromaDB is empty. Run: python ingest.py data/financial_report.pdf"
        )

    # ── Run actual RAG for each question ─────────────
    print(f"\nRunning RAG pipeline on {len(EVAL_QUESTIONS)} questions …\n")

    questions     = []
    answers       = []
    contexts_list = []
    ground_truths = []

    for idx, item in enumerate(EVAL_QUESTIONS, 1):
        q  = item["question"]
        gt = item["ground_truth"]

        print(f"  [{idx}/{len(EVAL_QUESTIONS)}] {q}")
        answer, contexts = run_rag_for_question(
            q, vectorstore, evaluator_llm, k=retrieval_k
        )
        print(f"        Answer (first 100 chars): {answer[:100].strip()} …")
        print(f"        Retrieved {len(contexts)} chunks")

        questions.append(q)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(gt)

    # ── Build RAGAs dataset ───────────────────────────
    dataset = Dataset.from_dict({
        "question":    questions,
        "answer":      answers,
        "contexts":    contexts_list,
        "ground_truth": ground_truths,
    })

    # ── Run RAGAs evaluation ──────────────────────────
    print("\nRunning RAGAs evaluation (LLM-as-judge) …")
    result = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
        llm=evaluator_llm,
        embeddings=embeddings,
    )

    # ── Display results ───────────────────────────────
    print("\n" + "=" * 55)
    print(" RAGAs Results")
    print("=" * 55)

    scores = {
        "context_precision": result["context_precision"],
        "context_recall":    result["context_recall"],
        "faithfulness":      result["faithfulness"],
        "answer_relevancy":  result["answer_relevancy"],
    }

    for metric, score in scores.items():
        bar  = "█" * int(score * 20)
        flag = "✓" if score >= 0.85 else "✗"
        print(f"  {flag} {metric:<22} {score:.4f}  {bar}")

    aggregate = sum(scores.values()) / len(scores)
    print(f"\n  Aggregate (mean):     {aggregate:.4f}")

    if aggregate >= 0.85:
        print("  CLAIM SUPPORTED: Aggregate score >= 0.85")
    else:
        print("  Aggregate score < 0.85 - see improvement suggestions below.")
        _suggest_improvements(scores)

    # ── Save CSV ──────────────────────────────────────
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = result.to_pandas()
    df["question"]     = questions
    df["answer"]       = answers
    df["ground_truth"] = ground_truths
    df.to_csv(output_csv, index=False)
    print(f"\n  Detailed results saved → {output_csv}")

    return scores


def _suggest_improvements(scores: dict) -> None:
    """Print targeted improvement suggestions based on which metric is lowest."""
    weakest = min(scores, key=scores.get)
    print(f"\n  Weakest metric: {weakest} ({scores[weakest]:.4f})")

    suggestions = {
        "context_precision": (
            "Reduce retrieval k or use MMR retrieval to return more focused chunks."
        ),
        "context_recall": (
            "Increase retrieval k, add more chunks per document, "
            "or improve chunking overlap."
        ),
        "faithfulness": (
            "Tighten the answer-generation system prompt to prohibit claims "
            "not grounded in context. Increase self-critique MAX_ITERATIONS."
        ),
        "answer_relevancy": (
            "Improve the question-answering prompt to stay on topic. "
            "Consider using a query-rewriting step."
        ),
    }
    print(f"  Suggestion: {suggestions.get(weakest, 'Review the pipeline.')}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    scores = run_evaluation()

    # Save scores as JSON for inspection
    with open("data/ragas_scores.json", "w") as f:
        json.dump(scores, f, indent=2)
    print("\nScores also saved → data/ragas_scores.json")