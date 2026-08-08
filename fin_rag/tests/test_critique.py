"""
tests/test_critique.py

Test Case 10: Self-critique refinement loop.

Tests that:
  - critique_answer node scores correctly
  - refine_answer node rewrites the draft
  - The loop terminates at MAX_ITERATIONS
  - The loop terminates early when score >= QUALITY_THRESHOLD
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agent import (
    critique_answer,
    refine_answer,
    generate_draft,
    finalize,
    route_after_critique,
    QUALITY_THRESHOLD,
    MAX_ITERATIONS,
)


def _base_state(
    query: str = "What was TechNova's revenue for FY2025?",
    draft: str = "",
    tool_results: str = "",
    score: float = 0.0,
    iterations: int = 0,
    critique: str = "",
) -> dict:
    """Build a minimal state dict for testing individual nodes."""
    return {
        "messages":     [HumanMessage(content=query)],
        "query":        query,
        "tool_results": tool_results,
        "draft":        draft,
        "critique":     critique,
        "score":        score,
        "iterations":   iterations,
        "final_answer": "",
    }


# ─────────────────────────────────────────────
# Test 10: Self-Critique Loop
# ─────────────────────────────────────────────

class TestCritiqueNode:
    """Unit tests for the critique_answer node."""

    def test_critique_returns_score(self):
        """critique_answer must return a numeric score."""
        state  = _base_state(
            query        = "What was TechNova's revenue for FY2025?",
            draft        = "TechNova's revenue for FY2025 was $520 million.",
            tool_results = "Revenue: $520 million for FY2025.",
        )
        result = critique_answer(state)
        assert "score" in result
        assert isinstance(result["score"], float)

    def test_critique_score_in_valid_range(self):
        """Score must be between 0.0 and 1.0."""
        state  = _base_state(
            query        = "What is TechNova's net income?",
            draft        = "TechNova's net income for FY2025 was $105 million.",
            tool_results = "Net income: $105 million.",
        )
        result = critique_answer(state)
        assert 0.0 <= result["score"] <= 1.0

    def test_critique_returns_critique_text(self):
        """critique_answer must return a critique text string."""
        state  = _base_state(
            query        = "What is TechNova's gross profit?",
            draft        = "I don't know.",
            tool_results = "Gross Profit: $315 million.",
        )
        result = critique_answer(state)
        assert "critique" in result
        assert isinstance(result["critique"], str)

    def test_critique_increments_iterations(self):
        """critique_answer must increment the iterations counter."""
        state  = _base_state(iterations=1)
        result = critique_answer(state)
        assert result["iterations"] == 2

    def test_good_answer_gets_high_score(self):
        """A well-grounded, specific answer should score higher than a vague one."""
        good_state = _base_state(
            query        = "What was TechNova's revenue?",
            draft        = "According to the FY2025 report, TechNova's total revenue was $520 million, representing a 16.9% increase from FY2024's $445 million.",
            tool_results = "Revenue: $520 million for FY2025. Prior year: $445 million.",
        )
        bad_state = _base_state(
            query        = "What was TechNova's revenue?",
            draft        = "The revenue was something related to millions of dollars.",
            tool_results = "Revenue: $520 million for FY2025.",
        )
        good_result = critique_answer(good_state)
        bad_result  = critique_answer(bad_state)
        assert good_result["score"] >= bad_result["score"], (
            f"Good answer ({good_result['score']:.2f}) should score >= bad answer ({bad_result['score']:.2f})"
        )

    def test_hallucinated_answer_gets_lower_score(self):
        """An answer with unsupported claims should score lower."""
        factual   = _base_state(
            query        = "What was TechNova's revenue?",
            draft        = "TechNova revenue was $520 million in FY2025.",
            tool_results = "Revenue: $520 million.",
        )
        hallucinated = _base_state(
            query        = "What was TechNova's revenue?",
            draft        = "TechNova revenue was $5 billion in FY2025 and they acquired three companies.",
            tool_results = "Revenue: $520 million.",
        )
        factual_result      = critique_answer(factual)
        hallucinated_result = critique_answer(hallucinated)
        assert factual_result["score"] >= hallucinated_result["score"]


class TestRefineNode:
    """Unit tests for the refine_answer node."""

    def test_refine_returns_updated_draft(self):
        """refine_answer must return a new draft string."""
        state  = _base_state(
            query        = "What is TechNova's gross profit?",
            draft        = "I'm not sure about the gross profit.",
            tool_results = "Gross Profit: $315 million for FY2025.",
            critique     = "The answer does not use the available context. Context states gross profit is $315 million.",
        )
        result = refine_answer(state)
        assert "draft" in result
        assert isinstance(result["draft"], str)
        assert len(result["draft"]) > 0

    def test_refined_draft_incorporates_critique(self):
        """Refined draft should address the critique (incorporate missing data)."""
        state  = _base_state(
            query        = "What is TechNova's gross profit?",
            draft        = "The report discusses finances.",
            tool_results = "Gross Profit: $315 million for FY2025.",
            critique     = "The answer is too vague. The context clearly states gross profit is $315 million.",
        )
        result = refine_answer(state)
        # The refined answer should be more specific
        new_draft = result["draft"].lower()
        assert "315" in new_draft or "gross profit" in new_draft or "million" in new_draft, (
            f"Refined draft should incorporate the missing data. Got: {result['draft'][:200]}"
        )


class TestRoutingLogic:
    """Tests for route_after_critique routing decision."""

    def test_routes_to_finalize_when_threshold_met(self):
        state = _base_state(score=QUALITY_THRESHOLD, iterations=1)
        route = route_after_critique(state)
        assert route == "finalize"

    def test_routes_to_finalize_when_score_above_threshold(self):
        state = _base_state(score=0.95, iterations=1)
        route = route_after_critique(state)
        assert route == "finalize"

    def test_routes_to_refine_when_below_threshold(self):
        state = _base_state(score=0.5, iterations=1)
        route = route_after_critique(state)
        assert route == "refine_answer"

    def test_routes_to_finalize_when_max_iterations_reached(self):
        """Even with low score, should finalize after MAX_ITERATIONS."""
        state = _base_state(score=0.3, iterations=MAX_ITERATIONS)
        route = route_after_critique(state)
        assert route == "finalize", (
            f"Should finalize at max iterations ({MAX_ITERATIONS}), even with score={0.3}"
        )

    def test_routes_to_refine_when_below_threshold_and_under_max(self):
        state = _base_state(score=0.7, iterations=MAX_ITERATIONS - 1)
        route = route_after_critique(state)
        assert route == "refine_answer"


class TestCritiqueCycleIntegration:
    """Integration test: full critique → refine cycle."""

    def test_critique_then_refine_improves_draft(self):
        """Run one full critique → refine cycle and verify draft is updated."""
        initial_draft = "The document mentions some financial information."
        context       = "TechNova FY2025 Revenue: $520 million. Gross Profit: $315 million."
        query         = "What was TechNova's revenue for FY2025?"

        state = _base_state(
            query        = query,
            draft        = initial_draft,
            tool_results = context,
            iterations   = 0,
        )

        # Step 1: Critique
        critique_result = critique_answer(state)
        state.update(critique_result)

        # Step 2: If below threshold, refine
        if state["score"] < QUALITY_THRESHOLD:
            refine_result = refine_answer(state)
            state.update(refine_result)
            assert state["draft"] != initial_draft, "Draft should have changed after refinement"

        # Step 3: Verify iterations incremented
        assert state["iterations"] >= 1
