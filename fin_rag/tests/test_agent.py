"""
tests/test_agent.py

Test Cases 5 and 9: LangGraph agent execution and multi-tool reasoning.

  Test 5:  Agent runs end-to-end and returns a final answer.
  Test 9:  Agent routes to correct tool(s) based on query type.
           - Document question  → search_financial_documents
           - Stock price        → get_stock_quote
           - News question      → search_financial_news
           - Combined question  → multiple tools
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from agent import finrag_agent, AgentState


def _make_initial_state(query: str) -> dict:
    """Create a fresh initial state for the agent."""
    return {
        "messages":     [HumanMessage(content=query)],
        "query":        query,
        "tool_results": "",
        "draft":        "",
        "critique":     "",
        "score":        0.0,
        "iterations":   0,
        "final_answer": "",
    }


def _get_tool_names_used(result: dict) -> list[str]:
    """Extract the names of all tools that were called in the agent run."""
    tool_names = []
    for msg in result.get("messages", []):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_names.append(tc.get("name", ""))
    return tool_names


# ─────────────────────────────────────────────
# Test 5: Agent Execution
# ─────────────────────────────────────────────

class TestAgentExecution:
    """Test Case 5: LangGraph agent runs and produces a final answer."""

    def test_agent_returns_final_answer(self):
        """Agent must always produce a non-empty final_answer."""
        state  = _make_initial_state("What was TechNova's revenue for FY2025?")
        result = finrag_agent.invoke(state)
        assert result.get("final_answer"), "final_answer must be non-empty"

    def test_agent_final_answer_is_string(self):
        state  = _make_initial_state("Tell me about TechNova.")
        result = finrag_agent.invoke(state)
        assert isinstance(result.get("final_answer"), str)

    def test_agent_has_critique_score(self):
        """Agent must have run through the critique loop and set a score."""
        state  = _make_initial_state("What is TechNova's gross profit?")
        result = finrag_agent.invoke(state)
        score  = result.get("score", None)
        assert score is not None, "score must be set after critique loop"
        assert 0.0 <= score <= 1.0, f"score must be 0-1, got {score}"

    def test_agent_has_iteration_count(self):
        """iterations must be >= 1 after running."""
        state  = _make_initial_state("What is TechNova's net income?")
        result = finrag_agent.invoke(state)
        assert result.get("iterations", 0) >= 1

    def test_agent_state_contains_query(self):
        """Original query must be preserved in final state."""
        query  = "What are TechNova's business risks?"
        state  = _make_initial_state(query)
        result = finrag_agent.invoke(state)
        assert result.get("query") == query

    def test_agent_messages_populated(self):
        """Message history must contain at least the initial human message + final AI message."""
        state  = _make_initial_state("Summarise TechNova's financial position.")
        result = finrag_agent.invoke(state)
        msgs   = result.get("messages", [])
        assert len(msgs) >= 2, "Should have at least HumanMessage + final AIMessage"


# ─────────────────────────────────────────────
# Test 9: Multi-tool Reasoning / Routing
# ─────────────────────────────────────────────

class TestMultiToolReasoning:
    """Test Case 9: Agent selects correct tool(s) based on query intent."""

    def test_document_question_uses_doc_tool(self):
        """
        'What does the uploaded report say about revenue?'
        → should call search_financial_documents
        """
        state  = _make_initial_state(
            "What does the uploaded financial report say about TechNova's revenue?"
        )
        result = finrag_agent.invoke(state)
        tools_used = _get_tool_names_used(result)
        assert "search_financial_documents" in tools_used, (
            f"Expected search_financial_documents. Tools used: {tools_used}"
        )

    def test_stock_price_question_uses_fmp(self):
        """
        'What is Apple's current stock price?'
        → should call get_stock_quote (not search_financial_documents)
        """
        state  = _make_initial_state("What is Apple's current stock price?")
        result = finrag_agent.invoke(state)
        tools_used = _get_tool_names_used(result)

        fmp_tools = {"get_stock_quote", "get_company_profile", "get_financial_statements"}
        used_fmp  = bool(fmp_tools.intersection(set(tools_used)))

        assert used_fmp, (
            f"Expected an FMP tool for stock price. Tools used: {tools_used}"
        )

    def test_news_question_uses_tavily(self):
        """
        'What are the latest news about Apple's AI strategy?'
        → should call search_financial_news
        """
        state  = _make_initial_state(
            "What are the latest news developments about Apple's AI strategy?"
        )
        result = finrag_agent.invoke(state)
        tools_used = _get_tool_names_used(result)
        assert "search_financial_news" in tools_used, (
            f"Expected search_financial_news. Tools used: {tools_used}"
        )

    def test_combined_question_uses_multiple_tools(self):
        """
        'Compare TechNova's revenue with Apple's latest revenue.'
        → should call both search_financial_documents AND an FMP tool
        """
        state = _make_initial_state(
            "Compare the revenue reported in the uploaded TechNova report "
            "with Apple's latest reported revenue."
        )
        result     = finrag_agent.invoke(state)
        tools_used = _get_tool_names_used(result)

        fmp_tools  = {"get_stock_quote", "get_company_profile", "get_financial_statements"}
        used_doc   = "search_financial_documents" in tools_used
        used_fmp   = bool(fmp_tools.intersection(set(tools_used)))

        assert used_doc and used_fmp, (
            f"Expected both doc retrieval and FMP. Tools used: {tools_used}"
        )

    def test_agent_does_not_call_all_tools_unnecessarily(self):
        """
        A simple document-only question should NOT call every tool.
        """
        state      = _make_initial_state(
            "What does the uploaded report say about TechNova's net income in FY2025?"
        )
        result     = finrag_agent.invoke(state)
        tools_used = _get_tool_names_used(result)

        # For a pure document question, Tavily should NOT be called
        assert "search_financial_news" not in tools_used, (
            f"Should not use Tavily for a document-only question. Tools: {tools_used}"
        )
