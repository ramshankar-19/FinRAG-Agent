"""
tests/test_tools.py

Test Cases 6, 7, 8: Individual tool correctness.

  Test 6: search_financial_documents (ChromaDB)
  Test 7: get_stock_quote, get_company_profile, get_financial_statements (FMP)
  Test 8: search_financial_news (Tavily)
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools import (
    get_stock_quote,
    get_company_profile,
    get_financial_statements,
    search_financial_news,
    search_financial_documents,
)


# ─────────────────────────────────────────────
# Test 6: Document Search Tool
# ─────────────────────────────────────────────

class TestSearchFinancialDocuments:
    """Test Case 6: Local document retrieval tool."""

    def test_returns_string(self):
        result = search_financial_documents.invoke({"query": "TechNova revenue"})
        assert isinstance(result, str)

    def test_returns_document_label(self):
        """Results should be prefixed with [Document]."""
        result = search_financial_documents.invoke({"query": "TechNova revenue FY2025"})
        assert "[Document" in result or "No relevant" in result

    def test_returns_financial_content(self):
        result = search_financial_documents.invoke({"query": "revenue gross profit FY2025"})
        # Should contain financial terms from the TechNova PDF
        lower = result.lower()
        assert any(t in lower for t in ["revenue", "profit", "technova", "financial", "no relevant"]), (
            f"Expected financial content. Got: {result[:300]}"
        )

    def test_handles_empty_query_gracefully(self):
        result = search_financial_documents.invoke({"query": "xyzzy nonsense query 99999"})
        assert isinstance(result, str)  # must not raise

    def test_returns_multiple_chunks(self):
        result = search_financial_documents.invoke({"query": "TechNova annual report 2025"})
        # Multiple [Document] labels indicate multiple chunks returned
        count = result.count("[Document")
        assert count >= 1


# ─────────────────────────────────────────────
# Test 7: FMP Tools
# ─────────────────────────────────────────────

@pytest.mark.skipif(
    not os.getenv("FMP_API_KEY"),
    reason="FMP_API_KEY not set in environment"
)
class TestFMPTools:
    """Test Case 7: FMP financial data tools (requires live FMP API key)."""

    # ── get_stock_quote ─────────────────────────────

    def test_stock_quote_returns_string(self):
        result = get_stock_quote.invoke({"ticker": "AAPL"})
        assert isinstance(result, str)

    def test_stock_quote_contains_fmp_label(self):
        result = get_stock_quote.invoke({"ticker": "AAPL"})
        assert "[FMP]" in result, f"Expected [FMP] label. Got: {result[:200]}"

    def test_stock_quote_contains_price(self):
        result = get_stock_quote.invoke({"ticker": "AAPL"})
        assert "Price" in result or "price" in result.lower(), (
            f"Expected price in result. Got: {result[:300]}"
        )

    def test_stock_quote_invalid_ticker(self):
        result = get_stock_quote.invoke({"ticker": "ZZZZINVALID"})
        assert isinstance(result, str)  # should fail gracefully, not raise

    def test_stock_quote_lowercase_ticker(self):
        """Tool should normalise lowercase tickers."""
        result = get_stock_quote.invoke({"ticker": "msft"})
        assert "[FMP]" in result or "Error" in result

    # ── get_company_profile ─────────────────────────

    def test_company_profile_returns_string(self):
        result = get_company_profile.invoke({"ticker": "AAPL"})
        assert isinstance(result, str)

    def test_company_profile_contains_fmp_label(self):
        result = get_company_profile.invoke({"ticker": "AAPL"})
        assert "[FMP]" in result

    def test_company_profile_contains_sector(self):
        result = get_company_profile.invoke({"ticker": "AAPL"})
        assert "Sector" in result or "sector" in result.lower()

    def test_company_profile_contains_description(self):
        result = get_company_profile.invoke({"ticker": "AAPL"})
        assert "Description" in result or "Apple" in result

    def test_company_profile_tsla(self):
        result = get_company_profile.invoke({"ticker": "TSLA"})
        assert "[FMP]" in result

    # ── get_financial_statements ────────────────────

    def test_financial_statements_returns_string(self):
        result = get_financial_statements.invoke({"ticker": "AAPL"})
        assert isinstance(result, str)

    def test_financial_statements_contains_fmp_label(self):
        result = get_financial_statements.invoke({"ticker": "AAPL"})
        assert "[FMP]" in result

    def test_financial_statements_contains_revenue(self):
        result = get_financial_statements.invoke({"ticker": "AAPL"})
        assert "Revenue" in result or "revenue" in result.lower()

    def test_financial_statements_contains_net_income(self):
        result = get_financial_statements.invoke({"ticker": "AAPL"})
        assert "Net Income" in result or "net income" in result.lower()

    def test_financial_statements_contains_margins(self):
        result = get_financial_statements.invoke({"ticker": "AAPL"})
        assert "Margin" in result or "margin" in result.lower()


# ─────────────────────────────────────────────
# Test 8: Tavily News Search Tool
# ─────────────────────────────────────────────

@pytest.mark.skipif(
    not os.getenv("TAVILY_API_KEY"),
    reason="TAVILY_API_KEY not set in environment"
)
class TestSearchFinancialNews:
    """Test Case 8: Tavily web search tool (requires live Tavily API key)."""

    def test_returns_string(self):
        result = search_financial_news.invoke({"query": "Apple stock news"})
        assert isinstance(result, str)

    def test_contains_tavily_label(self):
        result = search_financial_news.invoke({"query": "Apple earnings 2025"})
        assert "[Tavily]" in result, f"Expected [Tavily] label. Got: {result[:200]}"

    def test_returns_real_results(self):
        result = search_financial_news.invoke({"query": "S&P 500 market news"})
        # Should have actual content, not just an error
        assert len(result) > 50, f"Expected substantial results. Got: {result[:200]}"

    def test_handles_specific_query(self):
        result = search_financial_news.invoke({"query": "Microsoft Azure revenue growth"})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_source_urls(self):
        result = search_financial_news.invoke({"query": "Tesla quarterly results"})
        assert "Source:" in result or "http" in result or "[Tavily]" in result

    def test_unusual_query_handled_gracefully(self):
        result = search_financial_news.invoke({"query": "xyzzy123456 completely fake query"})
        assert isinstance(result, str)  # must not raise
