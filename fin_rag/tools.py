"""
tools.py — LangChain tool definitions for the Finance RAG agent.

Three data-source categories:
  FMP (Financial Modeling Prep — live financial data):
    - get_stock_quote()
    - get_company_profile()
    - get_financial_statements()

  Tavily (web search):
    - search_financial_news()

  ChromaDB (local uploaded documents):
    - search_financial_documents()

All API keys are read from environment variables.
"""

import os
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from tavily import TavilyClient

load_dotenv()

# ─────────────────────────────────────────────
# FMP helpers
# ─────────────────────────────────────────────

FMP_BASE = "https://financialmodelingprep.com/stable"


def _fmp_request(endpoint: str, params: dict):
    """
    Internal FMP API caller using the stable/ endpoint base.
    Raises ValueError if API key is missing.
    Raises requests.HTTPError on non-2xx responses.
    """
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        raise ValueError(
            "FMP_API_KEY environment variable is not set. "
            "Obtain a free key at https://financialmodelingprep.com/developer/docs/"
        )
    params = {**params, "apikey": api_key}
    response = requests.get(f"{FMP_BASE}/{endpoint}", params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def _fmt_large(n) -> str:
    """Format a large number as $XB / $XM / $X."""
    if n is None:
        return "N/A"
    try:
        n = float(n)
    except (TypeError, ValueError):
        return str(n)
    if abs(n) >= 1e9:
        return f"${n/1e9:.2f}B"
    if abs(n) >= 1e6:
        return f"${n/1e6:.2f}M"
    return f"${n:,.0f}"


# ─────────────────────────────────────────────
# FMP Tool 1 — Stock Quote
# ─────────────────────────────────────────────

@tool
def get_stock_quote(ticker: str) -> str:
    """
    Get the current real-time stock quote for a ticker symbol using FMP.
    Returns price, intraday change, volume, and market cap.
    Use when the user asks about current stock price or recent price movement.
    ticker examples: 'AAPL', 'TSLA', 'MSFT', 'GOOGL'
    """
    ticker = ticker.strip().upper()
    try:
        data = _fmp_request("quote", {"symbol": ticker})
        if not data or not isinstance(data, list):
            return f"[FMP] No quote data found for ticker: {ticker}"

        q = data[0]
        change_pct = q.get("changePercentage", 0) or 0
        return (
            f"[FMP] Real-time Quote — {q.get('name', ticker)} ({q.get('symbol', ticker)})\n"
            f"  Price:       ${q.get('price', 'N/A')}\n"
            f"  Change:      {q.get('change', 'N/A')} ({change_pct:.2f}%)\n"
            f"  Day High:    ${q.get('dayHigh', 'N/A')}\n"
            f"  Day Low:     ${q.get('dayLow', 'N/A')}\n"
            f"  Volume:      {q.get('volume', 'N/A'):,}\n"
            f"  Market Cap:  {_fmt_large(q.get('marketCap'))}"
        )
    except Exception as exc:
        return f"[FMP] Error fetching quote for {ticker}: {exc}"


# ─────────────────────────────────────────────
# FMP Tool 2 — Company Profile
# ─────────────────────────────────────────────

@tool
def get_company_profile(ticker: str) -> str:
    """
    Get detailed company profile from FMP: sector, industry, CEO, headcount,
    headquarters, exchange, and a brief business description.
    Use when the user asks what a company does or wants background information.
    ticker examples: 'AAPL', 'JPM', 'NVDA'
    """
    ticker = ticker.strip().upper()
    try:
        data = _fmp_request("profile", {"symbol": ticker})
        if not data or not isinstance(data, list):
            return f"[FMP] No profile found for ticker: {ticker}"

        p = data[0]
        description = (p.get("description") or "N/A")[:400]
        employees = p.get("fullTimeEmployees")
        emp_str = f"{int(employees):,}" if employees else "N/A"

        return (
            f"[FMP] Company Profile — {p.get('companyName', ticker)}\n"
            f"  Symbol:       {p.get('symbol', ticker)}\n"
            f"  Exchange:     {p.get('exchangeShortName', 'N/A')}\n"
            f"  Sector:       {p.get('sector', 'N/A')}\n"
            f"  Industry:     {p.get('industry', 'N/A')}\n"
            f"  CEO:          {p.get('ceo', 'N/A')}\n"
            f"  Employees:    {emp_str}\n"
            f"  HQ:           {p.get('city', '')}, {p.get('country', '')}\n"
            f"  Market Cap:   {_fmt_large(p.get('marketCap'))}\n"
            f"  Description:  {description}…"
        )
    except Exception as exc:
        return f"[FMP] Error fetching profile for {ticker}: {exc}"


# ─────────────────────────────────────────────
# FMP Tool 3 — Financial Statements
# ─────────────────────────────────────────────

@tool
def get_financial_statements(ticker: str) -> str:
    """
    Get the latest annual financial statements for a company from FMP:
    revenue, gross profit, operating income, net income, EPS,
    P/E ratio, gross margin, and net margin.
    Use when the user asks about revenue, earnings, profit, or financial performance.
    ticker examples: 'AAPL', 'AMZN', 'META'
    """
    ticker = ticker.strip().upper()
    try:
        income  = _fmp_request("income-statement", {"symbol": ticker, "limit": 1})
        metrics = _fmp_request("key-metrics",       {"symbol": ticker, "limit": 1})

        if not income or not isinstance(income, list):
            return f"[FMP] No financial statement data found for ticker: {ticker}"

        i = income[0]
        m = metrics[0] if metrics and isinstance(metrics, list) else {}

        fiscal_label = i.get("fiscalYear") or i.get("date", "N/A")

        gpm = m.get("grossProfitMargin") or 0
        npm = m.get("netProfitMargin")   or 0

        return (
            f"[FMP] Financial Statements — {ticker} (FY {fiscal_label})\n"
            f"  Revenue:           {_fmt_large(i.get('revenue'))}\n"
            f"  Gross Profit:      {_fmt_large(i.get('grossProfit'))}\n"
            f"  Operating Income:  {_fmt_large(i.get('operatingIncome'))}\n"
            f"  Net Income:        {_fmt_large(i.get('netIncome'))}\n"
            f"  EPS:               ${i.get('eps', 'N/A')}\n"
            f"  P/E Ratio:         {m.get('peRatio', 'N/A')}\n"
            f"  Gross Margin:      {gpm*100:.1f}%\n"
            f"  Net Margin:        {npm*100:.1f}%"
        )
    except Exception as exc:
        return f"[FMP] Error fetching financial statements for {ticker}: {exc}"


# ─────────────────────────────────────────────
# Tavily — Web Search
# ─────────────────────────────────────────────

@tool
def search_financial_news(query: str) -> str:
    """
    Search the web for the latest financial news, analyst commentary, and
    recent developments using Tavily.
    Use when the user asks about news, events, analyst views, or anything
    requiring up-to-date web information.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "[Tavily] Error: TAVILY_API_KEY environment variable not set."

    try:
        client   = TavilyClient(api_key=api_key)
        response = client.search(query=query, search_depth="basic", max_results=3)
        results  = response.get("results", [])

        if not results:
            return "[Tavily] No relevant results found for this query."

        parts = []
        for res in results:
            title   = res.get("title", "No title")
            content = (res.get("content") or "")[:350]
            url     = res.get("url", "")
            parts.append(f"[Tavily] {title}\n  {content}\n  Source: {url}")

        return "\n\n".join(parts)

    except Exception as exc:
        return f"[Tavily] Error: {exc}"


# ─────────────────────────────────────────────
# ChromaDB — Local Document Search
# ─────────────────────────────────────────────

@tool
def search_financial_documents(query: str) -> str:
    """
    Search locally uploaded financial reports and documents (PDFs, DOCX, images)
    that have been indexed in the vector database.
    Use when the user asks about content in their uploaded files or local reports.
    """
    try:
        embeddings  = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(
            persist_directory="data/chroma_db",
            embedding_function=embeddings,
        )

        results = vectorstore.similarity_search(query, k=6)
        if not results:
            return "[Document] No relevant information found in uploaded documents."

        parts = []
        for res in results:
            src = res.metadata.get("file_name") or res.metadata.get("source", "unknown")
            parts.append(f"[Document | {src}]\n{res.page_content}")

        return "\n\n".join(parts)

    except Exception as exc:
        return f"[Document] Error searching documents: {exc}"


# ─────────────────────────────────────────────
# Tool list — imported by agent.py
# ─────────────────────────────────────────────

tools = [
    get_stock_quote,
    get_company_profile,
    get_financial_statements,
    search_financial_news,
    search_financial_documents,
]