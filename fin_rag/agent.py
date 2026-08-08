"""
agent.py — Stateful ReAct agent with self-critique refinement loop.

LangGraph graph topology:
    START
      │
      ▼
    [agent]  ←──────────────────────────┐
      │                                 │
      ├── tool calls? ──YES──► [tools] ─┘
      │
      NO (done with tools)
      │
      ▼
    [generate_draft]
      │
      ▼
    [critique_answer]  ◄──────────────────┐
      │                                   │
      ├── score >= 0.85 OR               │
      │   iterations >= 3  ──► [finalize] │
      │                                   │
      └── else ──────────► [refine_answer]┘
                                  │
                                  └──────────────────────►┘ (loops back)
    [finalize]
      │
      ▼
     END

Self-critique configuration:
    QUALITY_THRESHOLD = 0.85   (internal critique pass/fail gate)
    MAX_ITERATIONS    = 3      (hard cap on refinement loops)

Note: QUALITY_THRESHOLD is the internal refinement gate.
      It is NOT the RAGAs benchmark score (evaluated separately in evaluate.py).
"""

import os
import json
import re
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import (
    BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import tools

load_dotenv()

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

QUALITY_THRESHOLD = 0.85   # internal self-critique threshold
MAX_ITERATIONS    = 3      # max refinement loops before forced finalization

# ─────────────────────────────────────────────
# State schema
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    messages:     Annotated[list[BaseMessage], add_messages]
    query:        str    # original user question, preserved throughout
    tool_results: str    # concatenated raw tool outputs
    draft:        str    # current draft answer being refined
    critique:     str    # critique text from last evaluation
    score:        float  # quality score 0.0–1.0 from last critique
    iterations:   int    # number of refinement cycles completed
    final_answer: str    # approved final answer


# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────

llm            = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# ─────────────────────────────────────────────
# Node 1: Agent — reason and select tools
# ─────────────────────────────────────────────

_AGENT_SYSTEM = SystemMessage(content=(
    "You are a precise financial research assistant with access to five tools:\n"
    "  • search_financial_documents — search the user's uploaded reports/files\n"
    "  • get_stock_quote            — current stock price (FMP)\n"
    "  • get_company_profile        — company background (FMP)\n"
    "  • get_financial_statements   — revenue, earnings, margins (FMP)\n"
    "  • search_financial_news      — latest news and developments (Tavily)\n\n"
    "Rules:\n"
    "  - Only call tools that are genuinely needed for the question.\n"
    "  - Do NOT call every tool for every question.\n"
    "  - Stop calling tools once you have sufficient information.\n"
    "  - For questions about uploaded documents, use search_financial_documents.\n"
    "  - For current market data, use the FMP tools.\n"
    "  - For news/recent events, use search_financial_news.\n"
    "  - For questions combining documents and live data, use both."
))


def agent_node(state: AgentState) -> dict:
    """Reasoning engine: decides which tools (if any) to call."""
    messages  = [_AGENT_SYSTEM] + state["messages"]
    response  = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def route_after_agent(state: AgentState) -> str:
    """Route to tools if tool calls present, else proceed to draft generation."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "generate_draft"


# ─────────────────────────────────────────────
# Node 2: Tools — execute selected tools
# ─────────────────────────────────────────────

tool_node = ToolNode(tools)


# ─────────────────────────────────────────────
# Node 3: Generate Draft
# ─────────────────────────────────────────────

def generate_draft(state: AgentState) -> dict:
    """
    Aggregate all tool outputs from the message history and produce a
    first draft answer grounded in that context.
    """
    # Collect all ToolMessage outputs from the conversation so far
    tool_outputs = [
        msg.content
        for msg in state["messages"]
        if isinstance(msg, ToolMessage)
    ]
    tool_results = "\n\n".join(tool_outputs) if tool_outputs else "No tool results."

    query = state.get("query", "")

    prompt = (
        f"Based on the information gathered below, write a clear factual answer.\n\n"
        f"Question: {query}\n\n"
        f"Gathered Information:\n{tool_results}\n\n"
        "Instructions:\n"
        "  - Be specific and cite your sources using [Document], [FMP], or [Tavily] labels.\n"
        "  - Do not add any information not present in the gathered information above.\n"
        "  - If information is missing, clearly state what could not be found."
    )

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "draft":        response.content,
        "tool_results": tool_results,
        "iterations":   0,
        "score":        0.0,
        "critique":     "",
    }


# ─────────────────────────────────────────────
# Node 4: Critique Answer
# ─────────────────────────────────────────────

_CRITIQUE_SYSTEM = (
    "You are a strict quality evaluator for financial research answers.\n\n"
    "Score the draft answer on four criteria (each 0.0 to 1.0):\n"
    "  1. Faithfulness   — every claim is directly supported by the context\n"
    "  2. Relevance      — the answer directly addresses the question asked\n"
    "  3. Completeness   — important information in the context is not omitted\n"
    "  4. Accuracy       — no hallucinated or unsupported statements\n\n"
    "Compute the average of all four as the final score.\n\n"
    "Return ONLY a valid JSON object — no other text:\n"
    '{"score": <float 0.0-1.0>, "critique": "<specific issues, or \'Answer meets quality standards.\' if score >= 0.85>"}'
)


def critique_answer(state: AgentState) -> dict:
    """Evaluate the current draft and return a score + critique."""
    query        = state.get("query", "")
    tool_results = state.get("tool_results", "")
    draft        = state.get("draft", "")
    iterations   = state.get("iterations", 0)

    user_msg = (
        f"Question: {query}\n\n"
        f"Available Context:\n{tool_results}\n\n"
        f"Draft Answer:\n{draft}"
    )

    response = llm.invoke([
        SystemMessage(content=_CRITIQUE_SYSTEM),
        HumanMessage(content=user_msg),
    ])

    raw = response.content.strip()

    # Robustly parse JSON from response
    score    = 0.5
    critique = raw
    try:
        match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
        parsed = json.loads(match.group() if match else raw)
        score    = float(parsed.get("score", 0.5))
        critique = parsed.get("critique", raw)
    except (json.JSONDecodeError, AttributeError, ValueError):
        # Fallback: attempt to pull score from raw text
        m = re.search(r'"score"\s*:\s*([0-9.]+)', raw)
        if m:
            score = float(m.group(1))

    status = "✓ PASS" if score >= QUALITY_THRESHOLD else "✗ REFINE"
    print(
        f"  [Critique] Iteration {iterations + 1}/{MAX_ITERATIONS} | "
        f"Score: {score:.2f} | {status}"
    )
    if critique and "meets quality" not in critique.lower():
        print(f"  [Critique] {critique[:200]}")

    return {
        "score":      score,
        "critique":   critique,
        "iterations": iterations + 1,
    }


def route_after_critique(state: AgentState) -> str:
    """Finalize if score passes threshold or max iterations reached; else refine."""
    score      = state.get("score", 0.0)
    iterations = state.get("iterations", 0)

    if score >= QUALITY_THRESHOLD or iterations >= MAX_ITERATIONS:
        return "finalize"
    return "refine_answer"


# ─────────────────────────────────────────────
# Node 5: Refine Answer
# ─────────────────────────────────────────────

def refine_answer(state: AgentState) -> dict:
    """Rewrite the draft using critique feedback."""
    query        = state.get("query", "")
    tool_results = state.get("tool_results", "")
    draft        = state.get("draft", "")
    critique     = state.get("critique", "")

    prompt = (
        f"Rewrite the following financial research answer based on the quality feedback.\n\n"
        f"Question: {query}\n\n"
        f"Available Context:\n{tool_results}\n\n"
        f"Current Draft:\n{draft}\n\n"
        f"Quality Critique:\n{critique}\n\n"
        "Instructions:\n"
        "  - Directly address every issue raised in the critique.\n"
        "  - Only use information from the context above.\n"
        "  - Cite sources with [Document], [FMP], or [Tavily] labels.\n"
        "  - Be specific and factual."
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    print(f"  [Refine] Rewrote answer to address critique.")

    return {"draft": response.content}


# ─────────────────────────────────────────────
# Node 6: Finalize
# ─────────────────────────────────────────────

def finalize(state: AgentState) -> dict:
    """Accept the current draft as the final answer."""
    score      = state.get("score", 0.0)
    iterations = state.get("iterations", 0)
    draft      = state.get("draft", "")

    reason = "threshold met" if score >= QUALITY_THRESHOLD else "max iterations reached"
    print(f"  [Finalize] Score={score:.2f} | Iterations={iterations} | Reason: {reason}")

    return {
        "final_answer": draft,
        "messages":     [AIMessage(content=draft)],
    }


# ─────────────────────────────────────────────
# Build LangGraph
# ─────────────────────────────────────────────

builder = StateGraph(AgentState)

builder.add_node("agent",          agent_node)
builder.add_node("tools",          tool_node)
builder.add_node("generate_draft", generate_draft)
builder.add_node("critique_answer",critique_answer)
builder.add_node("refine_answer",  refine_answer)
builder.add_node("finalize",       finalize)

builder.add_edge(START,            "agent")
builder.add_conditional_edges(
    "agent", route_after_agent, ["tools", "generate_draft"]
)
builder.add_edge("tools",          "agent")
builder.add_edge("generate_draft", "critique_answer")
builder.add_conditional_edges(
    "critique_answer", route_after_critique, ["refine_answer", "finalize"]
)
builder.add_edge("refine_answer",  "critique_answer")
builder.add_edge("finalize",       END)

finrag_agent = builder.compile()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print(" 🚀 FinRAG Agent — ReAct + Self-Critique (LangGraph)")
    print(f" Quality threshold : {QUALITY_THRESHOLD}  |  Max iterations: {MAX_ITERATIONS}")
    print(" Type 'exit' to quit.")
    print("=" * 60)

    while True:
        try:
            user_query = input("\nUser Query > ").strip()

            if user_query.lower() in ["exit", "quit"]:
                print("\nGoodbye!")
                break

            if not user_query:
                continue

            print("\n" + "-" * 60)

            initial_state = {
                "messages":     [HumanMessage(content=user_query)],
                "query":        user_query,
                "tool_results": "",
                "draft":        "",
                "critique":     "",
                "score":        0.0,
                "iterations":   0,
                "final_answer": "",
            }

            result = finrag_agent.invoke(initial_state)

            print("\n📋 FINAL ANSWER:")
            print(result.get("final_answer", "No answer generated."))
            print(
                f"\n[Quality Score: {result.get('score', 0.0):.2f} | "
                f"Refinement Iterations: {result.get('iterations', 0)}]"
            )
            print("-" * 60)

        except KeyboardInterrupt:
            print("\nSession terminated.")
            break