
"""Model-guided orchestration for the Document Intelligence Agent."""
import re
import json
from dataclasses import dataclass

from app.agent_tools import execute_tool
from app.document_loader import DocumentResult
from app.llm_client import ask_llm
from app.qa_pipeline import answer_question


VALID_TOOLS = {
    "analytics",
    "chart",
    "document_qa",
}


@dataclass
class AgentResult:
    """Structured result returned by the agent."""

    answer: str
    tool_used: str


def _extract_tool_from_response(response: str) -> str | None:
    """Extract a valid tool decision from a JSON model response."""

    if not isinstance(response, str):
        return None

    decoder = json.JSONDecoder()

    # Search for a JSON object, even if the model wraps it
    # in Markdown or explanatory text.
    for index, character in enumerate(response):
        if character != "{":
            continue

        try:
            payload, _ = decoder.raw_decode(response[index:])
        except json.JSONDecodeError:
            continue

        if not isinstance(payload, dict):
            continue

        selected = payload.get("tool")

        if isinstance(selected, str) and selected in VALID_TOOLS:
            return selected

    return None


def _is_chart_intent(question: str) -> bool:
    """Detect explicit chart and trend visualization requests."""

    q = question.lower()

    chart_terms = (
        "chart",
        "charts",
        "graph",
        "plot",
        "visualize",
        "visualise",
        "histogram",
        "distribution",
        "trend",
        "trends",
        "over time",
        "time series",
        "month by month",
        "day by day",
        "week by week",
        "year by year",
    )

    return any(term in q for term in chart_terms)



def _fallback_tool(question: str) -> str:
    """Choose a tool deterministically if model routing fails."""

    q = question.lower().strip()

    if _is_chart_intent(question):
        return "chart"

    analytics_terms = (
        "total",
        "sum",
        "average",
        "mean",
        "maximum",
        "minimum",
        "highest",
        "lowest",
        "how many",
        "count",
        "compare",
        "calculate",
        "revenue",
        "sales",
        "units",
        "monthlycharges",
        "tenure",
    )

    # Match complete words or phrases, not arbitrary substrings.
    if any(
        re.search(r"\b" + re.escape(term) + r"\b", q)
        for term in analytics_terms
    ):
        return "analytics"

    return "document_qa"


def _is_document_overview_question(question: str) -> bool:
    """Identify questions that clearly request a document overview."""

    q = question.lower().strip()

    overview_terms = (
        "summarize",
        "summarise",
        "give me a summary",
        "provide a summary",
        "document summary",
        "report summary",
        "summarize the report",
        "summarise the report",
        "summarize this report",
        "summarise this report",
        "summarize the document",
        "summarise the document",
        "summarize this document",
        "summarise this document",
        "what is this document about",
        "what is the document about",
        "what is this report about",
        "what is the report about",
        "provide an overview",
        "give me an overview",
        "main points of this document",
        "key points of this document",
        "key findings of this document",
    )

    return any(term in q for term in overview_terms)


def _choose_tool(question: str) -> str:
    print("QUESTION:", repr(question))
    print("OVERVIEW MATCH:", _is_document_overview_question(question))
    print("FALLBACK TOOL:", _fallback_tool(question))
    """Select the best tool based on the user's intent."""
    # Skip the LLM router for obvious document overview questions.
    if _is_document_overview_question(question):
        print("ROUTING: document_qa (overview)")
        return "document_qa"
    # Explicit chart and trend requests bypass LLM routing.
    if _is_chart_intent(question):
        print("ROUTING: chart")
        return "chart"

    prompt = f"""
You are the routing component of a document intelligence agent.

Choose exactly one tool for the user's question.

Available tools:

1. analytics
   Use for calculations on structured CSV and Excel data.
   Examples:
   - Calculate total revenue.
   - Find average monthly charges.
   - Count the number of customers.
   - Find the highest sales value.

2. chart
   Use when the user requests a chart, graph, plot, histogram,
   visualization, trend analysis, or time-series visualization.

   Examples:
   - Create a revenue chart.
   - Show monthly revenue trends.
   - Plot sales over time.

3. document_qa
   Use for document summaries, explanations, text questions,
   and questions requiring document context or retrieval.

Important rules:
- Do not answer the user's question.
- Do not invent information.
- Choose only one of the three tools.
- Return exactly one JSON object.
- Do not include Markdown or explanations.

Examples:

Question: What is the total revenue?
Response: {{"tool": "analytics"}}

Question: Summarize this document.
Response: {{"tool": "document_qa"}}

Question: Create a revenue chart.
Response: {{"tool": "chart"}}

User question: {question}
"""

    try:
        response = ask_llm(prompt)
        selected = _extract_tool_from_response(response)

        if selected:
            return selected

    except Exception:
        # Use deterministic routing if the local LLM is unavailable
        # or returns an invalid response.
        pass

    return _fallback_tool(question)



def run_agent(
    document: DocumentResult,
    question: str,
) -> AgentResult:
    """Route a question to the appropriate document tool."""

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    # Route PDF numerical questions directly to analytics.
    if (
        document.file_type.lower() == "pdf"
        and not _is_chart_intent(question)
        and _fallback_tool(question) == "analytics"
    ):
        selected_tool = "analytics"
    else:
        selected_tool = _choose_tool(question)

    if selected_tool not in VALID_TOOLS:
        selected_tool = _fallback_tool(question)

    print("QUESTION:", question)
    print("SELECTED TOOL:", selected_tool)
    print(
        "OVERVIEW DETECTED:",
        _is_document_overview_question(question),
    )

    if selected_tool == "document_qa":
        result = answer_question(document, question)
        answer = result.answer

    else:
        try:
            answer = execute_tool(document, question)
        except ValueError as exc:
            if selected_tool == "chart":
                answer = f"I couldn't create the requested chart: {exc}"
            else:
                raise

    return AgentResult(
        answer=answer,
        tool_used=selected_tool,
    )
