import os 
import sqlite3
import warnings
from typing import Annotated

from oauthlib.uri_validate import query

warnings.filterwarnings("ignore", message = "The default value of `allowed_objects`")

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import  AIMessage , ToolMessage , HumanMessage
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.tools import tool , InjectedToolCallId
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, MessagesState , StateGraph
from langgraph.prebuilt import InjectedState , tools_condition , ToolNode
from langgraph.types import Command
from pydantic import BaseModel , Field
from tavily import TavilyClient


from backend.models import ClaimVerificationResult , RelevancyDecision , RouterDecision
from backend.vector_store import search as vs_search

load_dotenv()

llm = ChatOpenAI(model = "gpt-5.4-mini")

class RAGState(MessageState):
    session_id: str
    query : str
    route: str | None
    retrieved_docs = list[Document]
    retrieval_attempts : int
    claim_verdict : str | None
    claim_source : str | None
    superseding_papers: list[dict] | None
    answer : str | None
    is_relevant: bool | None
    rewrite_query : str| None
    
    
    
    
ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    (
    "system",
        "You are a routing assistant for a research paper Q&A system. "
        "Classify the user query into exactly one of three categories:\n\n"
        "  retrieve — Use this for TWO types of questions:\n"
        "    (a) Questions about the content of uploaded research papers "
        "(e.g. methods, results, conclusions, authors).\n"
        "    (b) Questions that require live or current information that cannot be "
        "answered from general knowledge alone — such as current events, today's weather, "
        "live prices, recent news, or anything where the answer changes over time "
        "(e.g. 'Who is the current president?', 'What is the price of gold today?', "
        "'What is the weather in Delhi?').\n"
        "  verify_claim — The user wants to check whether a specific claim or finding "
        "from a paper is still accurate or has been superseded.\n"
        "  direct_answer — A stable general knowledge question answerable from training data "
        "with no retrieval needed (e.g. 'What is softmax?', 'Who invented the transformer?', "
        "'Explain backpropagation.').\n\n"
        "When in doubt between retrieve and direct_answer, prefer retrieve.\n\n"
        "Return only the route field.",
    ),
    ("human" , "{query}"),
])

router_chain = ROUTER_PROMPT| llm.with_structured_output(RouterDecision)

def router_node(state: RAGState) -> dict:
    dict = state["messages"][-1].content
    decision : RouterDecision = router_chain.invoke({"query": query})
    return {"route": decision.route}


# ── Tool schemas ──────────────────────────────────────────────────────────────

class RetrieverInput(BaseModel):
    query: str = Field(..., description = "Semantic query to search research paper chunks")
    k: int = Field(default = 4 , ge = 1 , le = 10 , description="Number of chunks to retrieve"  )


class WebSearchInput(BaseModel):
    optimised_query: str = Field(description = "Query rewritten and optimized for web search")
    max_results: int = Field(default = 3 , ge = 1 , le = 10, description = "Maximum number of web search results to return")
    
    


# ── Tools ─────────────────────────────────────────────────────────────────────
@tool(args_schema = RetrieverInput)
def retrieve_from_vectorstore(
    query: str,
    k: int,
    session_id: Annotated[str, InjectedState("session_id")],
    current_docs: Annotated[list, InjectedState("retrieved_docs")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> list:
    """Search the uploaded research paper vector store for relevant passages."""
    docs = vs_search(query=query, session_id=session_id, k=k)
    if not docs:
        return [ToolMessage(content="No relevant documents found in the vector store.", tool_call_id=tool_call_id)]
    summary = f"Retrieved {len(docs)} chunk(s) from the vector store."
    return [
        ToolMessage(content=summary, tool_call_id=tool_call_id),
        Command(update={"retrieved_docs": (current_docs or []) + docs}),
    ]
    
    
@tool(args_schema= WebSearchInput)
def web_search(
    optimised_query: str,
    max_results: int,
    current_docs : Annotated[list , InjectedState("retrieved_docs")],
    tool_call_id:Annotated[str , InjectedToolCallId],
    
)->list:
    """Search the web for current or supplementary information using Tavily."""
    client= TavilyClient(api_key = os.environ.get("TAVILY_API_KEY"))
    results = client.search(optimised_query , max_results = max_results)
    if not results.get("results"):
        return [ToolMessage(content = "No relevant web search results found.", tool_call_id = tool_call_id)]
    
    web_docs = [Document(
        page_content = r["content"],
        metadata={"url": r["url"], "title": r.get("title", "Web Result")},
    ) for r in results["results"]]
    
    summary = f"Retrieved {len(web_docs)} web result(s) for: {optimised_query}"
    
    return [
        ToolMessage(content=summary, tool_call_id=tool_call_id),
        Command(update={"retrieved_docs": (current_docs or []) + web_docs}),
    ]

# ── Retrieval agent singletons ────────────────────────────────────────────────

RETRIEVAL_TOOLS = [retrieve_from_vectorstore , web_search]
retrieval_llm = llm.bind_tools(RETRIEVAL_TOOLS , parallel_tool_calls = False)
base_tool_node = ToolNode(RETRIEVAL_TOOLS)


RETRIEVE_SYSTEM = (
    "You are a research assistant gathering context to answer a user's question about research papers.\n\n"
    "You have two tools available and full control over how you use them:\n\n"
    "1. retrieve_from_vectorstore — searches the uploaded paper collection.\n"
    "   You decide:\n"
    "   - query: the semantic search query (phrase it to best match relevant paper chunks)\n"
    "   - k: how many chunks to retrieve (1–10; use more for broad questions, fewer for specific ones)\n\n"
    "2. web_search — searches the live web via Tavily.\n"
    "   You decide:\n"
    "   - optimized_query: rewrite the user's question as a concise, keyword-rich web search query\n"
    "   - max_results: how many results to fetch (1–10)\n\n"
    "Choose the right source based on the question:\n"
    "- Questions about the uploaded papers → use retrieve_from_vectorstore\n"
    "- Questions about current events, recent developments, or supplementary information → use web_search\n"
    "- Call only one tool per turn.\n\n"
    "Do NOT produce a final answer. Only call tools to collect context."
)

# ── Relevancy check ───────────────────────────────────────────────────────────

RELEVANCY_CHECK_SYSTEM = (
    "You are evaluating whether retrieved document chunks are relevant enough "
    "to answer a user's question about research papers.\n\n"
    "Return is_relevant=true if the chunks contain information that meaningfully "
    "addresses the question — even partially. "
    "Return is_relevant=false only if the chunks are clearly off-topic or contain "
    "no useful information.\n\nBe lenient: if there is any substantive overlap, return true."
)

relevancy_llm = llm.with_structured_output(RelevancyDecision)

QUERY_REWRITE_SYSTEM = (
    "You are a query rewriting assistant for a research paper retrieval system. "
    "The previous query failed to retrieve relevant document chunks. "
    "Rewrite the query using more specific or alternative terminology, "
    "domain-specific keywords, or a narrower sub-question.\n\n"
    "Return ONLY the rewritten query as plain text. No explanation, no preamble."
)


# ── Nodes ─────────────────────────────────────────────────────────────────────

def agent_node(state: RAGState) -> dict:
    current_attempts = state.get("retrieval_attempts",0)
    # Once at the cap, use plain LLM so the agent cannot emit more tool calls.
    # This prevents orphaned tool_call IDs from entering the persisted message history.
    # retrieval llm --> tool call --> tool result
    # llm --> no tools are bounded --> tool call
    
    lm = llm if current_attempts >= MAX_RETRIEVAL_ATTEMPTS else retrieval_llm
    messages = state["messages"] + [{"role": "system", "content": RETRIEVE_SYSTEM}]
    response = lm.invoke({"messages": messages})
    updates: dict = {"messages": [response]}
    if getattr(response, "tool_calls", None):
        updates["retrieval_attempts"] = current_attempts + 1
    return updates


def relevancy_check_node(state: RAGState) -> dict:
    query = state["query"]
    docs = state.get("retrieved_docs") or []
    doc_snippets = "\n\n---\n\n".join(doc.page_content[:300] for doc in docs[:3])
    if not doc_snippets:
        return {"is_relevant": False}
    
    prompt = (
        f"Question: {query}\n\nRetrieved chunks:\n{doc_snippets}\n\n"
        "Are these chunks relevant to answering the question?"
    )
    
    decision: RelevancyDecision = relevancy_llm.invoke([
        {"role": "system", "content": RELEVANCY_CHECK_SYSTEM},
        {"role": "user", "content": prompt},
    ])
    
    return {"is_relevant": decision.is_relevant}


def query_rewrite_node(state: RAGState) -> dict:
    original_query = state["query"]
    rewrite_count = state.get("rewrite_count", 0)
    response = llm.invoke([
        {"role": "system", "content": QUERY_REWRITE_SYSTEM},
        {"role": "user", "content": f"Original query: {original_query}\n\nWrite an improved search query."},
    ])
    rewritten = response.content.strip()
    return {
        "messages": [HumanMessage(content=rewritten)],
        "query": rewritten,
        "retrieved_docs": [],
        "retrieval_attempts": 0,
        "rewrite_count": rewrite_count + 1,
        "is_relevant": None,
    }
    
    


