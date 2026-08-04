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
