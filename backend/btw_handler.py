import os
from typing import Generator

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

from backend.config import get_llm_model
from backend.models import BtwRouteDecision

load_dotenv()


def get_btw_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=get_llm_model(),
        api_key=os.environ.get("OPENAI_API_KEY"),
    )


def handle_btw(query: str) -> Generator[str, None, None]:
    """Off-topic side channel — never touches the vector store or checkpointer."""
    llm = get_btw_llm()

    route_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Decide if answering this question requires a real-time web search (recent events, "
            "current prices, breaking news) or if your general knowledge is sufficient.",
        ),
        ("human", "{query}"),
    ])
    decision = (route_prompt | llm.with_structured_output(BtwRouteDecision)).invoke({"query": query})

    tavily_key = os.environ.get("TAVILY_API_KEY")
    if decision.needs_web_search and tavily_key:
        client = TavilyClient(api_key=tavily_key)
        results = client.search(query, max_results=3)
        context = "\n\n".join(r["content"] for r in results.get("results", []))
        sources = "\n".join(f"- {r['url']}" for r in results.get("results", []))

        answer_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Answer the question using the web search results below. Be concise.\n\n"
                f"Results:\n{context}\n\nSources:\n{sources}",
            ),
            ("human", "{query}"),
        ])
    else:
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question concisely from your general knowledge."),
            ("human", "{query}"),
        ])

    for chunk in (answer_prompt | llm).stream({"query": query}):
        if chunk.content:
            yield chunk.content