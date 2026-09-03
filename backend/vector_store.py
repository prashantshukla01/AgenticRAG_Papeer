import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from backend.config import get_embedding_dimension, get_embedding_model

load_dotenv()

CACHE_DIR = Path("./embedding_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
embedding_file_store = LocalFileStore(str(CACHE_DIR))


def get_embeddings() -> CacheBackedEmbeddings:
    model_name = get_embedding_model()
    base_embeddings = OpenAIEmbeddings(
        model=model_name,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    return CacheBackedEmbeddings.from_bytes_store(
        base_embeddings,
        embedding_file_store,
        namespace=model_name,
        query_embedding_cache=True,
        key_encoder="blake2b",
    )


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=os.environ.get("QDRANT_URL"),
        api_key=os.environ.get("QDRANT_API_KEY"),
        timeout=120,
    )


# ── Collection ───────────────────────────────────────────────────────────────
def get_collection_name(session_id: str) -> str:
    return f"papeer_{session_id.replace('-', '_')}"


def get_vectorstore(session_id: str) -> QdrantVectorStore:
    client = get_qdrant_client()
    collection_name = get_collection_name(session_id)
    embed_dim = get_embedding_dimension()
    embeddings = get_embeddings()

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
        )

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )


# ── Public API ───────────────────────────────────────────────────────────────

def add_paper(docs: list[Document], session_id: str) -> None:
    get_vectorstore(session_id).add_documents(docs)


def list_papers(session_id: str) -> list[str]:
    client = get_qdrant_client()
    collection_name = get_collection_name(session_id)
    if not client.collection_exists(collection_name):
        return []
    seen: set[str] = set()
    titles: list[str] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            with_payload=True,
            limit=100,
            offset=offset,
        )
        for point in points:
            title = (point.payload or {}).get("metadata", {}).get("title")
            if title and title not in seen:
                seen.add(title)
                titles.append(title)
        if offset is None:
            break
    return titles


def search(query: str, session_id: str, k: int = 4) -> list[Document]:
    return get_vectorstore(session_id).similarity_search(query, k=k)
