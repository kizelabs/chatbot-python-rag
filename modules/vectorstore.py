import os

import streamlit as st
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGEngine, PGVectorStore
from sqlalchemy import text


TABLE_NAME = "studybuddy_docs"

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://chatbot:chatbot123@localhost:5432/chatbot_rag",
)

# Vector dimension — matches NVIDIA llama-nemotron-embed-1b-v2 (2048) or fallback (1536)
VECTOR_SIZE = 2048


class GenericEmbeddings(Embeddings):
    """Fallback embeddings using OpenAI-compatible API or mock."""

    def embed_documents(self, texts):
        return [[0.1] * 1536 for _ in texts]

    def embed_query(self, text):
        return [0.1] * 1536


def get_embeddings():
    """Get embedding model instance."""
    if st.session_state.get("nvidia_api_key"):
        try:
            from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

            return NVIDIAEmbeddings(
                model="nvidia/llama-nemotron-embed-1b-v2",
                api_key=st.session_state.nvidia_api_key,
                truncate="END",
            )
        except Exception:
            pass

    return GenericEmbeddings()


def _get_engine():
    """Create PGEngine from connection string."""
    return PGEngine.from_connection_string(url=DATABASE_URL)


def _ensure_table(engine: PGEngine, vector_size: int):
    """Create the vector store table if it doesn't already exist."""
    from sqlalchemy import create_engine as sa_create_engine

    sa_engine = sa_create_engine(DATABASE_URL)
    create_sql = text(f"""
        CREATE TABLE IF NOT EXISTS "public"."{TABLE_NAME}" (
            "langchain_id" UUID PRIMARY KEY,
            "content" TEXT NOT NULL,
            "embedding" vector({vector_size}) NOT NULL,
            "langchain_metadata" JSON
        );
    """)
    with sa_engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.execute(create_sql)
        conn.commit()
    sa_engine.dispose()


def get_vectorstore():
    """Get or create the PGVectorStore instance."""
    embeddings = get_embeddings()

    # Determine vector size based on embedding model
    if isinstance(embeddings, GenericEmbeddings):
        vector_size = 1536
    else:
        vector_size = VECTOR_SIZE

    engine = _get_engine()
    _ensure_table(engine, vector_size)

    return PGVectorStore.create_sync(
        engine=engine,
        table_name=TABLE_NAME,
        embedding_service=embeddings,
    )


def similarity_search(query: str, k: int = 4):
    """Search for similar documents in the vector store."""
    try:
        vectorstore = get_vectorstore()
        results = vectorstore.similarity_search_with_score(query, k=k)
        return results
    except Exception:
        return []
