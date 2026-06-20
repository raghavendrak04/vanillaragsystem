"""
Configuration for the Defence RAG System.
Loads settings from environment variables and .env file.

Supports separate providers for embeddings and LLM generation.
"""

import os
from pathlib import Path

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
SOURCE_DIR = PROJECT_ROOT / "source"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"

# ──────────────────────────────────────────────
# Provider Config — supports split providers
# EMBEDDING_PROVIDER: used for embedding queries & documents
# LLM_PROVIDER: used for answer generation
# ──────────────────────────────────────────────
_DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", _DEFAULT_PROVIDER).lower()
LLM_PROVIDER = _DEFAULT_PROVIDER

# ──────────────────────────────────────────────
# OpenAI Configuration
# ──────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")

# ──────────────────────────────────────────────
# Google Gemini Configuration
# ──────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
GEMINI_LLM_MODEL = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash")

# ──────────────────────────────────────────────
# Chunking Configuration
# ──────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# ──────────────────────────────────────────────
# Retrieval Configuration
# ──────────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", "5"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "defence_rag")

# ──────────────────────────────────────────────
# Embedding batch size (to avoid rate limits)
# ──────────────────────────────────────────────
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "50"))


def get_api_key(provider: str) -> str:
    """Return the API key for the given provider."""
    if provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in .env")
        return OPENAI_API_KEY
    elif provider == "gemini":
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set in .env")
        return GEMINI_API_KEY
    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_active_api_key() -> str:
    """Return the API key for the active LLM provider."""
    return get_api_key(LLM_PROVIDER)


def get_embedding_api_key() -> str:
    """Return the API key for the embedding provider."""
    return get_api_key(EMBEDDING_PROVIDER)


def get_embedding_model() -> str:
    """Return the embedding model name for the embedding provider."""
    if EMBEDDING_PROVIDER == "openai":
        return OPENAI_EMBEDDING_MODEL
    return GEMINI_EMBEDDING_MODEL


def get_llm_model(provider: str = None) -> str:
    """Return the LLM model name for the given provider."""
    provider = provider or LLM_PROVIDER
    if provider == "openai":
        return OPENAI_LLM_MODEL
    return GEMINI_LLM_MODEL


def print_config():
    """Print current configuration for debugging."""
    print("=" * 60)
    print("  Defence RAG System - Configuration")
    print("=" * 60)
    print(f"  Embedding Provider: {EMBEDDING_PROVIDER}")
    print(f"  LLM Provider      : {LLM_PROVIDER}")
    print(f"  Embedding Model   : {get_embedding_model()}")
    print(f"  LLM Model         : {get_llm_model()}")
    print(f"  Chunk Size        : {CHUNK_SIZE}")
    print(f"  Chunk Overlap     : {CHUNK_OVERLAP}")
    print(f"  Top-K Retrieval   : {TOP_K}")
    print(f"  Data Dir          : {DATA_DIR}")
    print(f"  ChromaDB Dir      : {CHROMA_DB_DIR}")
    print("=" * 60)
