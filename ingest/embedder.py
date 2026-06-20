"""
Embedding Module — Generates vector embeddings for text chunks.

Uses the EMBEDDING_PROVIDER (may differ from LLM_PROVIDER).
Supports both OpenAI and Google Gemini embedding models.
"""

import time
from typing import Optional

from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def _embed_openai(texts: list[str], model: str, api_key: str) -> list[list[float]]:
    """Generate embeddings using OpenAI API."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


def _embed_gemini(texts: list[str], model: str, api_key: str) -> list[list[float]]:
    """Generate embeddings using Google Gemini API."""
    from google import genai
    client = genai.Client(api_key=api_key)
    result = client.models.embed_content(model=model, contents=texts)
    return [emb.values for emb in result.embeddings]


def _embed_batch_with_retry(
    batch: list[str], provider: str, model: str, api_key: str, max_retries: int = 5,
) -> list[list[float]]:
    """Embed a single batch with exponential backoff retry on rate limits."""
    for attempt in range(max_retries):
        try:
            if provider == "openai":
                return _embed_openai(batch, model, api_key)
            elif provider == "gemini":
                return _embed_gemini(batch, model, api_key)
            else:
                raise ValueError(f"Unknown provider: {provider}")
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = "429" in str(e) or "rate" in error_str or "quota" in error_str or "exhausted" in error_str
            if is_rate_limit and attempt < max_retries - 1:
                wait = min(30 * (2 ** attempt), 120)
                print(f"\n  [WAIT] Rate limited (attempt {attempt+1}/{max_retries}), waiting {wait}s...")
                time.sleep(wait)
            else:
                raise


def embed_texts(
    texts: list[str],
    batch_size: Optional[int] = None,
    provider: Optional[str] = None,
) -> list[list[float]]:
    """Generate embeddings for a list of texts using the EMBEDDING provider."""
    # Always use the EMBEDDING_PROVIDER for embeddings
    provider = provider or config.EMBEDDING_PROVIDER
    api_key = config.get_api_key(provider)
    model = config.get_embedding_model()

    if batch_size is None:
        batch_size = 20 if provider == "gemini" else config.EMBEDDING_BATCH_SIZE

    print(f"\n[EMBED] Embedding {len(texts)} texts using {provider}/{model} (batch={batch_size})")

    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", total=total_batches, unit="batch"):
        batch = texts[i:i + batch_size]
        embeddings = _embed_batch_with_retry(batch, provider, model, api_key)
        all_embeddings.extend(embeddings)

        if i + batch_size < len(texts):
            time.sleep(2.0 if provider == "gemini" else 0.2)

    print(f"   [OK] Generated {len(all_embeddings)} embeddings (dim={len(all_embeddings[0])})")
    return all_embeddings


def embed_query(query: str, provider: Optional[str] = None) -> list[float]:
    """Generate embedding for a single query using the EMBEDDING provider."""
    provider = provider or config.EMBEDDING_PROVIDER
    api_key = config.get_api_key(provider)
    model = config.get_embedding_model()
    result = _embed_batch_with_retry([query], provider, model, api_key)
    return result[0]
