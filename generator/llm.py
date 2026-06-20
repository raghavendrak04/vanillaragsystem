"""
LLM Module — Wrapper for generating answers using OpenAI or Google Gemini.

Uses the LLM_PROVIDER (may differ from EMBEDDING_PROVIDER).
Includes retry logic for rate limits.
"""

import time
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from generator.prompt import build_rag_prompt
from retriever.search import SearchResult, format_context


def _generate_openai(messages: list[dict], model: str, api_key: str) -> str:
    """Generate answer using OpenAI API."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()


def _generate_gemini(messages: list[dict], model: str, api_key: str) -> str:
    """Generate answer using Google Gemini API."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    system_msg = ""
    user_content = []
    for msg in messages:
        if msg["role"] == "system":
            system_msg = msg["content"]
        else:
            user_content.append(
                types.Content(
                    role="user" if msg["role"] == "user" else "model",
                    parts=[types.Part.from_text(text=msg["content"])],
                )
            )

    response = client.models.generate_content(
        model=model,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=system_msg,
            temperature=0.1,
            max_output_tokens=2048,
        ),
    )
    return response.text.strip()


def _generate_with_retry(messages: list[dict], provider: str, model: str, api_key: str, max_retries: int = 3) -> str:
    """Generate with exponential backoff retry on rate limits."""
    for attempt in range(max_retries):
        try:
            if provider == "openai":
                return _generate_openai(messages, model, api_key)
            elif provider == "gemini":
                return _generate_gemini(messages, model, api_key)
            else:
                raise ValueError(f"Unknown provider: {provider}")
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = "429" in str(e) or "rate" in error_str or "quota" in error_str or "exhausted" in error_str
            if is_rate_limit and attempt < max_retries - 1:
                wait = 20 * (2 ** attempt)
                print(f"  [WAIT] LLM rate limited (attempt {attempt+1}), waiting {wait}s...")
                time.sleep(wait)
            else:
                raise


def generate_answer(
    question: str,
    search_results: list[SearchResult],
    provider: Optional[str] = None,
) -> dict:
    """Generate a grounded answer using the LLM."""
    provider = provider or config.LLM_PROVIDER
    api_key = config.get_api_key(provider)
    model = config.get_llm_model(provider)

    context = format_context(search_results)
    messages = build_rag_prompt(question, context)

    answer = _generate_with_retry(messages, provider, model, api_key)

    citations = [r.citation() for r in search_results]

    return {
        "answer": answer,
        "citations": citations,
        "context_used": len(search_results),
        "model": f"{provider}/{model}",
        "search_results": search_results,
    }


def generate_raw(messages: list[dict], provider: Optional[str] = None) -> str:
    """Generate a raw LLM response (used for evaluation)."""
    provider = provider or config.LLM_PROVIDER
    api_key = config.get_api_key(provider)
    model = config.get_llm_model(provider)
    return _generate_with_retry(messages, provider, model, api_key)
