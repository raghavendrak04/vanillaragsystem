"""
Search Module — High-level retrieval interface.

Combines embedding + vector store query into a clean search API.
Returns ranked results with metadata for the generator.
"""

from dataclasses import dataclass
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from ingest.embedder import embed_query
from retriever.vector_store import query_collection


@dataclass
class SearchResult:
    """A single search result with text, metadata, and relevance score."""
    text: str
    doc_name: str
    page_number: int
    section_title: str
    distance: float  # Lower = more similar for cosine distance
    rank: int

    @property
    def similarity_score(self) -> float:
        """Convert cosine distance to similarity score (0-1)."""
        return max(0.0, 1.0 - self.distance)

    def citation(self) -> str:
        """Format as a citation string."""
        parts = [f"Source: {self.doc_name}", f"Page {self.page_number}"]
        if self.section_title:
            parts.append(f"Section: {self.section_title}")
        return "[" + ", ".join(parts) + "]"

    def __repr__(self):
        return (
            f"SearchResult(rank={self.rank}, "
            f"score={self.similarity_score:.3f}, "
            f"doc='{self.doc_name}', page={self.page_number})"
        )


def search(
    query: str,
    top_k: Optional[int] = None,
    doc_filter: Optional[str] = None,
) -> list[SearchResult]:
    """
    Search the vector store for chunks relevant to the query.

    Args:
        query: Natural language question.
        top_k: Number of results to return (default from config).
        doc_filter: Optional document name filter.

    Returns:
        Ranked list of SearchResult objects.
    """
    top_k = top_k or config.TOP_K

    # Embed the query
    query_embedding = embed_query(query)

    # Build metadata filter if needed
    where = None
    if doc_filter:
        where = {"doc_name": doc_filter}

    # Query vector store
    results = query_collection(
        query_embedding=query_embedding,
        top_k=top_k,
        where=where,
    )

    # Convert to SearchResult objects
    search_results = []

    if results and results.get("documents") and results["documents"][0]:
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            result = SearchResult(
                text=doc,
                doc_name=meta.get("doc_name", "Unknown"),
                page_number=meta.get("page_number", 0),
                section_title=meta.get("section_title", ""),
                distance=dist,
                rank=i + 1,
            )
            search_results.append(result)

    return search_results


def format_context(results: list[SearchResult]) -> str:
    """
    Format search results into a context string for the LLM.

    Args:
        results: List of SearchResult objects.

    Returns:
        Formatted context string with chunk texts and citations.
    """
    if not results:
        return "No relevant documents were found."

    context_parts = []
    for r in results:
        context_parts.append(
            f"--- Context {r.rank} {r.citation()} (Relevance: {r.similarity_score:.2f}) ---\n"
            f"{r.text}\n"
        )

    return "\n".join(context_parts)
