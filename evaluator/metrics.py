"""
Evaluation Metrics — Measures RAG pipeline quality.

Implements:
1. Retrieval Quality  — Did the right chunks come back?
2. Answer Correctness — Is the answer semantically correct?
3. Citation Accuracy  — Are citations valid and grounded?
4. Unanswerable Detection — Does the system correctly refuse?
"""

import re
import json
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def retrieval_precision(
    retrieved_docs: list[str],
    relevant_docs: list[str],
) -> float:
    """
    Calculate precision: fraction of retrieved docs that are relevant.

    Args:
        retrieved_docs: List of document names returned by retrieval.
        relevant_docs: List of document names that are actually relevant.

    Returns:
        Precision score (0.0 - 1.0).
    """
    if not retrieved_docs:
        return 0.0

    retrieved_set = set(retrieved_docs)
    relevant_set = set(relevant_docs)
    hits = retrieved_set & relevant_set

    return len(hits) / len(retrieved_set)


def retrieval_recall(
    retrieved_docs: list[str],
    relevant_docs: list[str],
) -> float:
    """
    Calculate recall: fraction of relevant docs that were retrieved.

    Args:
        retrieved_docs: List of document names returned by retrieval.
        relevant_docs: List of document names that are actually relevant.

    Returns:
        Recall score (0.0 - 1.0).
    """
    if not relevant_docs:
        return 1.0  # No relevant docs → vacuously true

    retrieved_set = set(retrieved_docs)
    relevant_set = set(relevant_docs)
    hits = retrieved_set & relevant_set

    return len(hits) / len(relevant_set)


def retrieval_mrr(
    retrieved_docs: list[str],
    relevant_docs: list[str],
) -> float:
    """
    Mean Reciprocal Rank — position of the first relevant result.

    Args:
        retrieved_docs: Ordered list of document names.
        relevant_docs: List of relevant document names.

    Returns:
        MRR score (0.0 - 1.0).
    """
    relevant_set = set(relevant_docs)
    for i, doc in enumerate(retrieved_docs):
        if doc in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def citation_check(answer: str, retrieved_sources: list[str]) -> dict:
    """
    Check if citations in the answer match retrieved sources.

    Args:
        answer: The generated answer text.
        retrieved_sources: List of source identifiers from retrieval.

    Returns:
        Dict with citation analysis results.
    """
    # Extract citations from the answer: [Source: X, Page Y]
    citation_pattern = r'\[Source:\s*([^,\]]+)'
    found_citations = re.findall(citation_pattern, answer)
    found_citations = [c.strip() for c in found_citations]

    # Check if cited sources were in the retrieved set
    retrieved_set = set(retrieved_sources)
    valid_citations = [c for c in found_citations if c in retrieved_set]
    invalid_citations = [c for c in found_citations if c not in retrieved_set]

    return {
        "total_citations": len(found_citations),
        "valid_citations": len(valid_citations),
        "invalid_citations": len(invalid_citations),
        "cited_sources": found_citations,
        "invalid_sources": invalid_citations,
        "has_citations": len(found_citations) > 0,
        "all_valid": len(invalid_citations) == 0 and len(found_citations) > 0,
    }


def unanswerable_detection(
    answer: str,
    is_unanswerable: bool,
) -> dict:
    """
    Check if the system correctly identifies unanswerable questions.

    Args:
        answer: The generated answer text.
        is_unanswerable: Whether the question is actually unanswerable.

    Returns:
        Dict with detection results.
    """
    # Patterns indicating the system recognized it can't answer
    refusal_patterns = [
        r"cannot answer",
        r"don'?t have.*information",
        r"not.*contain.*information",
        r"no.*relevant.*information",
        r"does not contain",
        r"insufficient information",
        r"not mentioned",
        r"not found in",
        r"unable to.*answer",
        r"outside.*scope",
        r"not covered",
    ]

    system_refused = any(
        re.search(pattern, answer, re.IGNORECASE)
        for pattern in refusal_patterns
    )

    if is_unanswerable:
        # True positive: system correctly refused
        correct = system_refused
        label = "TRUE_POSITIVE" if correct else "FALSE_NEGATIVE"
    else:
        # Should have answered
        correct = not system_refused
        label = "TRUE_NEGATIVE" if correct else "FALSE_POSITIVE"

    return {
        "is_unanswerable": is_unanswerable,
        "system_refused": system_refused,
        "correct": correct,
        "label": label,
    }


def text_similarity(text1: str, text2: str) -> float:
    """
    Simple word-overlap similarity (Jaccard) for answer comparison.
    Uses this as a fallback when embedding-based similarity isn't available.

    Args:
        text1: First text.
        text2: Second text.

    Returns:
        Similarity score (0.0 - 1.0).
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union)


def compute_all_metrics(
    question: str,
    answer: str,
    retrieved_docs: list[str],
    relevant_docs: Optional[list[str]] = None,
    expected_answer: Optional[str] = None,
    is_unanswerable: bool = False,
) -> dict:
    """
    Compute all available metrics for a single question.

    Args:
        question: The question asked.
        answer: The generated answer.
        retrieved_docs: Document names from retrieval.
        relevant_docs: Ground truth relevant documents (if known).
        expected_answer: Ground truth answer (if known).
        is_unanswerable: Whether the question is unanswerable.

    Returns:
        Dict with all metric scores.
    """
    metrics = {"question": question}

    # Retrieval metrics (if ground truth is available)
    if relevant_docs is not None:
        metrics["retrieval_precision"] = retrieval_precision(retrieved_docs, relevant_docs)
        metrics["retrieval_recall"] = retrieval_recall(retrieved_docs, relevant_docs)
        metrics["retrieval_mrr"] = retrieval_mrr(retrieved_docs, relevant_docs)

    # Citation check
    citation_result = citation_check(answer, retrieved_docs)
    metrics["citation_count"] = citation_result["total_citations"]
    metrics["citations_valid"] = citation_result["all_valid"]
    metrics["has_citations"] = citation_result["has_citations"]

    # Answer similarity (if expected answer is available)
    if expected_answer:
        metrics["answer_similarity"] = text_similarity(answer, expected_answer)

    # Unanswerable detection
    unanswerable_result = unanswerable_detection(answer, is_unanswerable)
    metrics["unanswerable_correct"] = unanswerable_result["correct"]
    metrics["unanswerable_label"] = unanswerable_result["label"]

    return metrics
