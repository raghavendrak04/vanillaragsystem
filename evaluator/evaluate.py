"""
Evaluation Runner — Runs the RAG pipeline against a question set and reports metrics.

Supports:
- CSV question files with columns: question, expected_answer, source_doc, is_unanswerable
- Built-in sample questions for quick testing
- Aggregated metrics report
"""

import csv
import json
import time
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from retriever.search import search
from generator.llm import generate_answer
from evaluator.metrics import compute_all_metrics


# Built-in test questions for the Navy Regulations document
SAMPLE_QUESTIONS = [
    {
        "question": "What are the eligibility criteria for recruitment to the Indian Naval Auxiliary Service?",
        "source_doc": "RegsNavyIV",
        "is_unanswerable": False,
    },
    {
        "question": "Who is the Administrative Authority as defined in Navy Regulations Part IV?",
        "source_doc": "RegsNavyIV",
        "is_unanswerable": False,
    },
    {
        "question": "What is the definition of 'Emergency' in the Navy Regulations?",
        "source_doc": "RegsNavyIV",
        "is_unanswerable": False,
    },
    {
        "question": "What is the tenure of appointment for permanent staff officers in INAS?",
        "source_doc": "RegsNavyIV",
        "is_unanswerable": False,
    },
    {
        "question": "What are the physical fitness standards required for enrolment?",
        "source_doc": "RegsNavyIV",
        "is_unanswerable": False,
    },
    {
        "question": "What is the capital of France?",
        "source_doc": None,
        "is_unanswerable": True,
    },
    {
        "question": "What is the GDP of India in 2024?",
        "source_doc": None,
        "is_unanswerable": True,
    },
    {
        "question": "Who constitutes the Service as per the regulations?",
        "source_doc": "RegsNavyIV",
        "is_unanswerable": False,
    },
]


def load_questions_csv(csv_path: str | Path) -> list[dict]:
    """
    Load questions from a CSV file.

    Expected columns:
    - question (required)
    - expected_answer (optional)
    - source_doc (optional)
    - is_unanswerable (optional, "true"/"false")
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Questions file not found: {csv_path}")

    questions = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = {
                "question": row.get("question", "").strip(),
                "expected_answer": row.get("expected_answer", "").strip() or None,
                "source_doc": row.get("source_doc", "").strip() or None,
                "is_unanswerable": row.get("is_unanswerable", "false").lower() == "true",
            }
            if q["question"]:
                questions.append(q)

    return questions


def run_evaluation(
    questions: Optional[list[dict]] = None,
    csv_path: Optional[str | Path] = None,
    use_samples: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Run the full RAG pipeline on a set of questions and compute metrics.

    Args:
        questions: List of question dicts.
        csv_path: Path to CSV file with questions.
        use_samples: Use built-in sample questions.
        verbose: Print detailed output for each question.

    Returns:
        Dict with per-question results and aggregated metrics.
    """
    # Load questions
    if csv_path:
        questions = load_questions_csv(csv_path)
    elif use_samples:
        questions = SAMPLE_QUESTIONS
    elif not questions:
        raise ValueError("Provide questions, csv_path, or use_samples=True")

    print(f"\n{'='*70}")
    print(f"  EVALUATION -- {len(questions)} questions")
    print(f"{'='*70}")

    results = []
    total_time = 0

    for i, q in enumerate(questions):
        question = q["question"]
        expected = q.get("expected_answer")
        source_doc = q.get("source_doc")
        is_unanswerable = q.get("is_unanswerable", False)

        if verbose:
            print(f"\n{'-'*70}")
            print(f"  Q{i+1}: {question}")
            if is_unanswerable:
                print(f"  [Expected: UNANSWERABLE]")

        start = time.time()

        try:
            # Retrieve
            search_results = search(question)
            retrieved_docs = [r.doc_name for r in search_results]

            # Generate
            response = generate_answer(question, search_results)
            answer = response["answer"]

            elapsed = time.time() - start
            total_time += elapsed

            # Compute metrics
            relevant_docs = [source_doc] if source_doc else None
            metrics = compute_all_metrics(
                question=question,
                answer=answer,
                retrieved_docs=retrieved_docs,
                relevant_docs=relevant_docs,
                expected_answer=expected,
                is_unanswerable=is_unanswerable,
            )
            metrics["time_seconds"] = round(elapsed, 2)
            metrics["error"] = None

            if verbose:
                print(f"  Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}")
                print(f"  Citations: {metrics['citation_count']}, Valid: {metrics['citations_valid']}")
                if is_unanswerable:
                    print(f"  Unanswerable Detection: {'PASS' if metrics['unanswerable_correct'] else 'FAIL'} ({metrics['unanswerable_label']})")
                print(f"  Time: {elapsed:.2f}s")

        except Exception as e:
            elapsed = time.time() - start
            total_time += elapsed
            metrics = {
                "question": question,
                "error": str(e),
                "time_seconds": round(elapsed, 2),
            }
            if verbose:
                print(f"  [ERROR] {e}")

        results.append(metrics)

    # Aggregate metrics
    aggregated = _aggregate_metrics(results)
    aggregated["total_time_seconds"] = round(total_time, 2)
    aggregated["questions_evaluated"] = len(results)

    # Print summary
    print(f"\n{'='*70}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'='*70}")
    for key, value in aggregated.items():
        if isinstance(value, float):
            print(f"  {key:30s}: {value:.3f}")
        else:
            print(f"  {key:30s}: {value}")
    print(f"{'='*70}")

    return {
        "results": results,
        "aggregated": aggregated,
    }


def _aggregate_metrics(results: list[dict]) -> dict:
    """Compute average metrics across all questions."""
    metrics_keys = [
        "retrieval_precision", "retrieval_recall", "retrieval_mrr",
        "answer_similarity", "citation_count",
    ]

    aggregated = {}

    for key in metrics_keys:
        values = [r[key] for r in results if key in r and r[key] is not None]
        if values:
            aggregated[f"avg_{key}"] = sum(values) / len(values)

    # Citation rates
    has_citations = [r.get("has_citations", False) for r in results if "has_citations" in r]
    if has_citations:
        aggregated["citation_rate"] = sum(has_citations) / len(has_citations)

    # Unanswerable accuracy
    unanswerable = [r.get("unanswerable_correct") for r in results if "unanswerable_correct" in r]
    if unanswerable:
        aggregated["unanswerable_accuracy"] = sum(unanswerable) / len(unanswerable)

    # Error rate
    errors = [r for r in results if r.get("error")]
    aggregated["error_count"] = len(errors)

    return aggregated
