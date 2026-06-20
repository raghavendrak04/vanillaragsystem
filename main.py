"""
Defence RAG System — Main CLI Entry Point

A Plain-Vanilla RAG pipeline for answering questions about Indian defence
procurement policy, financial delegations, and naval regulations.

Usage:
    python main.py ingest --data-dir ./source
    python main.py query "What are the eligibility criteria for recruitment?"
    python main.py evaluate --samples
    python main.py info
"""

import argparse
import sys
import os
import time
from pathlib import Path

# Fix Windows console encoding for Unicode
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

import config


def cmd_ingest(args):
    """Ingest PDF documents: load -> chunk -> embed -> store."""
    from ingest.loader import load_documents
    from ingest.chunker import chunk_documents
    from ingest.embedder import embed_texts
    from retriever.vector_store import add_chunks, reset_collection, get_collection_stats

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print("[ERROR] Data directory not found: " + str(data_dir))
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  DEFENCE RAG - INGEST PIPELINE")
    print("=" * 60)
    config.print_config()

    start_time = time.time()

    # Step 1: Reset if requested
    if args.reset:
        print("\n[RESET] Resetting vector store...")
        reset_collection()

    # Step 2: Load documents
    print("\n[STEP 1/4] Loading PDF documents...")
    pages = load_documents(data_dir)
    if not pages:
        print("[ERROR] No pages extracted. Check your PDF files.")
        sys.exit(1)

    # Step 3: Chunk documents
    print("\n[STEP 2/4] Chunking documents...")
    chunks = chunk_documents(
        pages,
        chunk_size=args.chunk_size or config.CHUNK_SIZE,
        chunk_overlap=args.chunk_overlap or config.CHUNK_OVERLAP,
    )
    if not chunks:
        print("[ERROR] No chunks created. Check chunking configuration.")
        sys.exit(1)

    # Step 4: Embed chunks
    print("\n[STEP 3/4] Generating embeddings...")
    texts = [chunk.text for chunk in chunks]
    embeddings = embed_texts(texts)

    # Step 5: Store in vector database
    print("\n[STEP 4/4] Storing in vector database...")
    added = add_chunks(chunks, embeddings)

    elapsed = time.time() - start_time

    # Summary
    stats = get_collection_stats()
    print("\n" + "=" * 60)
    print("  INGEST COMPLETE")
    print("=" * 60)
    print(f"  Documents processed : {len(set(p.metadata['doc_name'] for p in pages))}")
    print(f"  Pages extracted     : {len(pages)}")
    print(f"  Chunks created      : {len(chunks)}")
    print(f"  Embeddings stored   : {added}")
    print(f"  Total in collection : {stats['count']}")
    print(f"  Time elapsed        : {elapsed:.1f}s")
    print("=" * 60)


def cmd_query(args):
    """Query the RAG pipeline with a natural language question."""
    from retriever.search import search
    from generator.llm import generate_answer
    from retriever.vector_store import get_collection_stats

    question = args.question
    if not question:
        print("[ERROR] Please provide a question.")
        sys.exit(1)

    # Check if the index exists
    stats = get_collection_stats()
    if stats["count"] == 0:
        print("[ERROR] No documents indexed. Run 'python main.py ingest' first.")
        sys.exit(1)

    print("\n" + "-" * 60)
    print(f"  QUESTION: {question}")
    print("-" * 60)

    start_time = time.time()

    # Step 1: Retrieve relevant chunks
    top_k = args.top_k or config.TOP_K
    results = search(question, top_k=top_k)

    if args.verbose:
        print(f"\n  Retrieved {len(results)} chunks:")
        for r in results:
            print(f"     #{r.rank} [score={r.similarity_score:.3f}] {r.doc_name}, Page {r.page_number}")
            if r.section_title:
                print(f"         Section: {r.section_title}")

    # Step 2: Generate answer
    response = generate_answer(question, results)

    elapsed = time.time() - start_time

    # Display answer
    print(f"\n  ANSWER ({response['model']}):")
    print("  " + "-" * 56)
    for line in response["answer"].split("\n"):
        print(f"  {line}")
    print("  " + "-" * 56)

    # Display citations
    if response["citations"]:
        print("\n  SOURCES USED:")
        for citation in response["citations"]:
            print(f"     * {citation}")

    print(f"\n  Time: {elapsed:.2f}s | Chunks used: {response['context_used']}")
    print("-" * 60)

    return response


def cmd_evaluate(args):
    """Run evaluation on a set of questions."""
    from evaluator.evaluate import run_evaluation
    from retriever.vector_store import get_collection_stats

    # Check if the index exists
    stats = get_collection_stats()
    if stats["count"] == 0:
        print("[ERROR] No documents indexed. Run 'python main.py ingest' first.")
        sys.exit(1)

    csv_path = args.questions if args.questions else None
    use_samples = args.samples or (csv_path is None)

    results = run_evaluation(
        csv_path=csv_path,
        use_samples=use_samples,
        verbose=args.verbose,
    )

    # Save results to file
    if args.output:
        import json
        output_path = Path(args.output)
        serializable = {
            "aggregated": results["aggregated"],
            "results": [
                {k: v for k, v in r.items() if not callable(v) and k != "search_results"}
                for r in results["results"]
            ],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"\n  Results saved to: {output_path}")

    return results


def cmd_info(args):
    """Show system information and index status."""
    from retriever.vector_store import get_collection_stats

    print("\n" + "=" * 60)
    print("  DEFENCE RAG - SYSTEM INFO")
    print("=" * 60)
    config.print_config()

    stats = get_collection_stats()
    print("\n  Vector Store Status:")
    print(f"    Collection    : {stats['name']}")
    print(f"    Total chunks  : {stats['count']}")
    print(f"    DB path       : {stats['db_path']}")
    if stats.get("sample_docs"):
        print(f"    Sample docs   : {', '.join(stats['sample_docs'])}")

    print("\n  Data Directories:")
    for label, path in [("Data", config.DATA_DIR), ("Source", config.SOURCE_DIR)]:
        if path.exists():
            pdfs = list(path.rglob("*.pdf"))
            print(f"    {label:12s}: {path} ({len(pdfs)} PDFs)")
        else:
            print(f"    {label:12s}: {path} (not found)")

    print("=" * 60)


def cmd_interactive(args):
    """Start an interactive query session."""
    from retriever.search import search
    from generator.llm import generate_answer
    from retriever.vector_store import get_collection_stats

    stats = get_collection_stats()
    if stats["count"] == 0:
        print("[ERROR] No documents indexed. Run 'python main.py ingest' first.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  DEFENCE RAG - INTERACTIVE MODE")
    print(f"  Index: {stats['count']} chunks | Type 'quit' to exit")
    print("=" * 60)

    while True:
        try:
            question = input("\n  Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not question or question.lower() in ("quit", "exit", "q"):
            print("  Goodbye!")
            break

        start = time.time()

        try:
            results = search(question, top_k=config.TOP_K)
            response = generate_answer(question, results)
            elapsed = time.time() - start

            print("\n  ANSWER:")
            print("  " + "-" * 56)
            for line in response["answer"].split("\n"):
                print(f"  {line}")
            print("  " + "-" * 56)

            print(f"  Sources: {', '.join(response['citations'][:3])}")
            print(f"  Time: {elapsed:.2f}s")

        except Exception as e:
            print(f"  [ERROR] {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Defence RAG System - Plain-Vanilla RAG for Defence Policy Documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py ingest --data-dir ./source
  python main.py query "What are the eligibility criteria for recruitment?"
  python main.py query "Who is the Administrative Authority?" --verbose
  python main.py evaluate --samples
  python main.py interactive
  python main.py info
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -- ingest --
    p_ingest = subparsers.add_parser("ingest", help="Ingest PDF documents into the vector store")
    p_ingest.add_argument("--data-dir", type=str, default="./source",
                          help="Directory containing PDF files (default: ./source)")
    p_ingest.add_argument("--reset", action="store_true",
                          help="Reset the vector store before ingesting")
    p_ingest.add_argument("--chunk-size", type=int, default=None,
                          help=f"Chunk size in characters (default: {config.CHUNK_SIZE})")
    p_ingest.add_argument("--chunk-overlap", type=int, default=None,
                          help=f"Chunk overlap in characters (default: {config.CHUNK_OVERLAP})")
    p_ingest.set_defaults(func=cmd_ingest)

    # -- query --
    p_query = subparsers.add_parser("query", help="Ask a question")
    p_query.add_argument("question", type=str, help="Natural language question")
    p_query.add_argument("--top-k", type=int, default=None,
                         help=f"Number of chunks to retrieve (default: {config.TOP_K})")
    p_query.add_argument("--verbose", "-v", action="store_true",
                         help="Show retrieved chunks and scores")
    p_query.set_defaults(func=cmd_query)

    # -- evaluate --
    p_eval = subparsers.add_parser("evaluate", help="Evaluate the pipeline")
    p_eval.add_argument("--questions", type=str, default=None,
                        help="Path to CSV file with questions")
    p_eval.add_argument("--samples", action="store_true",
                        help="Use built-in sample questions")
    p_eval.add_argument("--output", "-o", type=str, default=None,
                        help="Save results to JSON file")
    p_eval.add_argument("--verbose", "-v", action="store_true", default=True,
                        help="Show detailed output")
    p_eval.set_defaults(func=cmd_evaluate)

    # -- info --
    p_info = subparsers.add_parser("info", help="Show system info and index status")
    p_info.set_defaults(func=cmd_info)

    # -- interactive --
    p_interactive = subparsers.add_parser("interactive", help="Interactive query mode")
    p_interactive.set_defaults(func=cmd_interactive)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
