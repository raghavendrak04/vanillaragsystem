"""
Vector Store — ChromaDB wrapper for persistent vector storage.

Handles creating, loading, and managing the vector collection.
"""

import shutil
from pathlib import Path
from typing import Optional

import chromadb

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


_client: Optional[chromadb.ClientAPI] = None
_collection: Optional[chromadb.Collection] = None


def get_client() -> chromadb.ClientAPI:
    """Get or create a persistent ChromaDB client."""
    global _client
    if _client is None:
        db_path = str(config.CHROMA_DB_DIR)
        Path(db_path).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=db_path)
    return _client


def get_collection(name: Optional[str] = None) -> chromadb.Collection:
    """Get or create the vector collection."""
    global _collection
    name = name or config.COLLECTION_NAME

    client = get_client()
    _collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},  # Use cosine similarity
    )
    return _collection


def add_chunks(
    chunks: list,  # list of Chunk objects
    embeddings: list[list[float]],
    collection_name: Optional[str] = None,
) -> int:
    """
    Add chunks with their embeddings to the vector store.

    Args:
        chunks: List of Chunk objects with text and metadata.
        embeddings: Corresponding embedding vectors.
        collection_name: Optional collection name override.

    Returns:
        Number of chunks added.
    """
    collection = get_collection(collection_name)

    # Prepare data for ChromaDB
    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk_{chunk.chunk_id}"
        ids.append(chunk_id)
        documents.append(chunk.text)

        # ChromaDB metadata must be flat (str, int, float, bool)
        meta = {
            "doc_name": str(chunk.metadata.get("doc_name", "")),
            "doc_path": str(chunk.metadata.get("doc_path", "")),
            "page_number": int(chunk.metadata.get("page_number", 0)),
            "section_title": str(chunk.metadata.get("section_title", "") or ""),
            "chunk_index": int(chunk.metadata.get("chunk_index", i)),
            "char_count": int(chunk.metadata.get("char_count", len(chunk.text))),
        }
        metadatas.append(meta)

    # Add in batches (ChromaDB has a batch limit)
    batch_size = 500
    total_added = 0

    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_docs = documents[i:i + batch_size]
        batch_metas = metadatas[i:i + batch_size]
        batch_embeds = embeddings[i:i + batch_size]

        collection.add(
            ids=batch_ids,
            embeddings=batch_embeds,
            documents=batch_docs,
            metadatas=batch_metas,
        )
        total_added += len(batch_ids)

    print(f"\n[STORE] Stored {total_added} chunks in ChromaDB collection '{collection.name}'")
    print(f"   Total in collection: {collection.count()}")
    return total_added


def query_collection(
    query_embedding: list[float],
    top_k: Optional[int] = None,
    collection_name: Optional[str] = None,
    where: Optional[dict] = None,
) -> dict:
    """
    Query the vector store for similar chunks.

    Args:
        query_embedding: Query vector.
        top_k: Number of results to return.
        collection_name: Optional collection name override.
        where: Optional metadata filter.

    Returns:
        ChromaDB query results dict with keys: ids, documents, metadatas, distances.
    """
    collection = get_collection(collection_name)
    top_k = top_k or config.TOP_K

    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_params["where"] = where

    results = collection.query(**query_params)
    return results


def get_collection_stats(collection_name: Optional[str] = None) -> dict:
    """Get statistics about the vector collection."""
    collection = get_collection(collection_name)
    count = collection.count()

    stats = {
        "name": collection.name,
        "count": count,
        "db_path": str(config.CHROMA_DB_DIR),
    }

    # Sample some metadata to show document coverage
    if count > 0:
        sample = collection.peek(min(10, count))
        if sample and sample.get("metadatas"):
            doc_names = set()
            for meta in sample["metadatas"]:
                if meta and "doc_name" in meta:
                    doc_names.add(meta["doc_name"])
            stats["sample_docs"] = list(doc_names)

    return stats


def reset_collection(collection_name: Optional[str] = None):
    """Delete and recreate the collection."""
    name = collection_name or config.COLLECTION_NAME
    client = get_client()

    try:
        client.delete_collection(name)
        print(f"  [DEL] Deleted collection '{name}'")
    except Exception:
        pass

    global _collection
    _collection = None


def reset_all():
    """Delete the entire ChromaDB database directory."""
    global _client, _collection
    _client = None
    _collection = None

    db_path = config.CHROMA_DB_DIR
    if db_path.exists():
        shutil.rmtree(db_path)
        print(f"  [DEL] Deleted ChromaDB directory: {db_path}")
