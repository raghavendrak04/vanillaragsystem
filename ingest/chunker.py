"""
Text Chunker — Splits document pages into passages with metadata.

Uses RecursiveCharacterTextSplitter to respect paragraph/sentence boundaries.
Attaches source metadata (doc_name, page, section) for citation support.
"""

import re
import hashlib
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from ingest.loader import DocumentPage


class Chunk:
    """A text chunk with metadata for retrieval and citation."""

    def __init__(self, text: str, metadata: dict, chunk_id: str = ""):
        self.text = text
        self.metadata = metadata
        # Generate deterministic ID from content + metadata
        self.chunk_id = chunk_id or self._generate_id()

    def _generate_id(self) -> str:
        """Generate a deterministic chunk ID based on content and metadata."""
        content = f"{self.metadata.get('doc_name', '')}:{self.metadata.get('page_number', '')}:{self.text[:100]}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def __repr__(self):
        return (
            f"Chunk(id={self.chunk_id}, "
            f"doc='{self.metadata.get('doc_name', '?')}', "
            f"page={self.metadata.get('page_number', '?')}, "
            f"chars={len(self.text)})"
        )

    def citation_string(self) -> str:
        """Format a citation string for this chunk."""
        doc = self.metadata.get("doc_name", "Unknown")
        page = self.metadata.get("page_number", "?")
        section = self.metadata.get("section_title", "")
        if section:
            return f"[Source: {doc}, Page {page}, Section: {section}]"
        return f"[Source: {doc}, Page {page}]"


def _detect_section_for_chunk(chunk_text: str, page_section: Optional[str]) -> Optional[str]:
    """
    Try to detect the section/regulation number within a chunk.
    Falls back to the page-level section if none found.
    """
    # Look for regulation numbers at start of text
    match = re.match(r'^(\d+[\.\)]\s+[A-Z][^\.]{5,80})', chunk_text.strip())
    if match:
        return match.group(1).strip()

    # Look for regulation numbers anywhere in the chunk
    match = re.search(r'\n(\d+[\.\)]\s+[A-Z][^\.]{5,80})', chunk_text)
    if match:
        return match.group(1).strip()

    return page_section


def chunk_documents(
    pages: list[DocumentPage],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> list[Chunk]:
    """
    Split document pages into chunks with metadata.

    Args:
        pages: List of DocumentPage objects from the loader.
        chunk_size: Maximum chunk size in characters (default from config).
        chunk_overlap: Overlap between chunks in characters (default from config).

    Returns:
        List of Chunk objects ready for embedding.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=config.CHUNK_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )

    all_chunks = []
    chunk_index = 0

    for page in pages:
        if not page.text or len(page.text.strip()) < 20:
            continue

        # Split this page's text
        text_chunks = splitter.split_text(page.text)

        for text in text_chunks:
            text = text.strip()
            if len(text) < 20:
                continue

            # Detect section for this specific chunk
            section = _detect_section_for_chunk(text, page.metadata.get("section_title"))

            metadata = {
                "doc_name": page.metadata["doc_name"],
                "doc_path": page.metadata["doc_path"],
                "page_number": page.metadata["page_number"],
                "total_pages": page.metadata.get("total_pages", 0),
                "section_title": section,
                "chunk_index": chunk_index,
                "char_count": len(text),
            }

            all_chunks.append(Chunk(text=text, metadata=metadata))
            chunk_index += 1

    print(f"\n[CHUNK] Chunking complete:")
    print(f"   Input pages  : {len(pages)}")
    print(f"   Output chunks: {len(all_chunks)}")
    if all_chunks:
        avg_size = sum(c.metadata["char_count"] for c in all_chunks) / len(all_chunks)
        print(f"   Avg chunk size: {avg_size:.0f} chars")
        print(f"   Chunk size config: {chunk_size} chars, {chunk_overlap} overlap")

    return all_chunks


if __name__ == "__main__":
    # Quick test
    from ingest.loader import load_pdf

    test_path = Path(__file__).parent.parent / "source" / "RegsNavyIV.pdf"
    if test_path.exists():
        pages = load_pdf(test_path)
        chunks = chunk_documents(pages)

        print(f"\nSample chunks:")
        for chunk in chunks[:3]:
            print(f"\n--- {chunk} ---")
            print(f"Citation: {chunk.citation_string()}")
            print(f"Text preview: {chunk.text[:200]}...")
    else:
        print(f"Test PDF not found: {test_path}")
