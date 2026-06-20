"""
Document Loader — Extracts text from PDF files with metadata.

Supports:
- Single PDF files
- Directories of PDF files
- Preserves page numbers and document names for citation
"""

import os
import re
from pathlib import Path
from typing import Optional

from tqdm import tqdm


# Type for a loaded document page
class DocumentPage:
    """Represents a single page extracted from a PDF document."""

    def __init__(self, text: str, metadata: dict):
        self.text = text
        self.metadata = metadata

    def __repr__(self):
        return (
            f"DocumentPage(doc='{self.metadata.get('doc_name', '?')}', "
            f"page={self.metadata.get('page_number', '?')}, "
            f"chars={len(self.text)})"
        )


def _clean_text(text: str) -> str:
    """Clean extracted text: fix OCR artifacts, normalize whitespace."""
    if not text:
        return ""

    # Fix common OCR artifacts from the Navy Regulations PDF
    # Replace multiple spaces with single space (but preserve newlines)
    text = re.sub(r'[ \t]+', ' ', text)

    # Fix broken words across lines (hyphenation)
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)

    # Normalize line endings
    text = re.sub(r'\r\n', '\n', text)

    # Remove excessive blank lines (keep max 2)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()


def _extract_section_title(text: str) -> Optional[str]:
    """Try to extract the section/regulation number from the text."""
    # Match patterns like "10. Eligibility for Recruitment"
    match = re.match(r'^(\d+[\.\)]\s+[A-Z][^\.]{5,80})', text)
    if match:
        return match.group(1).strip()

    # Match patterns like "CHAPTER III"
    match = re.match(r'^(CHAPTER\s+[IVXLC]+[^\.]*)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Match patterns like "PART IV"
    match = re.match(r'^(PART\s+[IVXLC]+[^\.]*)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return None


def load_pdf_pdfplumber(pdf_path: str | Path) -> list[DocumentPage]:
    """Load a PDF using pdfplumber (better for tables and complex layouts)."""
    import pdfplumber

    pdf_path = Path(pdf_path)
    doc_name = pdf_path.stem
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            text = _clean_text(text)

            if not text or len(text.strip()) < 10:
                continue

            section_title = _extract_section_title(text)

            metadata = {
                "doc_name": doc_name,
                "doc_path": str(pdf_path),
                "page_number": i + 1,
                "total_pages": len(pdf.pages),
                "section_title": section_title,
            }

            pages.append(DocumentPage(text=text, metadata=metadata))

    return pages


def load_pdf_pypdf2(pdf_path: str | Path) -> list[DocumentPage]:
    """Load a PDF using PyPDF2 (fallback for simpler PDFs)."""
    from PyPDF2 import PdfReader

    pdf_path = Path(pdf_path)
    doc_name = pdf_path.stem
    pages = []

    reader = PdfReader(str(pdf_path))
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = _clean_text(text)

        if not text or len(text.strip()) < 10:
            continue

        section_title = _extract_section_title(text)

        metadata = {
            "doc_name": doc_name,
            "doc_path": str(pdf_path),
            "page_number": i + 1,
            "total_pages": len(reader.pages),
            "section_title": section_title,
        }

        pages.append(DocumentPage(text=text, metadata=metadata))

    return pages


def load_pdf(pdf_path: str | Path) -> list[DocumentPage]:
    """
    Load a PDF file, trying pdfplumber first, falling back to PyPDF2.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of DocumentPage objects with extracted text and metadata.
    """
    try:
        pages = load_pdf_pdfplumber(pdf_path)
        if pages:
            return pages
    except Exception as e:
        print(f"  [WARN] pdfplumber failed for {pdf_path}: {e}")

    try:
        pages = load_pdf_pypdf2(pdf_path)
        return pages
    except Exception as e:
        print(f"  [FAIL] PyPDF2 also failed for {pdf_path}: {e}")
        return []


def load_documents(data_dir: str | Path) -> list[DocumentPage]:
    """
    Load all PDF documents from a directory (recursively).

    Args:
        data_dir: Path to directory containing PDF files.

    Returns:
        List of all DocumentPage objects from all PDFs.
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find all PDF files
    pdf_files = sorted(data_dir.rglob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {data_dir}")

    print(f"\n[LOAD] Found {len(pdf_files)} PDF file(s) in {data_dir}")

    all_pages = []
    for pdf_path in tqdm(pdf_files, desc="Loading PDFs", unit="file"):
        pages = load_pdf(pdf_path)
        all_pages.extend(pages)
        if pages:
            print(f"  [OK] {pdf_path.name}: {len(pages)} pages extracted")
        else:
            print(f"  [FAIL] {pdf_path.name}: no text extracted")

    print(f"\n[TOTAL] {len(all_pages)} pages from {len(pdf_files)} documents")
    return all_pages


if __name__ == "__main__":
    # Quick test with the Navy Regs PDF
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    test_path = Path(__file__).parent.parent / "source" / "RegsNavyIV.pdf"
    if test_path.exists():
        pages = load_pdf(test_path)
        print(f"\nLoaded {len(pages)} pages from {test_path.name}")
        if pages:
            print(f"\nFirst page preview:\n{pages[0].text[:500]}")
            print(f"\nMetadata: {pages[0].metadata}")
    else:
        print(f"Test PDF not found: {test_path}")
