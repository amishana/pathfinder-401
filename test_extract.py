"""Quick test of PyMuPDF text extraction.

Usage:
    uv pip install pymupdf
    python test_extract.py path/to/cv.pdf

Prints per-page stats and a preview, and saves the full text next to the PDF
as <name>.txt (plus <name>.unsorted.txt so you can compare reading order).
"""
import sys
from pathlib import Path

import pymupdf

MIN_CHARS_PER_PAGE = 50


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python test_extract.py path/to/file.pdf")

    pdf_path = Path(sys.argv[1])
    doc = pymupdf.open(pdf_path)
    print(f"{pdf_path.name}: {len(doc)} page(s)\n")

    sorted_pages, unsorted_pages = [], []
    for i, page in enumerate(doc, 1):
        text = page.get_text("text", sort=True)  # sort=True: top-to-bottom, left-to-right
        sorted_pages.append(text)
        unsorted_pages.append(page.get_text("text"))  # raw content-stream order

        flag = "  <-- almost no text, likely scanned (needs OCR)" if len(text.strip()) < MIN_CHARS_PER_PAGE else ""
        print(f"--- Page {i}: {len(text)} chars, {len(page.get_text('blocks'))} blocks{flag}")
        print(text[:500].strip() or "(empty)")
        print()

    doc.close()

    out = pdf_path.with_suffix(".txt")
    out.write_text("\n\n".join(sorted_pages), encoding="utf-8")
    pdf_path.with_suffix(".unsorted.txt").write_text("\n\n".join(unsorted_pages), encoding="utf-8")

    total = sum(len(t) for t in sorted_pages)
    print(f"Total: {total} chars. Full text saved to {out}")
    print("Check for: columns interleaved, missing sections, garbled characters.")


if __name__ == "__main__":
    main()