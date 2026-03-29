import pypdf
from pathlib import Path

def load_pdf(pdf_path: str) -> list[dict]:
    """
    Load a PDF and return a list of pages.
    Each page is a dict with 'page_number' and 'text'.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {pdf_path}")
    
    pages = []
    
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f) # Load the PDF
        total_pages = len(reader.pages)
        print(f"Loaded PDF: {pdf_path} with {total_pages} pages.")
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append({
                "page_number": i + 1,
                "text": text.strip()
            })
            
    return pages

def detect_scanned(pages: list[dict], threshold: float = 0.8) -> bool:
    """
    Returns True if the PDF is likely scanned (no extractable text).
    threshold: fraction of pages that must be empty to call it scanned.
    """
    if not pages:
        return False
    
    empty = 0

    # For each page, if text length is less than 20 → count it as 1
    for p in pages:
        if len(p["text"]) < 20:
            empty += 1
            
    return (empty / len(pages)) >= threshold

def load_and_validate(pdf_path: str) -> list[dict]:
    """
    Main entry point. Loads PDF, checks for scanned content,
    returns pages or raises a clear error.
    """
    pages = load_pdf(pdf_path)
    
    if detect_scanned(pages):
        raise ValueError(
            f"This PDF appears to be scanned or image-based — "
            f"no extractable text found. "
            f"You would need OCR (e.g. pytesseract) to process it."
        )

    total_chars = sum(len(p["text"]) for p in pages)
    print(f"Text extracted: {total_chars:,} characters across {len(pages)} pages")
    return pages
