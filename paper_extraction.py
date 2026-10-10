import os
import re
import requests
import fitz  # PyMuPDF
from typing import Dict, Any


PDF_DIR = "paper_cache"
os.makedirs(PDF_DIR, exist_ok=True)


# Cleans text taken from a PDF: squeezes repeated spaces, reduces many blank lines to one blank line,
# and removes lines that are only a page number.
def clean_text(text: str) -> str:
    """Clean extracted PDF text while preserving useful academic content."""
    if not text:
        return ""

    # Remove excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove obvious repeated page artifacts
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)

    return text.strip()


# Finds the PDF link of a paper record: uses the pdf_url given by the corpus;
# for old records with only an arXiv id it builds the arXiv PDF link. Returns '' if none.
def get_pdf_url(paper: Dict[str, Any]) -> str:
    """
    Return the scholarly PDF URL supplied by the corpus.

    The MongoDB corpus provides pdf_url for eligible published papers.
    Legacy arXiv support is retained only for backward compatibility.
    """

    # Preferred source: MongoDB/OpenAlex PDF URL
    pdf_url = paper.get("pdf_url")

    if pdf_url:
        return str(pdf_url).strip()

    # Backward compatibility for older cached/arXiv records
    arxiv_id = paper.get("arxiv_id")

    if arxiv_id:
        arxiv_id = str(arxiv_id).strip()
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    return ""


# Downloads a paper's PDF and returns the local file path (or '' if it fails).
#  1. Gets the PDF URL and makes a safe file name from the paper id/DOI/title.
#  2. If the file already exists in paper_cache/ (bigger than 1 KB) it is reused - no new download.
#  3. Otherwise requests the URL (30 s timeout, follows redirects).
#  4. Checks that the answer really is a PDF (content type or '%PDF' header) and is not tiny/empty.
#  5. Saves it into paper_cache/.
def download_pdf(paper: Dict[str, Any]) -> str:
    """
    Download a scholarly PDF if it is not already cached.

    The preferred PDF URL comes from the MongoDB/OpenAlex corpus.

    Returns:
        Local PDF path, or empty string on failure.
    """

    pdf_url = get_pdf_url(paper)

    if not pdf_url:
        print("   ⚠️ No PDF URL available.")
        return ""

    # Create a stable filename from paper_id / DOI / title
    paper_id = (
        paper.get("paper_id")
        or paper.get("doi")
        or paper.get("arxiv_id")
        or paper.get("title")
        or "paper"
    )

    safe_id = re.sub(
        r"[^a-zA-Z0-9._-]",
        "_",
        str(paper_id)
    )

    # Prevent excessively long Windows filenames
    safe_id = safe_id[:180]

    pdf_path = os.path.join(PDF_DIR, f"{safe_id}.pdf")

    # Reuse cached PDF
    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000:
        print(f"   ♻️ Using cached PDF: {safe_id}.pdf")
        return pdf_path

    print(f"   📥 Downloading PDF: {paper.get('title', '')[:60]}")
    print(f"   🔗 Source: {pdf_url[:120]}")

    try:
        response = requests.get(
            pdf_url,
            timeout=30,
            headers={
                "User-Agent": "MultiAgentLiteratureReviewGenerator/1.0"
            },
            allow_redirects=True,
        )

        response.raise_for_status()

        content_type = response.headers.get(
            "content-type",
            ""
        ).lower()

        # Validate that the response is actually a PDF
        if (
            "pdf" not in content_type
            and not response.content.startswith(b"%PDF")
        ):
            print(
                "   ⚠️ Downloaded response is not a valid PDF."
            )
            return ""

        with open(pdf_path, "wb") as f:
            f.write(response.content)

        if os.path.getsize(pdf_path) < 1000:
            print(
                "   ⚠️ Downloaded PDF appears to be empty."
            )

            try:
                os.remove(pdf_path)
            except OSError:
                pass

            return ""

        print(
            f"   ✅ PDF downloaded: "
            f"{os.path.getsize(pdf_path) / 1024:.1f} KB"
        )

        return pdf_path

    except requests.exceptions.RequestException as e:
        print(f"   ⚠️ PDF download failed: {e}")
        return ""

    except Exception as e:
        print(f"   ⚠️ PDF processing failed: {e}")
        return ""


# Reads all the text of a PDF file with PyMuPDF (fitz).
# It opens the file, takes the text of every page and marks it with '--- PAGE n ---',
# joins the pages and cleans the result. A page that fails is skipped; a bad file returns ''.
def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from all pages of a PDF using PyMuPDF."""

    if not pdf_path or not os.path.exists(pdf_path):
        return ""

    try:
        document = fitz.open(pdf_path)

        pages = []

        for page_number, page in enumerate(
            document,
            start=1
        ):
            try:
                text = page.get_text("text")

                if text:
                    pages.append(
                        f"\n--- PAGE {page_number} ---\n{text}"
                    )

            except Exception as page_error:
                print(
                    f"   ⚠️ Could not extract page "
                    f"{page_number}: {page_error}"
                )

        document.close()

        full_text = "\n".join(pages)
        full_text = clean_text(full_text)

        return full_text

    except Exception as e:
        print(
            f"   ⚠️ PDF text extraction failed: {e}"
        )
        return ""


# Complete extraction for ONE paper:
#  1. download_pdf() -> 2. extract_pdf_text().
#  3. If at least 1000 characters of text were extracted, mark content_source='full_text'.
#  4. Otherwise fall back to the abstract and mark content_source='abstract'.
# Returns the paper record plus content_source, pdf_path, full_text and text_length.
def extract_paper_content(
    paper: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Complete paper-content extraction pipeline.

    Strategy:
        1. Use scholarly PDF URL from the corpus.
        2. Download and cache PDF.
        3. Extract full text.
        4. Fall back to abstract if necessary.
    """

    title = paper.get(
        "title",
        "Unknown paper"
    )

    print(
        f"\n📄 [Paper Extraction]: "
        f"{title[:80]}"
    )

    pdf_path = download_pdf(paper)

    if pdf_path:

        full_text = extract_pdf_text(
            pdf_path
        )

        if full_text and len(full_text) >= 1000:

            print(
                f"   ✅ Full text extracted "
                f"({len(full_text):,} characters)"
            )

            return {
                **paper,
                "content_source": "full_text",
                "pdf_path": pdf_path,
                "full_text": full_text,
                "text_length": len(full_text),
            }

        print(
            "   ⚠️ Full-text extraction "
            "produced insufficient text."
        )

    # Safe fallback to abstract
    abstract = paper.get(
        "abstract",
        ""
    )

    print(
        "   ↪ Falling back to paper abstract."
    )

    return {
        **paper,
        "content_source": "abstract",
        "pdf_path": pdf_path,
        "full_text": abstract,
        "text_length": len(abstract),
    }


# Runs extract_paper_content() for every selected paper, one after another,
# printing progress, and returns the list of paper records with their text.
def extract_papers_content(
    papers: list[Dict[str, Any]]
) -> list[Dict[str, Any]]:
    """Extract content for multiple selected papers."""

    extracted = []

    print("\n" + "=" * 100)
    print("📚 PAPER FULL-TEXT EXTRACTION")
    print("=" * 100)

    for index, paper in enumerate(
        papers,
        start=1
    ):

        print(
            f"\n[{index}/{len(papers)}] "
            f"{paper.get('title', '')[:80]}"
        )

        result = extract_paper_content(
            paper
        )

        extracted.append(result)

    print("\n" + "=" * 100)
    print(
        f"✅ [Paper Extraction]: "
        f"Processed {len(extracted)} papers."
    )
    print("=" * 100)

    return extracted