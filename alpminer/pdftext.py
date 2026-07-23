"""PDF -> plain text with caching, scanned-PDF detection, and optional OCR.

Extraction order for each downloaded paper:
  1. the embedded text layer (PyMuPDF) -- covers almost all modern papers;
  2. when that yields too little text (a scanned PDF) and OCR is available,
     rasterize the pages and run Tesseract via pytesseract.

OCR is optional: `pip install alpminer[ocr]` plus the Tesseract binary
(https://github.com/UB-Mannheim/tesseract/wiki on Windows). Without it,
scanned PDFs are flagged `text: failed` exactly as before -- visible in
`alpminer status`, never silently dropped. `ocr_enabled = false` in
alpminer.toml turns the fallback off even when installed.
"""

from __future__ import annotations

import re
from pathlib import Path

from . import db
from .config import Config
from .utils import atomic_write_text, log

MIN_TEXT_CHARS = 600  # below this the PDF is almost certainly scanned/broken
OCR_MAX_PAGES = 60    # safety cap: OCR is slow (~1-3 s/page)
OCR_DPI = 200         # good accuracy/speed balance for journal scans

_HYPHEN_BREAK = re.compile(r"-\n(?=[a-z])")
_MANY_BLANKS = re.compile(r"\n{3,}")


def _clean(text: str) -> str:
    text = text.replace("\u00ad", "")             # soft hyphens
    text = _HYPHEN_BREAK.sub("", text)            # re-join hyphen line breaks
    text = _MANY_BLANKS.sub("\n\n", text)
    return text.strip()


def pdf_to_text(pdf_path: Path) -> str:
    import fitz  # PyMuPDF; imported lazily so unit tests can stub if needed

    doc = fitz.open(pdf_path)
    try:
        pages = [page.get_text("text") for page in doc]
    finally:
        doc.close()
    return _clean("\n".join(pages))


# ---- OCR fallback (optional) --------------------------------------------------

def ocr_available() -> bool:
    """True when pytesseract AND the Tesseract binary are both usable."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:  # noqa: BLE001 - any import/binary problem means "no"
        return False


def ocr_pdf_to_text(pdf_path: Path, max_pages: int = OCR_MAX_PAGES) -> str:
    """Rasterize the PDF and OCR each page with Tesseract. Slow; only called
    for PDFs whose text layer came up empty."""
    import io

    import fitz
    import pytesseract
    from PIL import Image

    doc = fitz.open(pdf_path)
    try:
        n = min(doc.page_count, max_pages)
        pages = []
        for i in range(n):
            pix = doc[i].get_pixmap(dpi=OCR_DPI)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            pages.append(pytesseract.image_to_string(img))
    finally:
        doc.close()
    return _clean("\n".join(pages))


def ensure_text(conn, cfg: Config, paper, ocr: str = "inline") -> str | None:
    """Return cached/extracted text for a downloaded paper, or None on failure.

    `ocr` controls what happens when the PDF has no usable text layer:
      "inline"  OCR it right now (when enabled + installed);
      "defer"   flag it text_status='ocr_pending' and return None, so the
                caller can process all text-layer papers first;
      "force"   OCR it now regardless of mode (the deferred phase).

    Updates text_status in the DB either way, so failures (e.g. scanned PDFs
    with no OCR installed) are visible in `alpminer status` instead of
    silently lost. When OCR supplied the text, text_ocr=1 is recorded so
    every export can flag OCR-sourced records.
    """
    txt_path = cfg.text_dir / f"{paper['id']}.txt"
    if paper["text_status"] == "ok" and txt_path.exists():
        return txt_path.read_text(encoding="utf-8")

    pdf_path = Path(paper["pdf_path"]) if paper["pdf_path"] else None
    if not pdf_path or not pdf_path.exists():
        # Stored paths are absolute, so moving or renaming the project folder
        # strands them; the file itself lives at the canonical location.
        # Self-heal: look there, and repair the row so this never recurs.
        canonical = cfg.pdf_dir / f"{paper['id']}.pdf"
        if canonical.exists():
            pdf_path = canonical
            db.set_fields(conn, paper["id"], pdf_path=str(canonical))
            log.info("[%s] pdf found at the canonical location; stored path "
                     "repaired", paper["id"])
        else:
            db.set_fields(conn, paper["id"], text_status="failed",
                          text_error="pdf file missing on disk")
            return None

    try:
        text = pdf_to_text(pdf_path)
    except Exception as exc:  # noqa: BLE001
        db.set_fields(conn, paper["id"], text_status="failed",
                      text_error=f"pdf parse error: {exc}"[:300])
        log.warning("[%s] PDF text extraction failed: %s", paper["id"], exc)
        return None

    source = "text layer"
    if len(text) < MIN_TEXT_CHARS:
        ocr_usable = getattr(cfg, "ocr_enabled", True) and ocr_available()
        if ocr_usable and ocr == "defer":
            db.set_fields(conn, paper["id"], text_status="ocr_pending",
                          text_error=None)
            log.info("[%s] scanned PDF flagged for the deferred OCR phase",
                     paper["id"])
            return None
        if ocr_usable:
            log.info("[%s] only %d chars in the text layer -- running OCR "
                     "(this can take a minute)...", paper["id"], len(text))
            try:
                text = ocr_pdf_to_text(pdf_path)
                source = "OCR"
            except Exception as exc:  # noqa: BLE001
                db.set_fields(conn, paper["id"], text_status="failed",
                              text_error=f"OCR failed: {exc}"[:300])
                log.warning("[%s] OCR failed: %s", paper["id"], exc)
                return None
        if len(text) < MIN_TEXT_CHARS:
            hint = ("" if source == "OCR"
                    else "; install OCR with `pip install alpminer[ocr]` "
                         "+ the Tesseract binary")
            db.set_fields(conn, paper["id"], text_status="failed",
                          text_error=f"only {len(text)} chars extracted "
                                     f"(scanned PDF?{hint})"[:300])
            log.warning("[%s] too little text (%d chars) -- likely a scanned "
                        "PDF", paper["id"], len(text))
            return None

    atomic_write_text(txt_path, text)
    db.set_fields(conn, paper["id"], text_status="ok", text_error=None,
                  text_path=str(txt_path),
                  text_ocr=1 if source == "OCR" else 0)
    if source == "OCR":
        log.info("[%s] recovered %d chars via OCR", paper["id"], len(text))
    return text
