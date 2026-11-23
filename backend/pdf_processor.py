"""PDF processing: text (PyMuPDF), tables (pdfplumber), OCR (Tesseract), chart detection, page rendering."""
import io
import re
import pymupdf as fitz
from PIL import Image
from . import config
from .ocr_processor import ocr_available, ocr_image
from .table_extractor import extract_tables

CAPTION_RE = re.compile(r"^\s*(fig(ure)?\.?|chart|graph|exhibit|diagram|map)\s*[\dA-Za-z.\-:]*\b.*", re.I)


class PDFError(Exception):
    """Raised for malformed/unsupported PDFs; the message is safe to show users."""


def _page_text(page):
    blocks = [b for b in page.get_text("blocks") if b[6] == 0]
    blocks.sort(key=lambda b: (round(b[1] / 5), b[0]))
    paras = [" ".join(b[4].split()) for b in blocks]
    return "\n\n".join(p for p in paras if p)


def _item(doc_id, filename, page, typ, text, method, **extra):
    return {"doc_id": doc_id, "filename": filename, "page": page, "type": typ,
            "text": text, "method": method, "extra": extra}


def process_pdf(path, doc_id, filename, progress=None):
    """Returns {'metadata','items','page_texts','page_stats'}. Raises PDFError on failure."""
    try:
        doc = fitz.open(path)
    except Exception as e:
        raise PDFError("The file could not be opened as a PDF (it may be corrupted).") from e
    try:
        if doc.needs_pass:
            raise PDFError("The PDF is password-protected.")
        n = doc.page_count
        if n == 0:
            raise PDFError("The PDF has no pages.")
        if n > config.MAX_PAGES:
            raise PDFError(f"The PDF has {n} pages; the limit is {config.MAX_PAGES}.")
        md = doc.metadata or {}
        metadata = {"title": md.get("title") or "", "author": md.get("author") or "",
                    "created": md.get("creationDate") or "", "pages": n}
        items, page_texts, stats = [], [], []
        ocr_ok = ocr_available()
        for i, page in enumerate(doc, start=1):
            text = _page_text(page)
            method, conf, note = "pymupdf", None, None
            has_images = bool(page.get_images(full=True))
            if len(text.strip()) < config.MIN_TEXT_CHARS:
                if ocr_ok:
                    try:
                        pix = page.get_pixmap(dpi=200)
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        text, conf = ocr_image(img)
                        method = "tesseract"
                        if conf < 60:
                            note = f"Low OCR confidence ({conf:.0f}/100): text may contain errors."
                    except Exception:
                        note = "OCR failed for this page."
                elif has_images:
                    note = "Scanned page: OCR unavailable (Tesseract not installed), content not indexed."
            try:
                drawings = len(page.get_drawings())
            except Exception:
                drawings = 0
            has_visual = has_images or drawings >= 30
            if text.strip():
                items.append(_item(doc_id, filename, i, "text", text, method, ocr_confidence=conf, note=note))
            captions = [p.strip() for p in text.split("\n\n") if CAPTION_RE.match(p) and len(p) < 250]
            if has_visual and captions:
                items.append(_item(doc_id, filename, i, "chart", "\n".join(captions), method,
                                   note="Caption text only; use visual analysis for chart values."))
            page_texts.append(text)
            stats.append({"page": i, "chars": len(text), "method": method, "ocr_confidence": conf,
                          "has_visual": has_visual, "note": note})
            if progress:
                progress(0.1 + 0.5 * i / n, f"Reading page {i}/{n}")
        try:
            tables = extract_tables(path, lambda i, t: progress and progress(0.6 + 0.3 * i / t, f"Tables {i}/{t}"))
        except Exception:
            tables = []
        for t in tables:
            items.append(_item(doc_id, filename, t["page"], "table", t["markdown"], "pdfplumber", rows=t["rows"]))
        return {"metadata": metadata, "items": items, "page_texts": page_texts, "page_stats": stats}
    finally:
        doc.close()


def render_page_png(path, page_no, dpi=None):
    doc = fitz.open(path)
    try:
        return doc[page_no - 1].get_pixmap(dpi=dpi or config.VISUAL_DPI).tobytes("png")
    finally:
        doc.close()


def _phrases(text, ctype):
    if ctype == "table":
        cells = [c.strip() for ln in text.splitlines()[:4] for c in ln.split("|")]
        return [c for c in cells if len(c) > 2 and not set(c) <= set("-: ")][:8]
    w = text.split()
    if len(w) <= 6:
        return [" ".join(w)]
    mid = len(w) // 2
    return [" ".join(w[:6]), " ".join(w[mid:mid + 6]), " ".join(w[-6:])]


def render_page(path, page_no, highlight=None, ctype="text", dpi=None):
    """Render a page (PNG bytes); highlight located evidence. Returns (png, highlighted: bool)."""
    doc = fitz.open(path)
    try:
        page, found = doc[page_no - 1], False
        for ph in _phrases(highlight or "", ctype):
            for r in page.search_for(ph):
                page.draw_rect(r, color=(0.70, 0.58, 0.35), fill=(0.70, 0.58, 0.35), fill_opacity=0.30, width=1)
                found = True
        return page.get_pixmap(dpi=dpi or config.PREVIEW_DPI).tobytes("png"), found
    finally:
        doc.close()
