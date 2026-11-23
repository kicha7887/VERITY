"""Per-session document registry: validation, processing, persistence, isolation."""
import gc
import hashlib
import json
import re
import shutil
import time
from pathlib import Path
import numpy as np
from . import config
from .chunker import chunk_items
from .pdf_processor import PDFError, process_pdf
from .retriever import Retriever


class UploadError(Exception):
    """User-facing validation error."""


class DocumentManager:
    def __init__(self, session_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", session_id):
            raise ValueError("invalid session id")
        self.sid = session_id
        self.up, self.proc, self.idx = (config.UPLOADS / session_id, config.PROCESSED / session_id,
                                        config.INDEXES / session_id)
        for p in (self.up, self.proc, self.idx):
            p.mkdir(parents=True, exist_ok=True)
        self.docs, self.retriever, self._pages = {}, Retriever(), {}
        self._load()

    # ---- persistence -------------------------------------------------
    def _load(self):
        for f in self.proc.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                rec = data["record"]
                if rec["status"] != "ready" or not (self.up / f"{rec['doc_id']}.pdf").exists():
                    continue
                vec_f = self.idx / f"{rec['doc_id']}.npy"
                self.retriever.add(rec["doc_id"], data["chunks"], np.load(vec_f) if vec_f.exists() else None)
                self._pages[rec["doc_id"]] = data["page_stats"], data["page_texts"]
                self.docs[rec["doc_id"]] = rec
            except Exception:
                continue

    def _save(self, rec, chunks=None, page_texts=None, page_stats=None):
        (self.proc / f"{rec['doc_id']}.json").write_text(json.dumps(
            {"record": rec, "chunks": chunks or [], "page_texts": page_texts or [], "page_stats": page_stats or []}))

    # ---- public API --------------------------------------------------
    def add_upload(self, name, data: bytes, progress=None):
        cb = progress or (lambda f, l: None)
        safe = re.sub(r"[^\w.\- ()]", "_", Path(name).name)[:120] or "document.pdf"
        if not safe.lower().endswith(".pdf"):
            raise UploadError(f"'{safe}' is not a PDF file.")
        if not data[:1024].lstrip().startswith(b"%PDF"):
            raise UploadError(f"'{safe}' does not look like a valid PDF.")
        if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
            raise UploadError(f"'{safe}' exceeds the {config.MAX_UPLOAD_MB} MB limit.")
        doc_id = hashlib.sha1(data).hexdigest()[:10]
        if doc_id in self.docs and self.docs[doc_id]["status"] == "ready":
            return self.docs[doc_id]
        path = self.up / f"{doc_id}.pdf"
        path.write_bytes(data)
        rec = {"doc_id": doc_id, "filename": safe, "status": "uploaded", "size": len(data), "pages": None,
               "error": None, "metadata": {}, "stats": {}, "added": time.time()}
        self.docs[doc_id] = rec
        cb(0.05, "Uploaded")
        rec["status"] = "processing"
        try:
            out = process_pdf(str(path), doc_id, safe, cb)
            chunks = chunk_items(out["items"])
            if not chunks:
                raise PDFError("No extractable content (scanned PDF and OCR is unavailable?).")
            cb(0.93, "Building local vector index")
            index = self.retriever.add(doc_id, chunks)
            if index.vectors is not None:
                np.save(self.idx / f"{doc_id}.npy", index.vectors)
            rec.update(status="ready", pages=out["metadata"]["pages"], metadata=out["metadata"], stats={
                "chunks": len(chunks),
                "tables": sum(c["type"] == "table" for c in chunks),
                "charts": sum(c["type"] == "chart" for c in chunks),
                "ocr_pages": sum(s["method"] == "tesseract" for s in out["page_stats"]),
                "visual_pages": sum(s["has_visual"] for s in out["page_stats"])})
            self._pages[doc_id] = out["page_stats"], out["page_texts"]
            self._save(rec, chunks, out["page_texts"], out["page_stats"])
            cb(1.0, "Ready")
        except PDFError as e:
            self._fail(rec, str(e))
        except Exception as e:  # never leave a half-processed doc looking indexed
            self._fail(rec, f"Unexpected processing error ({type(e).__name__}).")
        return rec

    def _fail(self, rec, msg):
        rec.update(status="failed", error=msg)
        self.retriever.remove(rec["doc_id"])
        gc.collect()
        try:
            (self.up / f"{rec['doc_id']}.pdf").unlink(missing_ok=True)
        except OSError:
            pass
        self._save(rec)

    def remove(self, doc_id):
        self.docs.pop(doc_id, None); self._pages.pop(doc_id, None); self.retriever.remove(doc_id)
        gc.collect()
        for p in (self.up / f"{doc_id}.pdf", self.proc / f"{doc_id}.json", self.idx / f"{doc_id}.npy"):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

    def ready_ids(self):
        return [d for d, r in self.docs.items() if r["status"] == "ready"]

    def list_docs(self):
        return sorted(self.docs.values(), key=lambda r: r["added"])

    def pdf_path(self, doc_id):
        return str(self.up / f"{doc_id}.pdf")

    def page_count(self, doc_id):
        return (self.docs.get(doc_id) or {}).get("pages") or 0

    def page_text(self, doc_id, page):
        texts = self._pages.get(doc_id, ([], []))[1]
        return texts[page - 1] if 0 < page <= len(texts) else ""

    def has_visual(self, doc_id, page):
        stats = self._pages.get(doc_id, ([], []))[0]
        return bool(0 < page <= len(stats) and stats[page - 1]["has_visual"])

    def chunks(self, doc_id):
        idx = self.retriever.indexes.get(doc_id)
        return idx.chunks if idx else []

    def clear_session(self):
        for p in (self.up, self.proc, self.idx):
            shutil.rmtree(p, ignore_errors=True)
            p.mkdir(parents=True, exist_ok=True)
        self.docs.clear(); self._pages.clear(); self.retriever = Retriever()

    @staticmethod
    def cleanup_old(hours=None):
        cutoff = time.time() - 3600 * (hours or config.SESSION_MAX_AGE_HOURS)
        for root in (config.UPLOADS, config.PROCESSED, config.INDEXES):
            for d in root.iterdir():
                if d.is_dir() and d.stat().st_mtime < cutoff:
                    shutil.rmtree(d, ignore_errors=True)
