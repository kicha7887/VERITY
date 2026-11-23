"""Paragraph-aware chunking. Chunks never cross page boundaries; tables stay structured."""
import re
from . import config
from .table_extractor import to_markdown


def _split_long(par, size):
    if len(par) <= size:
        return [par]
    out, cur = [], ""
    for s in re.split(r"(?<=[.!?])\s+", par):
        while len(s) > size:
            if cur:
                out.append(cur); cur = ""
            out.append(s[:size]); s = s[size:]
        if len(cur) + len(s) + 1 > size and cur:
            out.append(cur); cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        out.append(cur)
    return out


def _tail(text, n):
    if n <= 0 or len(text) <= n:
        return ""
    t = text[-n:]
    return t[t.find(" ") + 1:] if " " in t else t


def _text_chunks(text, size, overlap):
    pieces = [p for par in re.split(r"\n\s*\n", text) if par.strip() for p in _split_long(par.strip(), size)]
    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + len(p) + 2 > size:
            chunks.append(cur)
            cur = (_tail(cur, overlap) + "\n\n" + p).strip()
        else:
            cur = (cur + "\n\n" + p).strip()
    if cur:
        chunks.append(cur)
    return chunks


def _table_chunks(it, limit):
    rows = it["extra"].get("rows")
    if len(it["text"]) <= limit or not rows:
        return [(it["text"], rows)]
    head, body = rows[0], rows[1:]
    groups, cur = [], []
    for r in body:
        if cur and len(to_markdown([head] + cur + [r])) > limit:
            groups.append(cur); cur = []
        cur.append(r)
    if cur:
        groups.append(cur)
    return [(to_markdown([head] + g), [head] + g) for g in groups]


def chunk_items(items, size=None, overlap=None):
    size = size or config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    out, counters = [], {}
    for it in items:
        if it["type"] == "table":
            parts = _table_chunks(it, config.TABLE_MAX_CHARS)
        elif it["type"] == "text":
            parts = [(t, None) for t in _text_chunks(it["text"], size, overlap)]
        else:
            parts = [(it["text"], None)]
        for text, rows in parts:
            key = (it["doc_id"], it["page"])
            counters[key] = counters.get(key, 0) + 1
            extra = dict(it["extra"])
            if it["type"] == "table":
                extra["rows"] = rows
            out.append({"chunk_id": f"{it['doc_id']}-p{it['page']}-{counters[key]}", "doc_id": it["doc_id"],
                        "filename": it["filename"], "page": it["page"], "type": it["type"],
                        "text": text, "method": it["method"], "extra": extra})
    return out
