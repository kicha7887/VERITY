"""Builds validated citations and renumbers [E#] markers to [1], [2]..."""
import re


def _n(s):
    return " ".join(s.lower().split())


def passage_on_page(ev, page_text):
    """Deterministic check that the cited passage really exists on the cited page."""
    if ev["type"] == "image":
        return True
    pt = _n(page_text)
    if ev["type"] == "table":
        rows = (ev.get("extra") or {}).get("rows") or []
        cells = [c for r in rows[:2] for c in r if c.strip()]
        return bool(cells) and all(_n(c) in pt for c in cells)
    first = _n(ev["text"].splitlines()[0] if ev["type"] == "chart" else ev["text"])[:80]
    return bool(first) and first in pt


def build_citations(used_ids, evidence, dm):
    """Returns (citations, id_to_num, rejected_ids). Invalid citations are rejected, never shown."""
    cits, id_map, rejected = [], {}, []
    for i in used_ids:
        ev = evidence.get(i)
        if not ev or ev["doc_id"] not in dm.docs or not 1 <= ev["page"] <= dm.page_count(ev["doc_id"]):
            rejected.append(i); continue
        located = passage_on_page(ev, dm.page_text(ev["doc_id"], ev["page"]))
        if not located and ev["type"] in ("text", "chart"):
            rejected.append(i); continue
        id_map[i] = len(cits) + 1
        cits.append({"num": len(cits) + 1, "ev_id": i, "chunk_id": ev["chunk_id"], "doc_id": ev["doc_id"],
                     "filename": ev["filename"], "page": ev["page"], "type": ev["type"], "text": ev["text"],
                     "score": ev.get("score"), "mode": ev.get("mode"), "method": ev.get("method"),
                     "note": (ev.get("extra") or {}).get("note"),
                     "ocr_confidence": (ev.get("extra") or {}).get("ocr_confidence"),
                     "extra": {"rows": (ev.get("extra") or {}).get("rows")} if ev["type"] == "table" else {},
                     "located": located})
    return cits, id_map, rejected


def renumber(text, id_map):
    return re.sub(r"\s?\[E\d+\]", lambda m: f" [{id_map[m.group().strip(' []')]}]"
                  if m.group().strip(" []") in id_map else "", text)
