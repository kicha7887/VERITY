"""Decides when page images are needed and renders them for Gemini's multimodal analysis."""
import re
from . import config
from .pdf_processor import render_page_png

VISUAL_RE = re.compile(r"\b(charts?|graphs?|plots?|figures?|fig\.|diagrams?|images?|pictures?|trends?|visuals?|infographics?|maps?|bars?|pie)\b", re.I)


def wants_visual(question):
    return bool(VISUAL_RE.search(question))


def pick_pages(evidence, dm, question):
    """Pages (doc_id, page) to analyse visually: chart evidence, or visual pages among hits for visual questions."""
    pages, want = [], wants_visual(question)
    if not want:  # keep Gemini usage economical: images only for visual questions
        return []
    for e in evidence.values():
        key = (e["doc_id"], e["page"])
        if key in pages:
            continue
        if e["type"] == "chart" or dm.has_visual(*key):
            pages.append(key)
    return pages[:config.MAX_VISUAL_PAGES]


def load_images(dm, pages, evidence):
    """Returns [(label, png)] and the evidence ids those images correspond to (adds image evidence if needed)."""
    images, ids = [], set()
    for doc_id, page in pages:
        try:
            png = render_page_png(dm.pdf_path(doc_id), page)
        except Exception:
            continue
        match = next((e for e in evidence.values() if (e["doc_id"], e["page"]) == (doc_id, page)
                      and e["type"] in ("chart", "image")), None)
        if match is None:
            eid = f"E{len(evidence) + 1}"
            evidence[eid] = {"id": eid, "chunk_id": f"{doc_id}-p{page}-img", "doc_id": doc_id, "page": page,
                             "filename": dm.docs[doc_id]["filename"], "type": "image", "score": None,
                             "text": "Rendered page image supplied for visual analysis.", "method": "page-render",
                             "extra": {}, "mode": "visual"}
            match = evidence[eid]
        ids.add(match["id"])
        images.append((f"Page image for evidence {match['id']} ({match['filename']}, page {page}):", png))
    return images, ids
