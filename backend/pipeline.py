"""End-to-end RAG pipeline: retrieve -> (visual) -> generate -> verify -> cite, with intelligent refusal."""
import re
import time
from dataclasses import dataclass
from . import config, generator, verifier, citation_manager, visual_analyzer

OVERVIEW_RE = re.compile(r"\b(summari[sz]e|summary|overview|key findings|main (points|findings)|"
                         r"most important findings|what is this (report|document) about)\b", re.I)


@dataclass
class Settings:
    top_k: int = config.TOP_K
    min_score: float = 0.0      # 0 = automatic per retrieval mode
    use_visual: bool = True


def _msg(content, status, citations=None, notes=None, timings=None, **extra):
    return {"content": content, "status": status, "citations": citations or [], "notes": notes or [],
            "timings": timings or {}, **extra}


def _refusal(q, below, timings, searched=None, notes=None):
    text = config.REFUSAL
    if searched:
        text += f"\n\n**What I searched for:** {searched}"
    if below:
        rel = "; ".join(f"{b['filename']} p.{b['page']} ({b['type']})" for b in below)
        text += (f"\n\n**Closest related content (below the relevance threshold, not used as evidence):** {rel}. "
                 "You may need a more specific document or question.")
    return _msg(text, "refused", notes=notes, timings=timings, query=q)


def _from_prev(history, ready):
    """Re-use evidence cited in the previous answer so follow-ups like 'which page?' stay grounded."""
    for m in reversed(history or []):
        if m["role"] == "assistant" and m.get("citations"):
            return [dict(c, score=None, mode="context") for c in m["citations"] if c["doc_id"] in ready]
    return []


def answer(dm, question, doc_ids, history=None, settings=None):
    t0, timings, notes = time.time(), {}, []
    settings = settings or Settings()
    ready = [d for d in doc_ids if d in dm.ready_ids()]
    if not ready:
        return _msg("Please upload and select at least one document that is **Ready** before asking questions.",
                    "error")
    # 1-2. context-aware query
    query = generator.rewrite_query(question, history)
    # 3-4. retrieval (text + tables share one index)
    t = time.time()
    overview = bool(OVERVIEW_RE.search(question))
    if overview:
        hits, below = dm.retriever.overview(ready), []
        notes.append(f"Summary is based on {len(hits)} passages sampled across the document(s), not a full read.")
    else:
        hits, below = dm.retriever.search(query, ready, settings.top_k, settings.min_score or None)
        if history and len(question.split()) <= 12:
            seen = {h["chunk_id"] for h in hits}
            hits += [p for p in _from_prev(history, ready) if p["chunk_id"] not in seen][:4]
    timings["retrieval_s"] = round(time.time() - t, 3)
    if not hits:
        return _refusal(query, below, timings, searched=query)
    evidence = {}
    for n, h in enumerate(hits, 1):
        evidence[f"E{n}"] = dict(h, id=f"E{n}")
    # 5. visual analysis if useful
    images, visual_ids = [], set()
    if settings.use_visual and generator.available():
        pages = visual_analyzer.pick_pages(evidence, dm, question)
        if pages:
            images, visual_ids = visual_analyzer.load_images(dm, pages, evidence)
            notes.append(f"Visual analysis used for {len(images)} page image(s).")
    # 6-7. generation
    if not generator.available():
        cits, _, _ = citation_manager.build_citations(list(evidence)[:4], evidence, dm)
        timings["total_s"] = round(time.time() - t0, 2)
        return _msg("**Gemini is not configured** (set `GEMINI_API_KEY`), so no answer was generated. "
                    "Below are the most relevant passages retrieved from your documents.",
                    "retrieval_only", cits, notes, timings, query=query)
    t = time.time()
    try:
        res = generator.generate(question, query, evidence, history, images)
    except generator.GenerationError as e:
        cits, _, _ = citation_manager.build_citations(list(evidence)[:4], evidence, dm)
        timings["total_s"] = round(time.time() - t0, 2)
        return _msg(f"⚠️ {e} The most relevant retrieved passages are shown below.", "retrieval_only", cits,
                    notes, timings, query=query)
    timings["generation_s"] = round(time.time() - t, 2)
    if not res["sufficient"]:
        timings["total_s"] = round(time.time() - t0, 2)
        return _refusal(query, below, timings, searched=res.get("searched_for") or query, notes=notes)
    # 8. deterministic verification
    page_counts = {d: dm.page_count(d) for d in ready}
    rep = verifier.verify(res, evidence, set(ready), page_counts, visual_ids)
    timings["total_s"] = round(time.time() - t0, 2)
    if rep["status"] == "failed":
        out = _refusal(query, below, timings, searched=res.get("searched_for") or query, notes=notes)
        out["content"] += "\n\n_The generated answer could not be verified against the evidence, so it was withheld._"
        out["verification"] = rep
        return out
    used = []
    for c in rep["kept"] + rep["conflicts"]:
        used += [i for i in c["evidence"] if i in evidence and i not in used]
    cits, id_map, rejected = citation_manager.build_citations(used, evidence, dm)
    if rejected:  # drop anything relying on a citation that failed validation
        bad = set(rejected)
        rep["kept"] = [c for c in rep["kept"] if not bad & set(c["evidence"])]
        if not rep["kept"]:
            return _refusal(query, below, timings, searched=query, notes=notes)
        rep["status"], rep["answer_ok"] = "partial", False
    # 9. compose final text from verified material only
    if rep["status"] == "verified":
        body = citation_manager.renumber(res["answer"], id_map)
    else:
        body = "\n".join("- " + citation_manager.renumber(c["text"] + "".join(f" [{i}]" for i in c["evidence"]), id_map)
                         for c in rep["kept"])
        notes.append("Some statements could not be verified against the evidence and were removed.")
    if rep["conflicts"]:
        body += "\n\n**⚠ Conflicting evidence**\n" + "\n".join(
            "- " + citation_manager.renumber(c["description"] + "".join(f" [{i}]" for i in c["evidence"]), id_map)
            for c in rep["conflicts"])
    if any(c.get("visual_estimate") for c in rep["kept"]):
        notes.append("Values read from charts are visual estimates, not exact figures.")
    if any(c.get("derived") for c in rep["kept"]):
        notes.append("Some figures were calculated from the cited values.")
    for c in cits:
        if c.get("note"):
            notes.append(f"[{c['num']}] {c['note']}")
    return _msg(body, rep["status"], cits, list(dict.fromkeys(notes)), timings, query=query,
                verification={k: rep[k] for k in ("status", "answer_ok")} | {
                    "claims_total": len(rep["claims"]), "claims_dropped": len(rep["dropped"])})
