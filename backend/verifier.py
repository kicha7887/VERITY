"""Deterministic answer verification (does not trust the LLM's self-assessment)."""
import re

NUM_RE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
MARK_RE = re.compile(r"\[E\d+\]")
STOP = set("this that with from have were which their there been also than into over such these those will "
           "report document states shows about during after before between other more most only".split())
MIN_OVERLAP = 0.4


def _norm_num(n):
    n = n.replace(",", "").rstrip(".")
    try:
        f = float(n)
        return str(int(f)) if f == int(f) else repr(f)
    except ValueError:
        return n


def numbers(text):
    return {_norm_num(n) for n in NUM_RE.findall(MARK_RE.sub("", text))}


def _tokens(text):
    return {w[:5] for w in re.findall(r"[a-z0-9]{4,}", text.lower()) if w not in STOP}


def check_claim(claim, evidence, selected, page_counts, visual_ids):
    reasons, ids = [], [i for i in claim.get("evidence") or [] if isinstance(i, str)]
    if not ids:
        return ["no evidence cited"]
    cited = []
    for i in ids:
        ev = evidence.get(i)
        if ev is None:
            reasons.append(f"cites unknown evidence {i}"); continue
        if ev["doc_id"] not in selected:
            reasons.append(f"{i} is not from a selected document")
        elif not 1 <= ev["page"] <= page_counts.get(ev["doc_id"], 0):
            reasons.append(f"{i} cites non-existent page {ev['page']}")
        else:
            cited.append(ev)
    if reasons or not cited:
        return reasons or ["no valid evidence"]
    text = claim["text"]
    visual_ok = claim.get("visual_estimate") and any(e["id"] in visual_ids for e in cited)
    if claim.get("visual_estimate") and not visual_ok:
        return ["visual claim without an analysed page image"]
    if visual_ok:
        return []
    ev_text = " ".join(e["text"] for e in cited)
    if not claim.get("derived"):
        missing = numbers(text) - numbers(ev_text)
        if missing:
            reasons.append("numbers not found in cited evidence: " + ", ".join(sorted(missing)))
    toks = _tokens(text)
    if toks and len(toks & _tokens(ev_text)) / len(toks) < MIN_OVERLAP:
        reasons.append("claim wording is not supported by the cited text")
    return reasons


def verify(result, evidence, selected, page_counts, visual_ids=frozenset()):
    """Returns {'status','claims','kept','dropped','answer_ok','conflicts'}."""
    checked = []
    for c in result.get("claims", []):
        r = check_claim(c, evidence, selected, page_counts, visual_ids)
        checked.append({**c, "ok": not r, "reasons": r})
    kept = [c for c in checked if c["ok"]]
    dropped = [c for c in checked if not c["ok"]]
    # answer-level check: every number in the prose must come from evidence cited by a kept claim (or be derived)
    cited_text = " ".join(evidence[i]["text"] for c in kept for i in c["evidence"] if i in evidence)
    allowed = numbers(cited_text)
    any_derived = any(c.get("derived") or c.get("visual_estimate") for c in kept)
    answer_ok = bool(result.get("answer")) and (numbers(result["answer"]) <= allowed or any_derived)
    answer_ok = answer_ok and all(m.strip("[]") in evidence for m in MARK_RE.findall(result.get("answer", "")))
    conflicts = [c for c in result.get("conflicts", [])
                 if c.get("description") and c.get("evidence") and all(i in evidence for i in c["evidence"])]
    status = "failed" if not kept else ("verified" if not dropped and answer_ok else "partial")
    return {"status": status, "claims": checked, "kept": kept, "dropped": dropped,
            "answer_ok": answer_ok, "conflicts": conflicts}
