"""VERITY - Streamlit front end. Every answer has evidence."""
import html
import json
import time
import uuid
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer

from backend import config, embeddings, generator, pipeline
from backend.document_manager import DocumentManager, UploadError
from backend.pdf_processor import render_page

ROOT = Path(__file__).parent
LOGO, USER = str(ROOT / "assets/logo.png"), str(ROOT / "assets/user.png")
SUGGESTIONS = ["Summarize this report.", "What are the most important findings?", "Find the key statistics.",
               "Explain the charts in this document.", "Compare information across different pages.",
               "Show me the evidence for a particular conclusion."]
STATUS_LABEL = {"verified": "✓ Verified against evidence", "partial": "◐ Partially verified",
                "refused": "⚠ Insufficient evidence", "retrieval_only": "Retrieval only (no generated answer)",
                "error": "⚠ Action needed"}

st.set_page_config(page_title="VERITY", page_icon=LOGO, layout="wide", initial_sidebar_state="expanded")
st.markdown(f"<style>{(ROOT / 'assets/style.css').read_text()}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------- state
def init():
    ss = st.session_state
    if "sid" not in ss:
        ss.sid = uuid.uuid4().hex[:16]
        DocumentManager.cleanup_old()
        ss.dm = DocumentManager(ss.sid)
        ss.convs, ss.cur, ss.pending, ss.up_key, ss.view = {}, None, None, 0, "Chat"
        new_conv()


def new_conv():
    cid = uuid.uuid4().hex[:8]
    st.session_state.convs[cid] = {"title": "New conversation", "messages": []}
    st.session_state.cur = cid


def conv():
    return st.session_state.convs[st.session_state.cur]


def selected_docs():
    dm = st.session_state.dm
    return [d for d in dm.ready_ids() if st.session_state.get(f"sel_{d}", True)]


def settings():
    ss = st.session_state
    return pipeline.Settings(top_k=ss.get("top_k", config.TOP_K), min_score=ss.get("min_score", 0.0),
                             use_visual=ss.get("use_visual", True))


# ---------------------------------------------------------------- ingestion
def ingest(name, data):
    dm = st.session_state.dm
    with st.status(f"Processing {name}…", expanded=True) as status:
        bar = st.progress(0.0)
        try:
            rec = dm.add_upload(name, data, lambda f, l: bar.progress(min(f, 1.0), text=l))
        except UploadError as e:
            status.update(label=str(e), state="error"); return
        if rec["status"] == "ready":
            status.update(label=f"{rec['filename']} is ready ({rec['pages']} pages)", state="complete")
        else:
            status.update(label=f"{rec['filename']} failed: {rec['error']}", state="error")


# ---------------------------------------------------------------- dialogs & renderers
@st.dialog("Source page", width="large")
def page_dialog(c):
    st.markdown(f"**{c['filename']}** · Page {c['page']} · {c['type'].title()}")
    try:
        png, found = render_page(st.session_state.dm.pdf_path(c["doc_id"]), c["page"], c["text"], c["type"])
        st.image(png, use_container_width=True)
        st.caption("The gold highlight marks the cited evidence." if found else
                   "Exact region could not be highlighted (scanned/image content) — the full page is shown.")
    except Exception:
        st.error("The original page could not be rendered.")


def set_feedback(msg, val):
    msg["feedback"] = val


def render_assistant(msg, mid):
    st.markdown(f"<span class='status s-{msg['status']}'>{STATUS_LABEL.get(msg['status'], '')}</span>",
                unsafe_allow_html=True)
    st.markdown(msg["content"])
    for n in msg.get("notes", []):
        st.caption("ℹ " + n)
    cits = msg.get("citations", [])
    if cits:
        with st.expander(f"Evidence · {len(cits)} source{'s' if len(cits) > 1 else ''}", expanded=False):
            for c in cits:
                st.markdown(f"<div class='cite-head'>[{c['num']}] {html.escape(c['filename'])}</div>", unsafe_allow_html=True)
                meta = [f"Page {c['page']}", c["type"].title(), f"extracted via {c['method']}"]
                if c.get("score") is not None:
                    meta.append(f"relevance {c['score']:.2f}")
                if c.get("ocr_confidence") is not None:
                    meta.append(f"OCR confidence {c['ocr_confidence']:.0f}/100")
                meta.append("passage located on page ✓" if c.get("located") else "page-level reference")
                st.markdown(f"<div class='cite-meta'>{' · '.join(meta)}</div>", unsafe_allow_html=True)
                if c["type"] == "table":
                    st.markdown(c["text"])
                else:
                    st.markdown(f"<div class='quote'>{html.escape(c['text'][:1500])}</div>", unsafe_allow_html=True)
                if st.button("🔍 Inspect original page", key=f"insp_{mid}_{c['num']}"):
                    page_dialog(c)
                st.divider()
    t = msg.get("timings", {})
    c1, c2, c3, c4 = st.columns([1.2, 0.6, 0.6, 5])
    with c1.popover("Copy answer"):
        st.code(msg["content"], language=None)
    c2.button("👍", key=f"up_{mid}", on_click=set_feedback, args=(msg, "up"), help="Helpful")
    c3.button("👎", key=f"dn_{mid}", on_click=set_feedback, args=(msg, "down"), help="Not helpful")
    if t.get("total_s") is not None:
        c4.caption(f"{t['total_s']}s total · feedback: {msg.get('feedback') or '—'} (kept in this session only)")


def render_messages():
    for i, m in enumerate(conv()["messages"]):
        with st.chat_message(m["role"], avatar=USER if m["role"] == "user" else LOGO):
            if m["role"] == "user":
                st.markdown(m["content"])
            else:
                render_assistant(m, f"{st.session_state.cur}_{i}")


def welcome():
    st.markdown("""<div class='welcome'>
    <div class='welcome-badge'>Verified AI · Multimodal RAG with Explainability</div>
    <h1>Welcome to VERITY</h1>
    <div class='sub'>Your trusted assistant for understanding government reports and complex documents.</div>
    <div class='sup'>Upload a document, ask a question, and explore answers backed by verifiable evidence.</div></div>""",
                unsafe_allow_html=True)
    cols = st.columns(3)
    for i, q in enumerate(SUGGESTIONS):
        cols[i % 3].button(q, key=f"sg{i}", use_container_width=True,
                           on_click=lambda q=q: st.session_state.update(pending=q))


# ---------------------------------------------------------------- sidebar
def sidebar():
    ss, dm = st.session_state, st.session_state.dm
    with st.sidebar:
        c1, c2 = st.columns([1, 3])
        c1.image(LOGO, width=48)
        c2.markdown("<p class='brand'>VERITY</p>", unsafe_allow_html=True)
        st.markdown("<div class='tagline'>Every Answer Has Evidence. Every Insight Has a Source.</div>", unsafe_allow_html=True)
        if st.button("＋  New Chat", key="newchat"):
            new_conv(); ss.view = "Chat"; st.rerun()
        st.markdown("<div class='side-label'>Document Library</div>", unsafe_allow_html=True)
        ups = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True,
                               key=f"up_{ss.up_key}", label_visibility="collapsed")
        if ups:
            for f in ups:
                ingest(f.name, f.getvalue())
            ss.up_key += 1; st.rerun()
        for r in dm.list_docs():
            row = st.columns([0.12, 0.88])
            if r["status"] == "ready":
                row[0].checkbox("sel", value=True, key=f"sel_{r['doc_id']}", label_visibility="collapsed")
            row[1].markdown(f"{html.escape(r['filename'])} <span class='badge b-{r['status']}'>{r['status'].title()}</span>",
                            unsafe_allow_html=True)
            if r["status"] == "ready":
                s = r["stats"]
                st.caption(f"{r['pages']} pages · {s['tables']} tables · {s['charts']} charts · {s['ocr_pages']} OCR pages")
            elif r["error"]:
                st.caption("⚠ " + r["error"])
            if st.button("Remove", key=f"rm_{r['doc_id']}"):
                dm.remove(r["doc_id"]); st.rerun()
        st.markdown("<div class='side-label'>Recent Conversations</div>", unsafe_allow_html=True)
        for cid, c in reversed(list(ss.convs.items())):
            if st.button(("● " if cid == ss.cur else "") + c["title"][:34], key=f"cv_{cid}"):
                ss.cur = cid; ss.view = "Chat"; st.rerun()
        st.caption("History is **temporary**: kept only for this browser session.")
        if st.button("Clear current conversation", key="clearconv"):
            conv()["messages"] = []; conv()["title"] = "New conversation"; st.rerun()
        st.markdown("<div class='side-label'>Settings</div>", unsafe_allow_html=True)
        with st.expander("Retrieval & AI settings"):
            st.slider("Passages retrieved (top-k)", 2, 12, config.TOP_K, key="top_k")
            st.slider("Relevance threshold (0 = automatic)", 0.0, 0.6, 0.0, 0.01, key="min_score")
            st.toggle("Use page images for charts (Gemini vision)", True, key="use_visual")
            st.caption(f"Embeddings: {embeddings.status()}")
            st.caption(f"Gemini: {'configured · ' + config.GEMINI_MODEL if generator.available() else 'NOT configured (set GEMINI_API_KEY)'}")
            if st.button("Delete all my documents", key="wipe"):
                dm.clear_session(); st.rerun()
        with st.expander("About VERITY"):
            st.caption("VERITY retrieves evidence from your PDFs, answers only from it, verifies every citation "
                       "deterministically, and refuses when evidence is insufficient. Uploaded files are stored "
                       "on the server under a private session ID and removed when you delete them or after "
                       f"{config.SESSION_MAX_AGE_HOURS} h. PS-03 · Multimodal RAG with Explainability.")


# ---------------------------------------------------------------- chat flow
def handle_question(text):
    ss, dm = st.session_state, st.session_state.dm
    c = conv()
    if c["title"] == "New conversation":
        c["title"] = text[:48]
    history = [{"role": m["role"], "content": m["content"], "citations": m.get("citations")} for m in c["messages"]]
    c["messages"].append({"role": "user", "content": text})
    with st.chat_message("user", avatar=USER):
        st.markdown(text)
    with st.chat_message("assistant", avatar=LOGO):
        with st.spinner("Searching the documents and verifying evidence…"):
            try:
                res = pipeline.answer(dm, text, selected_docs(), history, settings())
            except Exception as e:  # last-resort guard: never crash the UI
                res = {"content": f"Something went wrong while answering ({type(e).__name__}). Please try again.",
                       "status": "error", "citations": [], "notes": [], "timings": {}}
        c["messages"].append({"role": "assistant", **res})
        render_assistant(c["messages"][-1], f"{ss.cur}_{len(c['messages']) - 1}")


def chat_view():
    ss = st.session_state
    if not conv()["messages"] and not ss.pending:
        welcome()
    else:
        render_messages()
    val = st.chat_input("Ask about your documents… (Shift+Enter for a new line)", accept_file="multiple",
                        file_type=["pdf"], key="composer")
    text, files = None, []
    if isinstance(val, str):
        text = val
    elif val:
        text, files = (val.get("text") or "").strip(), list(val.get("files") or [])
    for f in files:
        ingest(f.name, f.getvalue())
    if files and not text:
        st.rerun()
    text = text or ss.pending
    ss.pending = None
    if text:
        handle_question(text)


# ---------------------------------------------------------------- insights
def insights_view():
    dm, docs = st.session_state.dm, selected_docs()
    if not docs:
        st.info("Upload and select a ready document to see insights."); return
    st.markdown("### Document insights")
    st.caption("Derived only from the processed content of the selected documents.")
    chunks = [c for d in docs for c in dm.chunks(d)]
    cols = st.columns(4)
    cols[0].metric("Documents", len(docs)); cols[1].metric("Pages", sum(dm.page_count(d) for d in docs))
    cols[2].metric("Tables", sum(c["type"] == "table" for c in chunks)); cols[3].metric("Chart captions", sum(c["type"] == "chart" for c in chunks))
    if st.button("Generate summary & key findings (Gemini)", disabled=not generator.available()):
        with st.spinner("Summarising with evidence…"):
            res = pipeline.answer(dm, "Summarize this report and list the key findings.", docs, None, settings())
        with st.container(border=True):
            render_assistant({**res, "feedback": None}, "insights")
    elif not generator.available():
        st.caption("Summary generation needs GEMINI_API_KEY.")
    import re
    stat_re = re.compile(r"[^.!?\n]*\d[\d,.]*\s?(%|per cent|percent|million|billion|crore|lakh|thousand)[^.!?\n]*[.!?]?", re.I)
    stats, seen = [], set()
    for c in chunks:
        if c["type"] != "text":
            continue
        for m in stat_re.finditer(c["text"]):
            s = " ".join(m.group().split())
            if 25 < len(s) < 300 and s not in seen:
                seen.add(s); stats.append((s, c["filename"], c["page"]))
    st.markdown("#### Extracted statistics (sentences containing figures)")
    for s, f, p in stats[:10]:
        st.markdown(f"- {s}  \n  <span class='cite-meta'>{html.escape(f)} · page {p}</span>", unsafe_allow_html=True)
    if not stats:
        st.caption("No sentences with percentage/quantity figures were found.")
    texts = [c["text"] for c in chunks if c["type"] == "text"]
    if len(texts) >= 2:
        try:
            cv = CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=15, min_df=1)
            X = cv.fit_transform(texts)
            df = pd.DataFrame({"term": cv.get_feature_names_out(), "mentions": X.sum(axis=0).A1}).sort_values("mentions")
            st.markdown("#### Frequently mentioned terms")
            fig = px.bar(df, x="mentions", y="term", orientation="h", color_discrete_sequence=["#D4A373"])
            fig.update_layout(
                plot_bgcolor="#211714",
                paper_bgcolor="#211714",
                font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", color="#F5EBE1", size=12),
                height=380,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(gridcolor="#3A2922", zerolinecolor="#3A2922"),
                yaxis=dict(gridcolor="#3A2922")
            )
            st.plotly_chart(fig, use_container_width=True)
        except ValueError:
            pass
    tables = [c for c in chunks if c["type"] == "table"]
    st.markdown(f"#### Extracted tables ({len(tables)})")
    for c in tables:
        with st.expander(f"{c['filename']} · page {c['page']}"):
            rows = (c.get("extra") or {}).get("rows") or []
            if len(rows) > 1:
                head = [f"{h or 'col'}_{i}" if rows[0].count(h) > 1 or not h else h for i, h in enumerate(rows[0])]
                st.dataframe(pd.DataFrame([r + [""] * (len(head) - len(r)) for r in rows[1:]], columns=head))
    charts = [c for c in chunks if c["type"] == "chart"]
    if charts:
        st.markdown("#### Identified charts / figures")
        for c in charts:
            st.markdown(f"- {html.escape(c['text'])} — {html.escape(c['filename'])}, page {c['page']}")


def eval_view():
    f = ROOT / "evaluation/results.json"
    st.markdown("### Evaluation (measured results only)")
    if not f.exists():
        st.info("No results yet. Run `python -m evaluation.evaluate` — values below are never pre-filled."); return
    r = json.loads(f.read_text())
    st.caption(f"Embedding backend: {r['environment']['embedding_backend']} · Gemini used: {r['environment']['gemini_used']}"
               + (f" · {r['environment']['note']}" if r["environment"].get("note") else ""))
    agg = {k: v for k, v in r["aggregate"].items() if k != "processing_s"}
    st.dataframe(pd.DataFrame({"metric": agg.keys(), "value": [("not measured" if v is None else v) for v in agg.values()]}),
                 hide_index=True)
    st.markdown("**Per question**")
    st.dataframe(pd.DataFrame(r["questions"]).drop(columns=["answer_preview"], errors="ignore"), hide_index=True)


# ---------------------------------------------------------------- main
init()
sidebar()
ss = st.session_state
docs = selected_docs()
tb1, tb2 = st.columns([3, 2], vertical_alignment="center")
tb1.markdown(f"<div class='topbar'><div><span class='title'>{html.escape(conv()['title'])}</span><br>"
             f"<span class='meta'>📄 {len(docs)} of {len(ss.dm.ready_ids())} ready document(s) selected</span></div></div>",
             unsafe_allow_html=True)
ss.view = tb2.radio("View", ["Chat", "Insights", "Evaluation"], horizontal=True,
                    index=["Chat", "Insights", "Evaluation"].index(ss.view), label_visibility="collapsed")
{"Chat": chat_view, "Insights": insights_view, "Evaluation": eval_view}[ss.view]()
