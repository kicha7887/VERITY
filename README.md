# VERITY — Every Answer Has Evidence

Evidence-first multimodal chatbot for government PDFs (**PS-03 · Multimodal RAG with Explainability**).
Upload reports → ask questions → get answers with **verified page-level citations**, or an explicit refusal.

## Quick start
```bash
python -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then set GEMINI_API_KEY (and GEMINI_MODEL if the default is unavailable)
# Optional OCR: install the Tesseract binary (Ubuntu: sudo apt install tesseract-ocr, macOS: brew install tesseract,
#               Windows: UB-Mannheim installer, then set TESSERACT_CMD in .env)
streamlit run app.py
```
Without `GEMINI_API_KEY` the app still runs in **retrieval-only mode** (shows retrieved passages, generates nothing).
The first run downloads `all-MiniLM-L6-v2`; if that fails (offline) VERITY falls back to TF-IDF retrieval and says so in Settings.

## Official dataset
```bash
python scripts/inspect_dataset.py     # lists files/modalities of Tarakeshwaran/CIT_Hackathon_PS03_rag, optional download
```
Inspect first, then add real questions + expected pages to `tests/test_questions.json`. Nothing here assumes the
dataset's structure. `tests/make_sample_pdf.py` builds a **synthetic supplementary** PDF (text, table, vector chart,
scanned page) used only for testing. It is not part of the official dataset.

## Tests & evaluation
```bash
python -m pytest -q tests                 # unit + integration tests
python -m evaluation.evaluate             # writes evaluation/results.json (measured values only)
```
Answer-level metrics (citation correctness, faithfulness, refusal, success rate) need `GEMINI_API_KEY`; otherwise they
are reported as `null` / "not measured". The **Evaluation** view only shows what is in `results.json`.

## Architecture
```
PDF -> pdf_processor (PyMuPDF text, OCR fallback, chart/caption detection) + table_extractor (pdfplumber)
    -> chunker (paragraph-aware, page-bounded, tables kept as rows) -> embeddings (MiniLM) -> retriever (FAISS | sklearn | TF-IDF)
Question -> generator.rewrite_query -> retrieve (+ previous-turn evidence for follow-ups)
         -> [visual_analyzer: page images only for visual questions] -> generator (Gemini, JSON claims citing [E#])
         -> verifier (deterministic) -> citation_manager (page/passage validation) -> answer | refusal
```
**Verification (not LLM self-grading):** every claim must cite known evidence from a *selected* document with an existing
page; numbers in a claim must appear in the cited text (unless flagged as a visual estimate on an analysed image, or
derived); claim wording must overlap the cited text; every citation passage must be found on its stated page. Failing
claims are removed; if nothing survives, the answer is withheld and VERITY refuses.

**Isolation:** one `DocumentManager`/`Retriever` per session ID, per-session directories, sanitized filenames, content-hash
document IDs. Document text/images are wrapped as untrusted data in the prompt.

**Interface:** `DocumentManager.add_upload(name, bytes, progress_cb)` and
`pipeline.answer(dm, question, doc_ids, history, Settings)` -> `{content, status, citations[], notes[], timings}`.
Dev 1 owns `backend/`, Dev 2 owns `app.py` and `assets/`.

## Deployment (Streamlit Community Cloud)
Push to GitHub (`.env` and data are git-ignored) -> share.streamlit.io -> select `app.py` -> add `GEMINI_API_KEY` /
`GEMINI_MODEL` under *Secrets*. `packages.txt` installs Tesseract. Free tiers are memory-limited; if
`sentence-transformers` does not fit, retrieval degrades to TF-IDF automatically.

## Known limitations
- Gemini calls were not exercised in the build environment (no API access there); the Gemini path was tested with a
  stubbed model response. **Run one real query before the demo.**
- The official dataset was unreachable from the build environment, so it is untested; only the synthetic PDF was used.
- Table pages appear twice in the index (page text and structured table); both cite correctly.
- Page-image analysis is capped at 3 pages per question; chart values are labelled visual estimates unless printed.
- Highlights use text search; scanned pages show the full page without a highlight. OCR quality depends on the scan.
- Summary questions use passages sampled across the document, not a full read (stated in the answer).
- Conversation history is temporary (browser session). Uploaded files stay on server disk until removed or 24 h.

## Demo script (5 min)
1. Welcome screen, then upload an official PDF (Uploaded, Processing, Ready). 2. "Summarize this report." and open **Evidence**.
3. **Inspect original page** (highlighted source: the key moment). 4. A numerical table question. 5. "Explain the charts…".
6. A cross-page question, then the follow-up "Which page contains the second statistic?".
7. Ask something absent from the document to show the refusal. 8. Upload a second PDF and compare; note per-document citations.
9. Open the Evaluation view.
