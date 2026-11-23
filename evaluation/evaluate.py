"""Measured evaluation. Usage:  python -m evaluation.evaluate [--questions tests/test_questions.json]
Retrieval metrics always run. Answer metrics (citations, faithfulness, refusal, success) require GEMINI_API_KEY;
without it they are reported as null — nothing is estimated or invented."""
import argparse
import json
import statistics
import time
from pathlib import Path
from backend import config, embeddings, generator, pipeline
from backend.document_manager import DocumentManager

ROOT = Path(__file__).resolve().parent.parent


def main(qpath, out):
    spec = json.loads(Path(qpath).read_text())
    dm = DocumentManager("eval_session"); dm.clear_session(); dm.__init__("eval_session")
    ids, proc_times = {}, {}
    for key, rel in spec["documents"].items():
        path = ROOT / rel
        if not path.exists() and path.name.startswith("sample_verity"):
            from tests.make_sample_pdf import build; build(str(path))
        t = time.time(); rec = dm.add_upload(path.name, path.read_bytes())
        proc_times[key] = round(time.time() - t, 2)
        if rec["status"] != "ready":
            raise SystemExit(f"{path.name} failed: {rec['error']}")
        ids[key] = rec["doc_id"]
    llm = generator.available()
    rows = []
    for q in spec["questions"]:
        did, exp = ids[q["doc"]], set(q.get("expected_pages", []))
        t = time.time(); hits, _ = dm.retriever.search(q["question"], [did]); r_time = time.time() - t
        pages = {h["page"] for h in hits}
        row = {"id": q["id"], "type": q["type"], "dataset": q.get("dataset"), "question": q["question"],
               "retrieved_pages": sorted(pages), "expected_pages": sorted(exp),
               "retrieval_hit": (bool(exp & pages) if exp else None), "retrieval_all_expected": (exp <= pages if exp else None),
               "retrieval_empty": not hits, "retrieval_s": round(r_time, 3)}
        if llm:
            res = pipeline.answer(dm, q["question"], [did]); cited = {c["page"] for c in res["citations"]}
            refused = res["status"] == "refused"
            row.update(status=res["status"], cited_pages=sorted(cited), total_s=res["timings"].get("total_s"),
                       answer_preview=res["content"][:200],
                       citation_correct=(exp <= cited if exp else None),
                       faithful=(res["status"] == "verified"),
                       keywords_present=all(k.lower() in res["content"].lower() for k in q.get("expected_keywords", [])),
                       refusal_correct=(refused == bool(q.get("should_refuse"))))
            row["success"] = row["refusal_correct"] if q.get("should_refuse") else bool(
                row["citation_correct"] and row["keywords_present"] and res["status"] in ("verified", "partial"))
            time.sleep(1)
        rows.append(row)

    def rate(key, subset=None):
        vals = [r[key] for r in (subset or rows) if r.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None
    answerable = [r for r in rows if r["expected_pages"]]
    refusals = [r for r in rows if r["type"] == "refusal"]
    agg = {"n_questions": len(rows),
           "retrieval_hit_rate": rate("retrieval_hit"), "retrieval_all_expected_rate": rate("retrieval_all_expected"),
           "retrieval_empty_on_refusal_questions": rate("retrieval_empty", refusals),
           "mean_retrieval_s": round(statistics.mean(r["retrieval_s"] for r in rows), 3),
           "processing_s": proc_times,
           "citation_correctness": rate("citation_correct", answerable) if llm else None,
           "faithfulness_verified_rate": rate("faithful", answerable) if llm else None,
           "refusal_accuracy": rate("refusal_correct", refusals) if llm else None,
           "overall_success_rate": rate("success") if llm else None,
           "mean_total_s": round(statistics.mean(r["total_s"] for r in rows if r.get("total_s")), 2) if llm else None}
    result = {"environment": {"embedding_backend": embeddings.status(), "gemini_used": llm,
                              "gemini_model": config.GEMINI_MODEL if llm else None,
                              "note": None if llm else "No GEMINI_API_KEY: answer-level metrics not measured."},
              "aggregate": agg, "questions": rows}
    Path(out).write_text(json.dumps(result, indent=2))
    dm.clear_session()
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "tests/test_questions.json"))
    ap.add_argument("--out", default=str(ROOT / "evaluation/results.json"))
    a = ap.parse_args(); main(a.questions, a.out)
