from backend import verifier

EV = {"E1": {"id": "E1", "doc_id": "d", "page": 2, "type": "table", "text": "| District | Households 2022 | Households 2023 |\n|---|---|---|\n| Southgate | 15,250 | 18,100 |"},
      "E2": {"id": "E2", "doc_id": "d", "page": 1, "type": "text",
             "text": "Household access to safe drinking water rose to 78.4% in 2023."}}
V = lambda res, vis=frozenset(): verifier.verify(res, EV, {"d"}, {"d": 5}, vis)


def claim(text, ev, **kw):
    return {"text": text, "evidence": ev, **kw}


def test_supported_claim_passes():
    r = V({"answer": "Southgate had 18,100 households [E1].", "claims": [claim("Southgate had 18,100 households connected in 2023", ["E1"])]})
    assert r["status"] == "verified"


def test_unsupported_number_dropped():
    r = V({"answer": "x [E1]", "claims": [claim("Southgate had 19,000 households connected", ["E1"])]})
    assert r["status"] == "failed" and "19000" in r["dropped"][0]["reasons"][0]


def test_unknown_evidence_and_bad_page():
    assert V({"answer": "a", "claims": [claim("water access 78.4%", ["E9"])]})["status"] == "failed"
    ev = {"E1": dict(EV["E1"], page=99)}
    assert verifier.verify({"answer": "a", "claims": [claim("Southgate 18,100", ["E1"])]}, ev, {"d"}, {"d": 5})["status"] == "failed"


def test_wrong_document_rejected():
    assert verifier.verify({"answer": "a", "claims": [claim("Southgate 18,100", ["E1"])]}, EV, {"other"}, {"d": 5})["status"] == "failed"


def test_partial_keeps_only_supported():
    r = V({"answer": "x", "claims": [claim("Access rose to 78.4% in 2023 water drinking", ["E2"]),
                                       claim("Budget was 999 million", ["E2"])]})
    assert r["status"] == "partial" and len(r["kept"]) == 1 and len(r["dropped"]) == 1


def test_visual_estimate_needs_image():
    c = claim("Bar is approximately 45", ["E1"], visual_estimate=True)
    assert V({"answer": "a", "claims": [c]})["status"] == "failed"
    assert V({"answer": "a", "claims": [c]}, {"E1"})["status"] in ("verified", "partial")


def test_number_formatting_equivalence():
    assert verifier.numbers("412.0 and 12,400") == verifier.numbers("412 12400")
