from backend import citation_manager as cm


class FakeDM:
    docs = {"d": {}}
    page_count = staticmethod(lambda d: 3)
    page_text = staticmethod(lambda d, p: "Household access rose to 78.4% in 2023. Budget details follow.")


def ev(**kw):
    base = {"id": "E1", "chunk_id": "c1", "doc_id": "d", "page": 1, "filename": "f.pdf", "type": "text",
            "text": "Household access rose to 78.4% in 2023.", "extra": {}}
    return {**base, **kw}


def test_valid_citation_built():
    cits, m, rej = cm.build_citations(["E1"], {"E1": ev()}, FakeDM)
    assert cits[0]["page"] == 1 and cits[0]["located"] and m == {"E1": 1} and not rej


def test_nonexistent_page_rejected():
    _, _, rej = cm.build_citations(["E1"], {"E1": ev(page=9)}, FakeDM)
    assert rej == ["E1"]


def test_fabricated_passage_rejected():
    _, _, rej = cm.build_citations(["E1"], {"E1": ev(text="Completely different invented sentence.")}, FakeDM)
    assert rej == ["E1"]


def test_unknown_document_rejected():
    _, _, rej = cm.build_citations(["E1"], {"E1": ev(doc_id="zzz")}, FakeDM)
    assert rej == ["E1"]


def test_renumber_markers():
    assert cm.renumber("A [E2] and B [E1] and C [E7].", {"E1": 1, "E2": 2}) == "A [2] and B [1] and C."
