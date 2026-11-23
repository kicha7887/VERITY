import pytest
from backend.document_manager import DocumentManager, UploadError
from tests.make_sample_pdf import build


@pytest.fixture(scope="module")
def dm(tmp_path_factory):
    path = build(str(tmp_path_factory.mktemp("p") / "s.pdf"))
    m = DocumentManager("pytest_a")
    m.clear_session(); m.__init__("pytest_a")
    rec = m.add_upload("s.pdf", open(path, "rb").read())
    assert rec["status"] == "ready"
    yield m
    m.clear_session()


def test_retrieves_correct_page(dm):
    hits, _ = dm.retriever.search("households connected Southgate 2023", dm.ready_ids())
    assert any(h["page"] == 2 and h["type"] == "table" for h in hits)


def test_irrelevant_query_returns_no_hits(dm):
    hits, _ = dm.retriever.search("quantum chromodynamics lagrangian", dm.ready_ids())
    assert hits == []


def test_session_isolation(dm):
    other = DocumentManager("pytest_b")
    assert other.ready_ids() == []
    hits, _ = other.retriever.search("water access", dm.ready_ids())
    assert hits == []
    other.clear_session()


def test_upload_validation(dm):
    with pytest.raises(UploadError):
        dm.add_upload("x.txt", b"hello")
    with pytest.raises(UploadError):
        dm.add_upload("x.pdf", b"not a pdf")


def test_failed_doc_never_indexed(dm):
    rec = dm.add_upload("bad.pdf", b"%PDF-1.4 garbage")
    assert rec["status"] == "failed" and rec["doc_id"] not in dm.ready_ids()
    dm.remove(rec["doc_id"])
