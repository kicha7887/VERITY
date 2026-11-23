import pytest
from backend import config
from backend.chunker import chunk_items
from backend.ocr_processor import ocr_available
from backend.pdf_processor import PDFError, process_pdf, render_page
from tests.make_sample_pdf import build


@pytest.fixture(scope="module")
def pdf(tmp_path_factory):
    return build(str(tmp_path_factory.mktemp("pdf") / "s.pdf"))


@pytest.fixture(scope="module")
def processed(pdf):
    return process_pdf(pdf, "d1", "s.pdf")


def test_pages_and_text(processed):
    assert processed["metadata"]["pages"] == 5
    p1 = [i for i in processed["items"] if i["type"] == "text" and i["page"] == 1][0]
    assert "78.4%" in p1["text"] and p1["filename"] == "s.pdf" and p1["doc_id"] == "d1"


def test_table_extracted_with_structure(processed):
    t = [i for i in processed["items"] if i["type"] == "table"]
    assert t and t[0]["page"] == 2
    assert ["Southgate", "15,250", "18,100"] in t[0]["extra"]["rows"]


def test_chart_caption_detected(processed):
    assert any(i["type"] == "chart" and i["page"] == 3 for i in processed["items"])


@pytest.mark.skipif(not ocr_available(), reason="Tesseract not installed")
def test_ocr_on_scanned_page(processed):
    p5 = [i for i in processed["items"] if i["page"] == 5 and i["type"] == "text"]
    assert p5 and p5[0]["method"] == "tesseract" and "pumping" in p5[0]["text"].lower()


def test_chunks_keep_page_metadata(processed):
    for c in chunk_items(processed["items"]):
        assert c["page"] >= 1 and c["doc_id"] == "d1" and c["text"].strip()


def test_malformed_pdf(tmp_path):
    bad = tmp_path / "bad.pdf"; bad.write_bytes(b"%PDF-1.4 garbage")
    with pytest.raises(PDFError):
        process_pdf(str(bad), "x", "bad.pdf")


def test_render_highlight(pdf):
    png, found = render_page(pdf, 1, "Household access to safe drinking water", "text")
    assert png[:4] == b"\x89PNG" and found
