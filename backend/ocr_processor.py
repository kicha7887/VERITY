"""Optional Tesseract OCR. Never raises if Tesseract is missing."""
import os
import shutil
from collections import defaultdict
from PIL import Image, ImageOps

try:
    import pytesseract
    if os.getenv("TESSERACT_CMD"):
        pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD")
except Exception:  # pragma: no cover
    pytesseract = None


def ocr_available() -> bool:
    cmd = os.getenv("TESSERACT_CMD") or shutil.which("tesseract")
    return pytesseract is not None and bool(cmd)


def ocr_image(img: Image.Image):
    """Return (text, mean_confidence 0-100). Paragraphs are preserved."""
    gray = ImageOps.autocontrast(ImageOps.grayscale(img))
    d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    lines, confs = defaultdict(list), []
    for i, w in enumerate(d["text"]):
        w = w.strip()
        try:
            c = float(d["conf"][i])
        except (TypeError, ValueError):
            c = -1
        if w and c >= 0:
            lines[(d["block_num"][i], d["par_num"][i], d["line_num"][i])].append(w)
            confs.append(c)
    paras, cur_key, cur = [], None, []
    for key in sorted(lines):
        if cur_key and key[:2] != cur_key[:2]:
            paras.append(" ".join(cur)); cur = []
        cur.append(" ".join(lines[key])); cur_key = key
    if cur:
        paras.append(" ".join(cur))
    return "\n\n".join(paras), (sum(confs) / len(confs) if confs else 0.0)
