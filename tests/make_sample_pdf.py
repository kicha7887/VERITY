"""Builds a SYNTHETIC supplementary test PDF (text, table, vector chart, scanned page).
It is NOT part of the official hackathon dataset."""
import io
import sys
from pathlib import Path
import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont

TABLE = [["District", "Households 2022", "Households 2023"], ["Northvale", "12,400", "14,950"],
         ["Eastmere", "9,800", "11,300"], ["Southgate", "15,250", "18,100"], ["Westfield", "7,600", "8,450"]]


def _para(page, y, text, size=11, width=480):
    page.insert_textbox(fitz.Rect(50, y, 50 + width, y + 160), text, fontsize=size, fontname="helv")


def build(path):
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((50, 60), "Rural Water Access Report 2023", fontsize=20)
    p.insert_text((50, 82), "SYNTHETIC SUPPLEMENTARY TEST DOCUMENT - not part of the official hackathon dataset", fontsize=8)
    _para(p, 110, "Executive summary. Household access to safe drinking water in the surveyed districts rose to 78.4% "
                  "in 2023, up from 71.2% in 2022. The programme budget for 2023 was 412 million rupees, of which "
                  "389 million rupees were spent.")
    p = doc.new_page()
    p.insert_text((50, 60), "Table 1: Households connected by district", fontsize=12)
    cw, rh = [140, 130, 130], 24
    for r, row in enumerate(TABLE):
        for c, cell in enumerate(row):
            x, y = 50 + sum(cw[:c]), 80 + r * rh
            p.draw_rect(fitz.Rect(x, y, x + cw[c], y + rh), color=(0, 0, 0), width=0.8)
            p.insert_text((x + 5, y + rh - 8), cell, fontsize=10)
    p = doc.new_page()
    p.insert_text((50, 60), "Figure 1: Households connected (thousands), 2020-2023", fontsize=12)
    base, vals = 400, {"2020": 36, "2021": 41, "2022": 45, "2023": 53}
    p.draw_line((80, base), (80, base - 260), width=1); p.draw_line((80, base), (420, base), width=1)
    for t in (0, 20, 40, 60):
        y = base - t * 4
        p.draw_line((75, y), (80, y)); p.insert_text((50, y + 3), str(t), fontsize=8)
    for i, (yr, v) in enumerate(vals.items()):
        x = 110 + i * 75
        p.draw_rect(fitz.Rect(x, base - v * 4, x + 45, base), color=(0.1, 0.2, 0.4), fill=(0.1, 0.2, 0.4))
        p.insert_text((x + 8, base + 14), yr, fontsize=9)
    for k in range(40):  # ensure vector-drawing detection threshold on this page
        p.draw_line((80, 140 + k * 6), (420, 140 + k * 6), color=(0.9, 0.9, 0.9), width=0.2)
    p.insert_text((50, 440), "Households connected (thousands)", fontsize=9)
    p = doc.new_page()
    p.insert_text((50, 60), "Methodology and notes", fontsize=14)
    _para(p, 90, "Preliminary field estimates published in March 2024 placed coverage at 77.9% for 2023; the audited "
                 "figure supersedes it. This report does not cover sanitation or electricity.")
    img = Image.new("RGB", (1200, 300), "white")
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
    except OSError:
        font = ImageFont.load_default()
    ImageDraw.Draw(img).text((40, 100), "Annex A scanned: Inspection of 14 pumping stations\ncompleted in November 2023.",
                             fill="black", font=font)
    buf = io.BytesIO(); img.save(buf, "PNG")
    p = doc.new_page()
    p.insert_image(fitz.Rect(30, 60, 570, 195), stream=buf.getvalue())
    doc.save(path); doc.close()
    return path


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/sample_verity_supplementary.pdf")
    print(build(str(out)))
