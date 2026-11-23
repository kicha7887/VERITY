"""Table extraction with pdfplumber. Rows/columns are preserved."""
import pdfplumber


def _clean(c):
    return " ".join(str(c or "").split())


def to_markdown(rows):
    if not rows:
        return ""
    w = max(len(r) for r in rows)
    rows = [r + [""] * (w - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * w]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def extract_tables(path, progress=None):
    """Return [{page, rows, markdown}] — page numbers are 1-based original pages."""
    found = []
    with pdfplumber.open(path) as pdf:
        total = len(pdf.pages)
        for i, page in enumerate(pdf.pages, start=1):
            try:
                tables = page.extract_tables()
            except Exception:
                tables = []
            for t in tables:
                rows = [[_clean(c) for c in r] for r in t if r and any(_clean(c) for c in r)]
                if len(rows) >= 2 and max(len(r) for r in rows) >= 2:
                    found.append({"page": i, "rows": rows, "markdown": to_markdown(rows)})
            if progress:
                progress(i, total)
    return found
