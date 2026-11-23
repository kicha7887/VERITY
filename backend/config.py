"""Central configuration. Secrets come only from environment variables / .env."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
UPLOADS, PROCESSED, INDEXES = DATA / "uploads", DATA / "processed", DATA / "indexes"
for _p in (UPLOADS, PROCESSED, INDEXES):
    _p.mkdir(parents=True, exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
# Model availability/quota changes over time: always override via GEMINI_MODEL if this default is unavailable.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "").strip() or "gemini-3.5-flash-lite"

EMBED_MODEL = "all-MiniLM-L6-v2"
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "400"))
MIN_TEXT_CHARS = 30          # below this a page is treated as scanned -> OCR
CHUNK_SIZE, CHUNK_OVERLAP = 900, 150
TABLE_MAX_CHARS = 2500
TOP_K = 6
MIN_SCORE_DENSE = 0.25       # cosine, MiniLM
MIN_SCORE_TFIDF = 0.06       # cosine, TF-IDF fallback
MAX_VISUAL_PAGES = 3
VISUAL_DPI, PREVIEW_DPI = 110, 130
SESSION_MAX_AGE_HOURS = 24

REFUSAL = ("I couldn't find sufficient evidence in the uploaded documents "
           "to answer this question reliably.")
