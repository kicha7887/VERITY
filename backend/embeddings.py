"""Local sentence-transformer embeddings. Returns None if the model cannot be loaded (retriever falls back to TF-IDF)."""
import numpy as np
from . import config

_model, _tried, _error = None, False, None


def _load():
    global _model, _tried, _error
    if not _tried:
        _tried = True
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(config.EMBED_MODEL)
        except Exception as e:  # missing package, offline, etc.
            _error = type(e).__name__
    return _model


def available() -> bool:
    return _load() is not None


def status() -> str:
    return config.EMBED_MODEL if available() else f"TF-IDF fallback (embedding model unavailable: {_error})"


def embed(texts):
    m = _load()
    if m is None:
        return None
    try:
        return np.asarray(m.encode(list(texts), normalize_embeddings=True, batch_size=32,
                                   show_progress_bar=False), dtype="float32")
    except Exception:
        return None
