"""Vector retrieval: FAISS (or scikit-learn NN) over MiniLM embeddings; TF-IDF fallback. One index per document."""
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from . import config, embeddings

try:
    import faiss
except Exception:  # pragma: no cover
    faiss = None


class DocIndex:
    def __init__(self, chunks, vectors=None):
        self.chunks = chunks
        self.vectors = vectors
        self.dense = None
        if vectors is not None and len(vectors):
            if faiss is not None:
                self.dense = faiss.IndexFlatIP(vectors.shape[1])
                self.dense.add(vectors)
                self.dense_name = "faiss"
            else:
                self.dense = NearestNeighbors(metric="cosine", algorithm="brute").fit(vectors)
                self.dense_name = "sklearn"
        self._tfidf = None

    @classmethod
    def build(cls, chunks):
        return cls(chunks, embeddings.embed([c["text"] for c in chunks]) if chunks else None)

    def _tfidf_search(self, query, k):
        if self._tfidf is None:
            v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
            try:
                self._tfidf = (v, v.fit_transform([c["text"] for c in self.chunks]))
            except ValueError:  # empty vocabulary
                self._tfidf = (None, None)
        v, X = self._tfidf
        if v is None:
            return []
        scores = (X @ v.transform([query]).T).toarray().ravel()
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0], "tfidf"

    def search(self, query, k):
        if not self.chunks:
            return [], "none"
        k = min(k, len(self.chunks))
        if self.dense is not None:
            q = embeddings.embed([query])
            if q is not None:
                if self.dense_name == "faiss":
                    s, idx = self.dense.search(q, k)
                    return [(int(i), float(x)) for i, x in zip(idx[0], s[0]) if i >= 0], "dense"
                d, idx = self.dense.kneighbors(q, n_neighbors=k)
                return [(int(i), 1 - float(x)) for i, x in zip(idx[0], d[0])], "dense"
        return self._tfidf_search(query, k) or ([], "tfidf")


class Retriever:
    """Holds the indexes of ONE session only, so cross-session retrieval is structurally impossible."""

    def __init__(self):
        self.indexes = {}

    def add(self, doc_id, chunks, vectors=None):
        self.indexes[doc_id] = DocIndex(chunks, vectors) if vectors is not None else DocIndex.build(chunks)
        return self.indexes[doc_id]

    def remove(self, doc_id):
        self.indexes.pop(doc_id, None)

    def search(self, query, doc_ids, top_k=None, min_score=None):
        """Returns (hits, below_threshold_hits). Hits carry score + mode."""
        top_k = top_k or config.TOP_K
        doc_ids = [d for d in doc_ids if d in self.indexes]
        hits, below = [], []
        for d in doc_ids:
            res, mode = self.indexes[d].search(query, top_k * 2)
            thr = min_score if min_score else (config.MIN_SCORE_DENSE if mode == "dense" else config.MIN_SCORE_TFIDF)
            for i, s in res:
                ev = dict(self.indexes[d].chunks[i], score=round(s, 4), mode=mode)
                (hits if s >= thr else below).append(ev)
        hits.sort(key=lambda e: -e["score"]); below.sort(key=lambda e: -e["score"])
        if len(doc_ids) > 1:  # guarantee per-document coverage for comparisons
            per = max(1, math.ceil(top_k / len(doc_ids)))
            chosen = []
            for d in doc_ids:
                chosen += [h for h in hits if h["doc_id"] == d][:per]
            ids = {h["chunk_id"] for h in chosen}
            chosen += [h for h in hits if h["chunk_id"] not in ids]
            hits = sorted(chosen[:top_k], key=lambda e: -e["score"])
        return hits[:top_k], below[:3]

    def overview(self, doc_ids, n=8):
        """Evenly sampled passages for 'summarize' style requests (no relevance threshold applies)."""
        out = []
        for d in doc_ids:
            if d not in self.indexes:
                continue
            cs = [c for c in self.indexes[d].chunks if c["type"] in ("text", "table", "chart")]
            if not cs:
                continue
            step = max(1, len(cs) / n)
            picks = {int(i * step) for i in range(min(n, len(cs)))}
            out += [dict(cs[i], score=None, mode="overview") for i in sorted(picks)]
        return out
