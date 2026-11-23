"""Hybrid retrieval over scheme documents.

Each scheme is split into section chunks (summary, benefits, eligibility,
documents, how to apply). A query is scored two ways and the scores fused:

* lexical  - TF-IDF over word unigrams/bigrams plus character n-grams, so exact
             terms ("PM-KISAN", "LPG") and misspellings both match
* semantic - dense vectors. Uses a sentence-transformers model when installed
             (``SAHAAYAK_EMBEDDINGS=st``); otherwise latent semantic analysis
             (TruncatedSVD on TF-IDF), which needs no download and runs offline

final = alpha * semantic + (1 - alpha) * lexical, after min-max normalising each.
A small synonym map expands common Hindi/colloquial words (kisan, ghar, ...).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

SECTIONS = [
    ("summary", "Overview"),
    ("benefits", "Benefits"),
    ("eligibility", "Eligibility"),
    ("documents", "Documents required"),
    ("how_to_apply", "How to apply"),
]

SYNONYMS = {
    "kisan": "farmer agriculture",
    "kheti": "farming agriculture",
    "ghar": "house housing",
    "makaan": "house housing",
    "awas": "house housing",
    "naukri": "job employment",
    "padhai": "education student scholarship",
    "chhatravritti": "scholarship",
    "beti": "girl child daughter",
    "ladki": "girl child daughter",
    "mahila": "woman women",
    "daughter": "girl child",
    "daughters": "girl child",
    "budhapa": "old age pension elderly",
    "bujurg": "old age elderly senior citizen",
    "ilaj": "health hospital treatment",
    "bimari": "health hospital illness",
    "karz": "loan credit",
    "rin": "loan credit",
    "gas": "lpg cooking gas",
    "dukaan": "shop business vendor",
    "thela": "street vendor cart",
    "garib": "poor bpl",
    "pregnant": "pregnancy maternity",
    "insurance": "insurance cover bima",
    "bima": "insurance cover",
}

_TOKEN = re.compile(r"[a-z0-9]+")


def expand_query(query: str) -> str:
    words = _TOKEN.findall(query.lower())
    extra = [SYNONYMS[w] for w in words if w in SYNONYMS]
    return " ".join([query.lower(), *extra])


@dataclass
class Chunk:
    scheme_id: str
    scheme_name: str
    section: str
    title: str
    text: str
    url: str


@dataclass
class Hit:
    chunk: Chunk
    score: float
    lexical: float
    semantic: float


def build_chunks(schemes: list[dict]) -> list[Chunk]:
    chunks = []
    for s in schemes:
        name = f"{s['name']} ({s['short_name']})"
        for key, title in SECTIONS:
            value = s.get(key)
            if not value:
                continue
            text = "; ".join(value) if isinstance(value, list) else value
            chunks.append(Chunk(s["id"], name, key, title, text, s["url"]))
    return chunks


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


class HybridRetriever:
    def __init__(self, schemes: list[dict], alpha: float = 0.5, backend: str | None = None):
        self.alpha = alpha
        self.chunks = build_chunks(schemes)
        corpus = [self._doc_text(c) for c in self.chunks]

        self.word_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.lex_matrix = normalize(hstack([self.word_vec.fit_transform(corpus),
                                            self.char_vec.fit_transform(corpus)]).tocsr())

        self.backend = backend or os.getenv("SAHAAYAK_EMBEDDINGS", "lsa")
        self._encoder = None
        if self.backend == "st":
            try:
                from sentence_transformers import SentenceTransformer  # optional dependency
                self._encoder = SentenceTransformer(os.getenv("SAHAAYAK_ST_MODEL", "all-MiniLM-L6-v2"))
                self.sem_matrix = self._encoder.encode(corpus, normalize_embeddings=True)
            except Exception:  # model unavailable: fall back to LSA
                self.backend = "lsa"
        if self.backend != "st":
            self.backend = "lsa"
            self.lsa_vec = TfidfVectorizer(sublinear_tf=True, stop_words="english")
            tfidf = self.lsa_vec.fit_transform(corpus)
            n_comp = max(2, min(64, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
            self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
            self.sem_matrix = normalize(self.svd.fit_transform(tfidf))

    @staticmethod
    def _doc_text(c: Chunk) -> str:
        return f"{c.scheme_name}. {c.title}: {c.text}"

    def _lexical(self, q: str) -> np.ndarray:
        qv = normalize(hstack([self.word_vec.transform([q]), self.char_vec.transform([q])]).tocsr())
        return (self.lex_matrix @ qv.T).toarray().ravel()

    def _semantic(self, q: str) -> np.ndarray:
        if self.backend == "st":
            qv = self._encoder.encode([q], normalize_embeddings=True)
            return (self.sem_matrix @ qv.T).ravel()
        qv = normalize(self.svd.transform(self.lsa_vec.transform([q])))
        return (self.sem_matrix @ qv.T).ravel()

    def search(self, query: str, k: int = 5, scheme_ids: set[str] | None = None) -> list[Hit]:
        q = expand_query(query)
        lex, sem = self._lexical(q), self._semantic(q)
        fused = self.alpha * _minmax(sem) + (1 - self.alpha) * _minmax(lex)
        order = np.argsort(-fused)
        hits = []
        for i in order:
            c = self.chunks[i]
            if scheme_ids is not None and c.scheme_id not in scheme_ids:
                continue
            hits.append(Hit(c, float(fused[i]), float(lex[i]), float(sem[i])))
            if len(hits) == k:
                break
        return hits
