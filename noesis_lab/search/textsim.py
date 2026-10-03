"""TF-IDF cosine and a 2-D PCA projection. Pure NumPy, deterministic; used for relevance ranking,
the T4 overlap check and the paper map. Text similarity only: never a verdict by itself."""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np


def tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def idf(docs: Sequence[str]) -> dict[str, float]:
    df = Counter(t for d in docs for t in set(tokens(d)))
    n = len(docs)
    return {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}


def vector(text: str, idf_: dict[str, float]) -> dict[str, float]:
    return {t: c * idf_[t] for t, c in Counter(tokens(text)).items() if t in idf_}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    na = math.sqrt(sum(v * v for v in a.values())) or 1.0
    nb = math.sqrt(sum(v * v for v in b.values())) or 1.0
    return round(sum(v * b.get(t, 0.0) for t, v in a.items()) / (na * nb), 6)


def similarities(query: str, docs: Sequence[str]) -> list[float]:
    """Cosine of `query` against each doc; IDF is computed over the docs only."""
    idf_ = idf(docs)
    q = vector(query, idf_)
    return [cosine(q, vector(d, idf_)) for d in docs]


def pca_2d(docs: Sequence[str]) -> list[list[float]]:
    """TF-IDF vectors -> first two principal components (SVD). Sign-fixed for determinism."""
    if len(docs) < 3:
        return [[0.0, 0.0] for _ in docs]
    idf_ = idf(docs)
    vocab = sorted(idf_)
    col = {t: i for i, t in enumerate(vocab)}
    m = np.zeros((len(docs), len(vocab)))
    for r, d in enumerate(docs):
        for t, v in vector(d, idf_).items():
            m[r, col[t]] = v
    m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
    m -= m.mean(axis=0)
    u, s, _ = np.linalg.svd(m, full_matrices=False)
    xy = u[:, :2] * s[:2]
    for j in range(2):                       # fix the sign so the map does not flip between runs
        if xy[np.argmax(np.abs(xy[:, j])), j] < 0:
            xy[:, j] = -xy[:, j]
    return [[round(float(a), 4), round(float(b), 4)] for a, b in xy]
