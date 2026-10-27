"""Гибридный поиск: векторный (bge-m3) + BM25, слияние по рангам (RRF)."""
import numpy as np

from .. import config
from ..schema import Hit
from .index import Index


def rrf(rankings: list[list[tuple[str, float]]], k: int = config.RRF_K) -> dict[str, float]:
    """Reciprocal Rank Fusion: score = Σ 1 / (k + ранг). Не зависит от шкал скоров."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (cid, _) in enumerate(ranking):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return fused


def hybrid_search(index: Index, question: str, qvec: np.ndarray,
                  companies: list[str] | None, k: int) -> list[Hit]:
    dense = index.dense_search(qvec, companies, k)
    sparse = index.bm25_search(question, companies, k)
    fused = rrf([dense, sparse])
    d, s = dict(dense), dict(sparse)
    ids = [i for i in sorted(fused, key=fused.get, reverse=True) if i in index.chunks][:k]
    return [Hit(index.chunks[i], fused[i], d.get(i), s.get(i)) for i in ids]
