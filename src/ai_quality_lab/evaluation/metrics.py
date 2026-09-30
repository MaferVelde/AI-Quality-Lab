from __future__ import annotations

import math
from collections.abc import Sequence


def _top_k(retrieved: Sequence[str], k: int) -> list[str]:
    if k <= 0:
        raise ValueError("k must be > 0")
    return list(retrieved[:k])


def recall_at_k(relevant: set[str] | frozenset[str], retrieved: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = _top_k(retrieved, k)
    return len(set(top) & set(relevant)) / len(relevant)


def precision_at_k(relevant: set[str] | frozenset[str], retrieved: Sequence[str], k: int) -> float:
    top = _top_k(retrieved, k)
    if not top:
        return 0.0
    return len(set(top) & set(relevant)) / k


def hit_rate_at_k(relevant: set[str] | frozenset[str], retrieved: Sequence[str], k: int) -> float:
    top = _top_k(retrieved, k)
    return 1.0 if set(top) & set(relevant) else 0.0


def reciprocal_rank(relevant: set[str] | frozenset[str], retrieved: Sequence[str]) -> float:
    for idx, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(relevant: set[str] | frozenset[str], retrieved: Sequence[str], k: int) -> float:
    top = _top_k(retrieved, k)
    dcg = 0.0
    for idx, chunk_id in enumerate(top, start=1):
        rel = 1.0 if chunk_id in relevant else 0.0
        if rel:
            dcg += rel / math.log2(idx + 1)

    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / idcg
