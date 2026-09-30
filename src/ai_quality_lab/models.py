from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class QueryGroundTruth:
    query_id: str
    question: str
    relevant_chunk_ids: frozenset[str]
    category: str | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    query_id: str
    rank: int
    chunk_id: str
    score: float | None = None
    latency_ms: float | None = None


def parse_chunk_ids(raw: object) -> frozenset[str]:
    if raw is None:
        return frozenset()
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return frozenset()
    return frozenset(part.strip() for part in text.split("|") if part.strip())


def validate_unique_ranks(rows: Iterable[RetrievedChunk]) -> None:
    seen: set[tuple[str, int]] = set()
    for row in rows:
        key = (row.query_id, row.rank)
        if key in seen:
            raise ValueError(f"Duplicate rank for query_id={row.query_id!r}, rank={row.rank}")
        seen.add(key)
