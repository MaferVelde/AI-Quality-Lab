from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def adapt_retrieval_json(
    source_path: str | Path,
    chunks_path: str | Path,
    output_path: str | Path,
) -> dict[str, int | str]:
    """Convert NuevaMente retrieval JSON to AIQL canonical retrieval CSV.

    The source JSON is never modified. The adapter validates successful cases,
    unique case IDs/ranks, and every retrieved chunk against the frozen corpus.
    """
    source_path = Path(source_path)
    chunks_path = Path(chunks_path)
    output_path = Path(output_path)

    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("NuevaMente retrieval source must be a non-empty JSON array")

    chunks = pd.read_csv(chunks_path)
    if "chunk_id" not in chunks.columns:
        raise ValueError("chunks file missing required column: chunk_id")
    known_chunks = set(chunks["chunk_id"].astype(str))

    seen_cases: set[str] = set()
    rows: list[dict] = []
    score_types: set[str] = set()
    top_k_values: set[int] = set()

    for case in payload:
        case_id = str(case.get("case_id", "")).strip()
        if not case_id:
            raise ValueError("Retrieval case missing case_id")
        if case_id in seen_cases:
            raise ValueError(f"Duplicate retrieval case_id: {case_id}")
        seen_cases.add(case_id)

        status = str(case.get("status", "")).strip().lower()
        if status != "success":
            raise ValueError(f"Retrieval case {case_id} has non-success status: {status!r}")
        if case.get("error") not in (None, ""):
            raise ValueError(f"Retrieval case {case_id} contains error: {case.get('error')}")

        top_k = int(case.get("top_k", 0))
        if top_k <= 0:
            raise ValueError(f"Retrieval case {case_id} has invalid top_k={top_k}")
        top_k_values.add(top_k)

        score_type = str(case.get("score_type", "")).strip()
        score_types.add(score_type)

        results = case.get("results")
        if not isinstance(results, list):
            raise ValueError(f"Retrieval case {case_id} results must be a list")
        if len(results) != top_k:
            raise ValueError(
                f"Retrieval case {case_id} expected {top_k} results, found {len(results)}"
            )

        seen_ranks: set[int] = set()
        for result in results:
            rank = int(result.get("rank", 0))
            if rank <= 0 or rank in seen_ranks:
                raise ValueError(f"Invalid or duplicate rank {rank} for {case_id}")
            seen_ranks.add(rank)

            chunk_id = str(result.get("chunk_id", "")).strip()
            if chunk_id not in known_chunks:
                raise ValueError(f"Unknown retrieved chunk for {case_id}: {chunk_id}")

            metadata = result.get("metadata") or {}
            rows.append(
                {
                    "query_id": case_id,
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "score": result.get("score"),
                    "document_id": result.get("document_id"),
                    "category": metadata.get("categoria"),
                    "score_type": score_type,
                    "source_contract_version": case.get("contract_version"),
                }
            )

    out = pd.DataFrame(rows).sort_values(["query_id", "rank"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    return {
        "queries": len(seen_cases),
        "rows": len(out),
        "chunks": len(chunks),
        "top_k": ",".join(str(v) for v in sorted(top_k_values)),
        "score_types": ",".join(sorted(score_types)),
    }
