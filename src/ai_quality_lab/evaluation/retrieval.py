from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ai_quality_lab.evaluation.metrics import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from ai_quality_lab.models import QueryGroundTruth, parse_chunk_ids


def load_ground_truth(path: str | Path) -> dict[str, QueryGroundTruth]:
    df = pd.read_csv(path)
    required = {"query_id", "question", "relevant_chunk_ids"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ground_truth missing columns: {sorted(missing)}")

    if df["query_id"].duplicated().any():
        duplicates = sorted(df.loc[df["query_id"].duplicated(), "query_id"].astype(str).unique())
        raise ValueError(f"Duplicate query_id in ground truth: {duplicates}")

    result: dict[str, QueryGroundTruth] = {}
    for row in df.to_dict(orient="records"):
        qid = str(row["query_id"])
        relevant = parse_chunk_ids(row["relevant_chunk_ids"])
        if not relevant:
            raise ValueError(f"query_id={qid!r} has no relevant chunks")
        result[qid] = QueryGroundTruth(
            query_id=qid,
            question=str(row["question"]),
            relevant_chunk_ids=relevant,
            category=None if pd.isna(row.get("category")) else str(row.get("category")),
        )
    return result


def load_retrieval(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"query_id", "rank", "chunk_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"retrieval_results missing columns: {sorted(missing)}")

    df = df.copy()
    df["query_id"] = df["query_id"].astype(str)
    df["chunk_id"] = df["chunk_id"].astype(str)
    df["rank"] = pd.to_numeric(df["rank"], errors="raise").astype(int)
    if (df["rank"] <= 0).any():
        raise ValueError("rank values must be positive integers")
    if df.duplicated(subset=["query_id", "rank"]).any():
        raise ValueError("retrieval_results has duplicate (query_id, rank) rows")
    return df.sort_values(["query_id", "rank"]).reset_index(drop=True)


def evaluate_retrieval(
    ground_truth_path: str | Path,
    retrieval_path: str | Path,
    ks: tuple[int, ...] = (3, 5),
) -> tuple[pd.DataFrame, dict]:
    gt = load_ground_truth(ground_truth_path)
    retrieval = load_retrieval(retrieval_path)

    unknown_qids = sorted(set(retrieval["query_id"]) - set(gt))
    if unknown_qids:
        raise ValueError(f"retrieval_results contains unknown query_id values: {unknown_qids[:10]}")

    grouped = {qid: group for qid, group in retrieval.groupby("query_id", sort=False)}
    rows: list[dict] = []

    for qid, item in gt.items():
        group = grouped.get(qid)
        retrieved_ids = [] if group is None else group["chunk_id"].tolist()
        row = {
            "query_id": qid,
            "question": item.question,
            "category": item.category,
            "relevant_count": len(item.relevant_chunk_ids),
            "retrieved_count": len(retrieved_ids),
            "mrr": reciprocal_rank(item.relevant_chunk_ids, retrieved_ids),
        }
        for k in ks:
            row[f"recall@{k}"] = recall_at_k(item.relevant_chunk_ids, retrieved_ids, k)
            row[f"precision@{k}"] = precision_at_k(item.relevant_chunk_ids, retrieved_ids, k)
            row[f"hit_rate@{k}"] = hit_rate_at_k(item.relevant_chunk_ids, retrieved_ids, k)
            row[f"ndcg@{k}"] = ndcg_at_k(item.relevant_chunk_ids, retrieved_ids, k)

        if group is not None and "latency_ms" in group.columns:
            vals = pd.to_numeric(group["latency_ms"], errors="coerce").dropna()
            row["latency_ms"] = float(vals.iloc[0]) if not vals.empty else np.nan

        rows.append(row)

    per_query = pd.DataFrame(rows)
    metric_cols = [
        col for col in per_query.columns
        if col == "mrr" or col.startswith(("recall@", "precision@", "hit_rate@", "ndcg@"))
    ]
    summary = {
        "n_queries": int(len(per_query)),
        "ks": list(ks),
        "metrics": {col: float(per_query[col].mean()) for col in metric_cols},
    }
    if "latency_ms" in per_query.columns:
        latency = per_query["latency_ms"].dropna()
        if not latency.empty:
            summary["latency_ms"] = {
                "mean": float(latency.mean()),
                "p50": float(latency.quantile(0.50)),
                "p95": float(latency.quantile(0.95)),
            }

    if "category" in per_query.columns and per_query["category"].notna().any():
        category_summary: dict[str, dict[str, float]] = {}
        for category, group in per_query.dropna(subset=["category"]).groupby("category"):
            category_summary[str(category)] = {
                col: float(group[col].mean()) for col in metric_cols
            }
        summary["by_category"] = category_summary

    return per_query, summary
