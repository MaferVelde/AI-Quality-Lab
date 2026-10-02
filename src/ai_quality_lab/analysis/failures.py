from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from ai_quality_lab.evaluation.retrieval import load_ground_truth, load_retrieval


_RECOMMENDATIONS = {
    "no_results": "Verify retriever execution, indexing availability, and query routing before tuning ranking.",
    "top_k_miss": "Inspect chunking and embedding fit for this query; the expected evidence was not retrieved in the available result set.",
    "relevant_below_cutoff": "Review K and ranking quality; relevant evidence exists in the retrieved list but falls below the audited cutoff.",
    "partial_coverage": "Inspect whether one chunk can answer the query alone; otherwise improve retrieval coverage for the missing relevant evidence.",
    "weak_ranking": "Improve ranking or reranking so relevant evidence is promoted closer to rank 1.",
    "pass": "No deterministic retrieval issue detected at the audited cutoff.",
}

_HARD_FAILURES = {"no_results", "top_k_miss", "relevant_below_cutoff"}


def _pipe(values: list[str] | set[str]) -> str:
    return "|".join(sorted(str(value) for value in values))


def analyze_failures(
    ground_truth_path: str | Path,
    retrieval_path: str | Path,
    k: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """Classify deterministic retrieval issues for every ground-truth query.

    This layer intentionally avoids speculative root-cause claims. It describes
    observable retrieval behavior only. Semantic/root-cause analysis can be
    layered on top later with corpus text and additional evidence.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")

    ground_truth = load_ground_truth(ground_truth_path)
    retrieval = load_retrieval(retrieval_path)

    unknown_qids = sorted(set(retrieval["query_id"]) - set(ground_truth))
    if unknown_qids:
        raise ValueError(
            f"retrieval_results contains unknown query_id values: {unknown_qids[:10]}"
        )

    grouped = {qid: group for qid, group in retrieval.groupby("query_id", sort=False)}
    rows: list[dict] = []

    for qid, item in ground_truth.items():
        group = grouped.get(qid)
        if group is None:
            retrieved_ids: list[str] = []
            rank_by_chunk: dict[str, int] = {}
        else:
            retrieved_ids = group["chunk_id"].astype(str).tolist()
            rank_by_chunk = {
                str(row.chunk_id): int(row.rank)
                for row in group[["chunk_id", "rank"]].itertuples(index=False)
            }

        relevant = set(item.relevant_chunk_ids)
        top_ids = retrieved_ids[:k]
        hit_ids = set(top_ids) & relevant
        all_retrieved_relevant = set(retrieved_ids) & relevant
        missed_relevant = relevant - hit_ids
        relevant_ranks = sorted(
            rank_by_chunk[chunk_id]
            for chunk_id in all_retrieved_relevant
            if chunk_id in rank_by_chunk
        )
        first_relevant_rank = relevant_ranks[0] if relevant_ranks else None

        secondary_issues: list[str] = []
        if not retrieved_ids:
            primary_issue = "no_results"
            severity = "critical"
        elif not hit_ids:
            if first_relevant_rank is not None and first_relevant_rank > k:
                primary_issue = "relevant_below_cutoff"
            else:
                primary_issue = "top_k_miss"
            severity = "high"
        elif len(hit_ids) < len(relevant):
            primary_issue = "partial_coverage"
            severity = "medium"
            if first_relevant_rank and first_relevant_rank > 1:
                secondary_issues.append("weak_ranking")
        elif first_relevant_rank and first_relevant_rank > 1:
            primary_issue = "weak_ranking"
            severity = "medium"
        else:
            primary_issue = "pass"
            severity = "none"

        rows.append(
            {
                "query_id": qid,
                "question": item.question,
                "category": item.category,
                "audit_k": k,
                "primary_issue": primary_issue,
                "severity": severity,
                "hard_failure": primary_issue in _HARD_FAILURES,
                "secondary_issues": "|".join(secondary_issues),
                "relevant_count": len(relevant),
                "retrieval_depth": len(retrieved_ids),
                "hits_at_k": len(hit_ids),
                "coverage_at_k": len(hit_ids) / len(relevant) if relevant else 0.0,
                "first_relevant_rank": first_relevant_rank,
                "relevant_chunk_ids": _pipe(relevant),
                "retrieved_top_k": "|".join(top_ids),
                "missed_relevant_ids": _pipe(missed_relevant),
                "recommendation": _RECOMMENDATIONS[primary_issue],
            }
        )

    analysis = pd.DataFrame(rows)
    issue_counts = Counter(analysis["primary_issue"].astype(str))
    hard_failures = int(analysis["hard_failure"].sum())
    passes = int((analysis["primary_issue"] == "pass").sum())
    warnings = int(len(analysis) - hard_failures - passes)

    summary: dict = {
        "n_queries": int(len(analysis)),
        "k": int(k),
        "hard_failures": hard_failures,
        "hard_failure_rate": hard_failures / len(analysis) if len(analysis) else 0.0,
        "warnings": warnings,
        "passes": passes,
        "issue_counts": dict(sorted(issue_counts.items())),
        "hard_failure_query_ids": analysis.loc[
            analysis["hard_failure"], "query_id"
        ].astype(str).tolist(),
    }

    if analysis["category"].notna().any():
        by_category: dict[str, dict] = {}
        for category, group in analysis.dropna(subset=["category"]).groupby("category"):
            cat_hard = int(group["hard_failure"].sum())
            by_category[str(category)] = {
                "n_queries": int(len(group)),
                "hard_failures": cat_hard,
                "hard_failure_rate": cat_hard / len(group) if len(group) else 0.0,
                "issue_counts": dict(
                    sorted(Counter(group["primary_issue"].astype(str)).items())
                ),
            }
        summary["by_category"] = by_category

    return analysis, summary
