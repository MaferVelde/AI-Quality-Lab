from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from ai_quality_lab.analysis.root_causes import analyze_root_causes


_RECOMMENDATION_SPECS: dict[str, dict[str, str]] = {
    "cross_document_ambiguity": {
        "recommended_strategy": "metadata_prefiltering",
        "experiment_hypothesis": (
            "Adding source-document or category context before semantic retrieval should "
            "reduce cross-document competition and increase recovery of relevant evidence."
        ),
        "primary_metric": "HitRate@K",
        "metrics_to_watch": "HitRate@K|Recall@K|MRR|nDCG@K",
        "success_criteria": (
            "Affected queries recover relevant evidence more often while global retrieval "
            "metrics do not regress beyond the configured comparison tolerance."
        ),
        "regression_checks": (
            "Compare global metrics and per-category metrics; verify that filtering does "
            "not suppress relevant evidence from other valid documents."
        ),
        "implementation_notes": (
            "Test metadata filtering or query enrichment using known source document/category "
            "context before vector retrieval."
        ),
    },
    "intradocument_competition": {
        "recommended_strategy": "section_aware_reranking",
        "experiment_hypothesis": (
            "Adding section/title-aware reranking should promote the most specific evidence "
            "above neighboring chunks from the same document."
        ),
        "primary_metric": "MRR",
        "metrics_to_watch": "MRR|nDCG@K|HitRate@K|Recall@K",
        "success_criteria": (
            "Relevant chunks move closer to rank 1 for affected queries without reducing "
            "global HitRate@K or Recall@K beyond the configured tolerance."
        ),
        "regression_checks": (
            "Check global and per-category metrics; inspect whether reranking harms queries "
            "whose current rank 1 result is already relevant."
        ),
        "implementation_notes": (
            "Use section titles, headings, document position, or a lightweight reranker as "
            "additional ranking signals."
        ),
    },
    "technical_token_miss": {
        "recommended_strategy": "hybrid_lexical_vector_retrieval",
        "experiment_hypothesis": (
            "Combining lexical retrieval with vector similarity should improve recovery of "
            "exact technical tokens such as filenames, code symbols, endpoints, and identifiers."
        ),
        "primary_metric": "Recall@K",
        "metrics_to_watch": "Recall@K|HitRate@K|MRR|nDCG@K",
        "success_criteria": (
            "Affected technical queries recover the relevant evidence within Top-K while "
            "global retrieval metrics do not regress beyond the configured tolerance."
        ),
        "regression_checks": (
            "Measure global and per-category metrics; inspect false-positive lexical matches "
            "and ensure exact-token weighting does not dominate semantic relevance."
        ),
        "implementation_notes": (
            "Test BM25 or another lexical scorer combined with vector scores, then tune the "
            "fusion weight using the frozen benchmark."
        ),
    },
    "inferential_query": {
        "recommended_strategy": "query_rewriting_or_decomposition",
        "experiment_hypothesis": (
            "Rewriting inferential questions into explicit factual retrieval queries should "
            "increase retrieval of evidence needed to answer the original question."
        ),
        "primary_metric": "HitRate@K",
        "metrics_to_watch": "HitRate@K|Recall@K|MRR|nDCG@K",
        "success_criteria": (
            "The rewritten/decomposed query retrieves relevant evidence for affected cases "
            "without degrading the untouched baseline query set."
        ),
        "regression_checks": (
            "Keep the original query set frozen; compare rewritten-query gains separately and "
            "check for semantic drift introduced by rewriting."
        ),
        "implementation_notes": (
            "Generate one or more explicit factual subqueries, retain the original query for "
            "traceability, and evaluate both paths against the same frozen ground truth."
        ),
    },
    "semantic_mismatch": {
        "recommended_strategy": "hybrid_retrieval_reranking_probe",
        "experiment_hypothesis": (
            "A controlled hybrid-retrieval or reranking experiment may improve retrieval, but "
            "the current evidence is insufficient to select a single mechanism confidently."
        ),
        "primary_metric": "MRR",
        "metrics_to_watch": "MRR|Recall@K|HitRate@K|nDCG@K",
        "success_criteria": (
            "At least one controlled candidate improves affected-query ranking while global "
            "metrics remain within the configured non-regression tolerance."
        ),
        "regression_checks": (
            "Run one intervention at a time, compare against the same baseline, and avoid "
            "attributing improvement to a mechanism that was not isolated experimentally."
        ),
        "implementation_notes": (
            "Start with a small A/B probe: baseline vector retrieval vs hybrid retrieval, then "
            "test reranking separately if the first experiment is inconclusive."
        ),
    },
    "retrieval_execution_failure": {
        "recommended_strategy": "retrieval_pipeline_reliability_check",
        "experiment_hypothesis": (
            "Restoring reliable retriever execution and indexing availability should remove "
            "no-result failures before ranking-quality experiments are attempted."
        ),
        "primary_metric": "HitRate@K",
        "metrics_to_watch": "HitRate@K|Recall@K|retrieval_success_rate",
        "success_criteria": (
            "Previously empty queries return valid ranked results and no new execution failures "
            "are introduced."
        ),
        "regression_checks": (
            "Verify query coverage, index availability, routing, and result-schema consistency "
            "before comparing ranking metrics."
        ),
        "implementation_notes": (
            "Treat this as a reliability experiment, not a ranking-tuning task."
        ),
    },
    "below_cutoff_ranking": {
        "recommended_strategy": "reranking_or_cutoff_tuning",
        "experiment_hypothesis": (
            "Improving ranking quality or adjusting the retrieval cutoff should move known "
            "relevant evidence into the audited Top-K."
        ),
        "primary_metric": "MRR",
        "metrics_to_watch": "MRR|nDCG@K|Recall@K|HitRate@K",
        "success_criteria": (
            "Relevant evidence moves into Top-K or closer to rank 1 without creating a global "
            "metric regression beyond the configured tolerance."
        ),
        "regression_checks": (
            "Compare both ranking quality and retrieval depth/cost so gains are not achieved "
            "only by increasing K excessively."
        ),
        "implementation_notes": (
            "Test reranking first; evaluate K changes separately so ranking gains and depth "
            "tradeoffs remain distinguishable."
        ),
    },
}

_REQUIRED_COLUMNS = {
    "query_id",
    "category",
    "root_cause_candidate",
    "confidence",
}

_OUTPUT_COLUMNS = [
    "query_id",
    "category",
    "root_cause_candidate",
    "confidence",
    "priority",
    "recommended_strategy",
    "experiment_hypothesis",
    "primary_metric",
    "metrics_to_watch",
    "success_criteria",
    "regression_checks",
    "implementation_notes",
]


def _priority(confidence: str) -> str:
    """Prioritize specific, higher-confidence diagnoses before broader probes."""
    return "high" if str(confidence).lower() == "high" else "medium"


def recommend_experiments_from_analysis(
    root_cause_analysis: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Turn root-cause candidates into deterministic experiment plans."""
    missing = sorted(_REQUIRED_COLUMNS - set(root_cause_analysis.columns))
    if missing:
        raise ValueError(
            f"root_cause_analysis is missing required columns: {missing}"
        )

    unknown_causes = sorted(
        set(root_cause_analysis["root_cause_candidate"].astype(str))
        - set(_RECOMMENDATION_SPECS)
    )
    if unknown_causes:
        raise ValueError(
            f"unsupported root_cause_candidate values: {unknown_causes}"
        )

    rows: list[dict] = []
    for item in root_cause_analysis.to_dict(orient="records"):
        cause = str(item["root_cause_candidate"])
        spec = _RECOMMENDATION_SPECS[cause]
        rows.append(
            {
                "query_id": str(item["query_id"]),
                "category": item.get("category"),
                "root_cause_candidate": cause,
                "confidence": str(item["confidence"]),
                "priority": _priority(str(item["confidence"])),
                **spec,
            }
        )

    plans = pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)

    strategy_counts = (
        dict(sorted(Counter(plans["recommended_strategy"]).items()))
        if not plans.empty
        else {}
    )
    priority_counts = (
        dict(sorted(Counter(plans["priority"]).items()))
        if not plans.empty
        else {}
    )

    summary: dict = {
        "planned_experiments": int(len(plans)),
        "strategy_counts": strategy_counts,
        "priority_counts": priority_counts,
        "guardrail": (
            "Recommendations are experiment hypotheses, not guaranteed fixes. "
            "Validate every intervention against the same frozen benchmark and compare "
            "baseline versus candidate results before accepting it."
        ),
    }

    if not plans.empty and plans["category"].notna().any():
        by_category: dict[str, dict[str, int]] = {}
        for category, group in plans.dropna(subset=["category"]).groupby("category"):
            by_category[str(category)] = dict(
                sorted(Counter(group["recommended_strategy"]).items())
            )
        summary["by_category"] = by_category

    return plans, summary


def recommend_experiments(
    ground_truth_path: str | Path,
    retrieval_path: str | Path,
    k: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """Run root-cause analysis and create experiment recommendations."""
    root_cause_analysis, root_summary = analyze_root_causes(
        ground_truth_path=ground_truth_path,
        retrieval_path=retrieval_path,
        k=k,
    )
    plans, summary = recommend_experiments_from_analysis(root_cause_analysis)
    summary["n_queries_evaluated"] = int(root_summary["n_queries_evaluated"])
    summary["k"] = int(k)
    summary["hard_failures_input"] = int(root_summary["hard_failures_input"])
    summary["root_cause_counts"] = dict(root_summary.get("cause_counts", {}))
    return plans, summary
