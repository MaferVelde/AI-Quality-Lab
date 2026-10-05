from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from ai_quality_lab.analysis.recommendations import recommend_experiments


_REQUIRED_COLUMNS = {
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
}

_OUTPUT_COLUMNS = [
    "experiment_id",
    "strategy",
    "root_cause_candidates",
    "affected_query_ids",
    "affected_categories",
    "n_affected_queries",
    "priority",
    "primary_metric",
    "metrics_to_watch",
    "experiment_hypothesis",
    "success_criteria",
    "regression_checks",
    "implementation_notes",
    "baseline_run",
    "candidate_run",
    "status",
]

_PRIORITY_RANK = {"high": 0, "medium": 1, "low": 2}


def _join_unique(values: pd.Series) -> str:
    cleaned = {
        str(value).strip()
        for value in values
        if value is not None and not pd.isna(value) and str(value).strip()
    }
    return "|".join(sorted(cleaned))


def _group_priority(values: pd.Series) -> str:
    priorities = [
        str(value).lower()
        for value in values
        if value is not None and not pd.isna(value)
    ]
    if not priorities:
        return "medium"
    return min(priorities, key=lambda value: _PRIORITY_RANK.get(value, 99))


def build_experiment_suite_from_plans(
    recommendation_plans: pd.DataFrame,
    baseline_run: str = "baseline",
) -> tuple[pd.DataFrame, dict]:
    """Group per-query recommendations into reproducible experiment definitions."""
    missing = sorted(_REQUIRED_COLUMNS - set(recommendation_plans.columns))
    if missing:
        raise ValueError(
            f"recommendation_plans is missing required columns: {missing}"
        )

    if recommendation_plans.empty:
        suite = pd.DataFrame(columns=_OUTPUT_COLUMNS)
        return suite, {
            "recommendation_plans_input": 0,
            "planned_experiments": 0,
            "affected_queries": 0,
            "priority_counts": {},
            "strategy_counts": {},
            "guardrail": (
                "Experiment definitions are plans only. A strategy is not accepted as an "
                "improvement until a candidate run is evaluated against the frozen baseline."
            ),
        }

    grouped_rows: list[dict] = []
    for strategy, group in recommendation_plans.groupby(
        "recommended_strategy", sort=True
    ):
        first = group.iloc[0]
        grouped_rows.append(
            {
                "strategy": str(strategy),
                "root_cause_candidates": _join_unique(
                    group["root_cause_candidate"]
                ),
                "affected_query_ids": _join_unique(group["query_id"]),
                "affected_categories": _join_unique(group["category"]),
                "n_affected_queries": int(group["query_id"].astype(str).nunique()),
                "priority": _group_priority(group["priority"]),
                "primary_metric": str(first["primary_metric"]),
                "metrics_to_watch": str(first["metrics_to_watch"]),
                "experiment_hypothesis": str(first["experiment_hypothesis"]),
                "success_criteria": str(first["success_criteria"]),
                "regression_checks": str(first["regression_checks"]),
                "implementation_notes": str(first["implementation_notes"]),
                "baseline_run": str(baseline_run),
                "candidate_run": "",
                "status": "planned",
            }
        )

    suite = pd.DataFrame(grouped_rows)
    suite["_priority_rank"] = suite["priority"].map(
        lambda value: _PRIORITY_RANK.get(str(value), 99)
    )
    suite = suite.sort_values(
        by=["_priority_rank", "n_affected_queries", "strategy"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    suite.insert(
        0,
        "experiment_id",
        [f"EXP-{index:03d}" for index in range(1, len(suite) + 1)],
    )
    suite = suite.drop(columns=["_priority_rank"])
    suite = suite[_OUTPUT_COLUMNS]

    priority_counts = dict(
        sorted(Counter(suite["priority"].astype(str)).items())
    )
    strategy_counts = dict(
        sorted(Counter(suite["strategy"].astype(str)).items())
    )

    summary = {
        "recommendation_plans_input": int(len(recommendation_plans)),
        "planned_experiments": int(len(suite)),
        "affected_queries": int(
            recommendation_plans["query_id"].astype(str).nunique()
        ),
        "priority_counts": priority_counts,
        "strategy_counts": strategy_counts,
        "baseline_run": str(baseline_run),
        "guardrail": (
            "Experiment definitions are plans only. A strategy is not accepted as an "
            "improvement until a candidate run is evaluated against the frozen baseline."
        ),
    }
    return suite, summary


def build_experiment_suite(
    ground_truth_path: str | Path,
    retrieval_path: str | Path,
    k: int = 5,
    baseline_run: str = "baseline",
) -> tuple[pd.DataFrame, dict]:
    """Run recommendation planning and group recommendations into experiments."""
    plans, recommendation_summary = recommend_experiments(
        ground_truth_path=ground_truth_path,
        retrieval_path=retrieval_path,
        k=k,
    )
    suite, summary = build_experiment_suite_from_plans(
        recommendation_plans=plans,
        baseline_run=baseline_run,
    )
    summary["n_queries_evaluated"] = int(
        recommendation_summary["n_queries_evaluated"]
    )
    summary["k"] = int(k)
    summary["hard_failures_input"] = int(
        recommendation_summary["hard_failures_input"]
    )
    summary["root_cause_counts"] = dict(
        recommendation_summary.get("root_cause_counts", {})
    )
    return suite, summary
