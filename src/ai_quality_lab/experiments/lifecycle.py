from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ai_quality_lab.comparison import compare_summaries


_REQUIRED_SUITE_COLUMNS = {
    "experiment_id",
    "strategy",
    "affected_query_ids",
    "priority",
    "primary_metric",
    "baseline_run",
    "candidate_run",
    "status",
}

_METRIC_ALIASES = {
    "MRR": "mrr",
    "HitRate@K": "hit_rate@{k}",
    "Recall@K": "recall@{k}",
    "Precision@K": "precision@{k}",
    "nDCG@K": "ndcg@{k}",
}


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metric_key(primary_metric: str, k: int) -> str:
    template = _METRIC_ALIASES.get(str(primary_metric))
    if template is None:
        raise ValueError(f"unsupported primary_metric: {primary_metric!r}")
    return template.format(k=k)


def _ground_truth_hash(manifest: dict) -> str | None:
    return (
        manifest.get("artifacts", {})
        .get("ground_truth", {})
        .get("sha256")
    )


def _manifest_ks(manifest: dict) -> list[int]:
    return [
        int(value)
        for value in manifest.get("evaluation", {}).get("ks", [])
    ]


def validate_run_compatibility(
    baseline_manifest: dict,
    candidate_manifest: dict,
    k: int,
) -> dict:
    """Validate that baseline and candidate are comparable evaluation runs."""
    baseline_gt = _ground_truth_hash(baseline_manifest)
    candidate_gt = _ground_truth_hash(candidate_manifest)

    if not baseline_gt or not candidate_gt:
        raise ValueError("both manifests must include a ground-truth sha256")

    if baseline_gt != candidate_gt:
        raise ValueError(
            "baseline and candidate use different ground-truth hashes; "
            "cross-version comparison is not valid"
        )

    baseline_ks = _manifest_ks(baseline_manifest)
    candidate_ks = _manifest_ks(candidate_manifest)
    if k not in baseline_ks:
        raise ValueError(f"baseline manifest does not include k={k}")
    if k not in candidate_ks:
        raise ValueError(f"candidate manifest does not include k={k}")

    return {
        "compatible": True,
        "ground_truth_sha256": baseline_gt,
        "baseline_ks": baseline_ks,
        "candidate_ks": candidate_ks,
        "audited_k": int(k),
    }


def _select_experiment(
    suite: pd.DataFrame,
    experiment_id: str,
) -> pd.Series:
    missing = sorted(_REQUIRED_SUITE_COLUMNS - set(suite.columns))
    if missing:
        raise ValueError(f"experiment_suite is missing required columns: {missing}")

    matched = suite.loc[suite["experiment_id"].astype(str) == str(experiment_id)]
    if matched.empty:
        raise ValueError(f"experiment_id not found: {experiment_id}")
    if len(matched) > 1:
        raise ValueError(f"duplicate experiment_id in suite: {experiment_id}")
    return matched.iloc[0]


def _affected_query_comparison(
    baseline_per_query: pd.DataFrame,
    candidate_per_query: pd.DataFrame,
    affected_query_ids: list[str],
    metric_key: str,
    tolerance: float,
) -> tuple[pd.DataFrame, dict]:
    required = {"query_id", metric_key}
    for name, frame in [
        ("baseline_per_query", baseline_per_query),
        ("candidate_per_query", candidate_per_query),
    ]:
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

    baseline = baseline_per_query.copy()
    candidate = candidate_per_query.copy()
    baseline["query_id"] = baseline["query_id"].astype(str)
    candidate["query_id"] = candidate["query_id"].astype(str)

    baseline = baseline.loc[
        baseline["query_id"].isin(affected_query_ids),
        ["query_id", metric_key],
    ].rename(columns={metric_key: "baseline"})
    candidate = candidate.loc[
        candidate["query_id"].isin(affected_query_ids),
        ["query_id", metric_key],
    ].rename(columns={metric_key: "candidate"})

    comparison = baseline.merge(
        candidate,
        on="query_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    missing_rows = comparison.loc[comparison["_merge"] != "both", "query_id"].tolist()
    if missing_rows:
        raise ValueError(
            "affected queries are missing from baseline/candidate per-query metrics: "
            f"{sorted(missing_rows)}"
        )

    expected = set(affected_query_ids)
    observed = set(comparison["query_id"].astype(str))
    if observed != expected:
        missing = sorted(expected - observed)
        if missing:
            raise ValueError(
                f"affected queries are missing from per-query metrics: {missing}"
            )

    comparison = comparison.drop(columns=["_merge"])
    comparison["baseline"] = pd.to_numeric(comparison["baseline"], errors="raise")
    comparison["candidate"] = pd.to_numeric(comparison["candidate"], errors="raise")
    comparison["delta"] = comparison["candidate"] - comparison["baseline"]

    def status(delta: float) -> str:
        if delta > tolerance:
            return "improvement"
        if delta < -tolerance:
            return "regression"
        return "unchanged"

    comparison["status"] = comparison["delta"].map(status)
    comparison = comparison.sort_values("query_id").reset_index(drop=True)

    baseline_mean = float(comparison["baseline"].mean())
    candidate_mean = float(comparison["candidate"].mean())
    mean_delta = candidate_mean - baseline_mean

    if mean_delta > tolerance:
        affected_status = "improvement"
    elif mean_delta < -tolerance:
        affected_status = "regression"
    else:
        affected_status = "unchanged"

    summary = {
        "metric": metric_key,
        "n_affected_queries": int(len(comparison)),
        "baseline_mean": baseline_mean,
        "candidate_mean": candidate_mean,
        "delta": mean_delta,
        "status": affected_status,
        "improved_queries": int((comparison["status"] == "improvement").sum()),
        "regressed_queries": int((comparison["status"] == "regression").sum()),
        "unchanged_queries": int((comparison["status"] == "unchanged").sum()),
    }
    return comparison, summary


def evaluate_experiment(
    *,
    experiment_suite_path: str | Path,
    experiment_id: str,
    baseline_summary_path: str | Path,
    candidate_summary_path: str | Path,
    baseline_per_query_path: str | Path,
    candidate_per_query_path: str | Path,
    baseline_manifest_path: str | Path,
    candidate_manifest_path: str | Path,
    candidate_run: str,
    k: int,
    tolerance: float = 1e-12,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Evaluate one candidate experiment against a frozen compatible baseline.

    Returns:
        result: structured decision record
        affected_comparison: per-query comparison for affected queries
        updated_suite: copy of the suite with the evaluated experiment updated
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")
    if not str(candidate_run).strip():
        raise ValueError("candidate_run must be non-empty")

    suite = pd.read_csv(experiment_suite_path)
    experiment = _select_experiment(suite, experiment_id)

    baseline_summary = _load_json(baseline_summary_path)
    candidate_summary = _load_json(candidate_summary_path)
    baseline_manifest = _load_json(baseline_manifest_path)
    candidate_manifest = _load_json(candidate_manifest_path)

    compatibility = validate_run_compatibility(
        baseline_manifest=baseline_manifest,
        candidate_manifest=candidate_manifest,
        k=k,
    )

    if baseline_summary.get("n_queries") != candidate_summary.get("n_queries"):
        raise ValueError(
            "baseline and candidate summaries contain different n_queries values"
        )

    metric_key = _metric_key(str(experiment["primary_metric"]), k)
    for name, summary in [
        ("baseline", baseline_summary),
        ("candidate", candidate_summary),
    ]:
        if metric_key not in summary.get("metrics", {}):
            raise ValueError(
                f"{name} summary does not include required metric {metric_key!r}"
            )

    global_comparison = compare_summaries(
        baseline_summary,
        candidate_summary,
        tolerance=tolerance,
    )

    affected_query_ids = [
        value
        for value in str(experiment["affected_query_ids"]).split("|")
        if value
    ]
    baseline_per_query = pd.read_csv(baseline_per_query_path)
    candidate_per_query = pd.read_csv(candidate_per_query_path)
    affected_comparison, affected_summary = _affected_query_comparison(
        baseline_per_query=baseline_per_query,
        candidate_per_query=candidate_per_query,
        affected_query_ids=affected_query_ids,
        metric_key=metric_key,
        tolerance=tolerance,
    )

    global_primary = global_comparison["metrics"].get(metric_key)
    if global_primary is None:
        raise ValueError(
            f"global comparison does not include required metric {metric_key!r}"
        )

    has_global_regression = bool(global_comparison["has_regression"])
    affected_status = affected_summary["status"]

    if has_global_regression or affected_status == "regression":
        outcome = "reject"
    elif affected_status == "improvement":
        outcome = "accept"
    else:
        outcome = "inconclusive"

    reasons: list[str] = []
    if has_global_regression:
        reasons.append(
            "one or more global retrieval metrics regressed beyond tolerance"
        )
    if affected_status == "regression":
        reasons.append("affected-query primary metric regressed")
    elif affected_status == "improvement":
        reasons.append("affected-query primary metric improved")
    else:
        reasons.append("affected-query primary metric did not change beyond tolerance")
    if not has_global_regression:
        reasons.append("no global metric regression detected")

    result = {
        "experiment_id": str(experiment_id),
        "strategy": str(experiment["strategy"]),
        "priority": str(experiment["priority"]),
        "baseline_run": str(experiment["baseline_run"]),
        "candidate_run": str(candidate_run),
        "status": "evaluated",
        "outcome": outcome,
        "audited_k": int(k),
        "primary_metric": metric_key,
        "tolerance": float(tolerance),
        "compatibility": compatibility,
        "affected_queries": affected_summary,
        "global_primary_metric": global_primary,
        "global_comparison": global_comparison,
        "decision_reasons": reasons,
        "guardrail": (
            "The outcome reflects this frozen baseline/candidate comparison only. "
            "It does not prove that the proposed mechanism is the true causal explanation."
        ),
    }

    updated_suite = suite.copy()
    mask = updated_suite["experiment_id"].astype(str) == str(experiment_id)
    updated_suite.loc[mask, "candidate_run"] = str(candidate_run)
    updated_suite.loc[mask, "status"] = "evaluated"
    if "outcome" not in updated_suite.columns:
        updated_suite["outcome"] = ""
    updated_suite.loc[mask, "outcome"] = outcome

    return result, affected_comparison, updated_suite
