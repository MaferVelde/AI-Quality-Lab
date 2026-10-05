import json
from pathlib import Path

import pandas as pd
import pytest

from ai_quality_lab.experiments.lifecycle import (
    evaluate_experiment,
    validate_run_compatibility,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _manifest(gt_hash: str, ks=(3, 5)) -> dict:
    return {
        "evaluation": {"ks": list(ks)},
        "artifacts": {
            "ground_truth": {"sha256": gt_hash},
            "retrieval_results": {"sha256": "retrieval-hash"},
        },
    }


def _suite(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "experiment_id": "EXP-001",
                "strategy": "metadata_prefiltering",
                "affected_query_ids": "Q1|Q2",
                "priority": "high",
                "primary_metric": "HitRate@K",
                "baseline_run": "baseline_v2",
                "candidate_run": "",
                "status": "planned",
            }
        ]
    ).to_csv(path, index=False)


def _per_query(path: Path, q1: float, q2: float, other: float) -> None:
    pd.DataFrame(
        [
            {"query_id": "Q1", "hit_rate@5": q1},
            {"query_id": "Q2", "hit_rate@5": q2},
            {"query_id": "Q3", "hit_rate@5": other},
        ]
    ).to_csv(path, index=False)


def _summary(hit_rate: float, recall: float = 0.7, mrr: float = 0.6) -> dict:
    return {
        "n_queries": 3,
        "metrics": {
            "hit_rate@5": hit_rate,
            "recall@5": recall,
            "mrr": mrr,
        },
    }


def test_validate_run_compatibility_rejects_ground_truth_mismatch():
    with pytest.raises(ValueError, match="different ground-truth hashes"):
        validate_run_compatibility(
            _manifest("gt-v1"),
            _manifest("gt-v2"),
            k=5,
        )


def test_evaluate_experiment_accepts_improvement_without_global_regression(tmp_path: Path):
    suite = tmp_path / "suite.csv"
    _suite(suite)

    baseline_summary = tmp_path / "baseline_summary.json"
    candidate_summary = tmp_path / "candidate_summary.json"
    _write_json(baseline_summary, _summary(0.50, recall=0.60, mrr=0.55))
    _write_json(candidate_summary, _summary(0.70, recall=0.65, mrr=0.60))

    baseline_per_query = tmp_path / "baseline_per_query.csv"
    candidate_per_query = tmp_path / "candidate_per_query.csv"
    _per_query(baseline_per_query, 0.0, 0.0, 1.0)
    _per_query(candidate_per_query, 1.0, 1.0, 1.0)

    baseline_manifest = tmp_path / "baseline_manifest.json"
    candidate_manifest = tmp_path / "candidate_manifest.json"
    _write_json(baseline_manifest, _manifest("same-gt"))
    _write_json(candidate_manifest, _manifest("same-gt"))

    result, affected, updated = evaluate_experiment(
        experiment_suite_path=suite,
        experiment_id="EXP-001",
        baseline_summary_path=baseline_summary,
        candidate_summary_path=candidate_summary,
        baseline_per_query_path=baseline_per_query,
        candidate_per_query_path=candidate_per_query,
        baseline_manifest_path=baseline_manifest,
        candidate_manifest_path=candidate_manifest,
        candidate_run="candidate_001",
        k=5,
    )

    assert result["outcome"] == "accept"
    assert result["affected_queries"]["status"] == "improvement"
    assert result["global_comparison"]["has_regression"] is False
    assert len(affected) == 2
    assert updated.loc[0, "candidate_run"] == "candidate_001"
    assert updated.loc[0, "status"] == "evaluated"
    assert updated.loc[0, "outcome"] == "accept"


def test_evaluate_experiment_rejects_global_regression(tmp_path: Path):
    suite = tmp_path / "suite.csv"
    _suite(suite)

    baseline_summary = tmp_path / "baseline_summary.json"
    candidate_summary = tmp_path / "candidate_summary.json"
    _write_json(baseline_summary, _summary(0.50, recall=0.70, mrr=0.60))
    _write_json(candidate_summary, _summary(0.70, recall=0.60, mrr=0.65))

    baseline_per_query = tmp_path / "baseline_per_query.csv"
    candidate_per_query = tmp_path / "candidate_per_query.csv"
    _per_query(baseline_per_query, 0.0, 0.0, 1.0)
    _per_query(candidate_per_query, 1.0, 1.0, 1.0)

    baseline_manifest = tmp_path / "baseline_manifest.json"
    candidate_manifest = tmp_path / "candidate_manifest.json"
    _write_json(baseline_manifest, _manifest("same-gt"))
    _write_json(candidate_manifest, _manifest("same-gt"))

    result, _, _ = evaluate_experiment(
        experiment_suite_path=suite,
        experiment_id="EXP-001",
        baseline_summary_path=baseline_summary,
        candidate_summary_path=candidate_summary,
        baseline_per_query_path=baseline_per_query,
        candidate_per_query_path=candidate_per_query,
        baseline_manifest_path=baseline_manifest,
        candidate_manifest_path=candidate_manifest,
        candidate_run="candidate_002",
        k=5,
    )

    assert result["outcome"] == "reject"
    assert "recall@5" in result["global_comparison"]["regressions"]


def test_evaluate_experiment_is_inconclusive_when_affected_metric_unchanged(tmp_path: Path):
    suite = tmp_path / "suite.csv"
    _suite(suite)

    baseline_summary = tmp_path / "baseline_summary.json"
    candidate_summary = tmp_path / "candidate_summary.json"
    _write_json(baseline_summary, _summary(0.50, recall=0.70, mrr=0.60))
    _write_json(candidate_summary, _summary(0.50, recall=0.70, mrr=0.60))

    baseline_per_query = tmp_path / "baseline_per_query.csv"
    candidate_per_query = tmp_path / "candidate_per_query.csv"
    _per_query(baseline_per_query, 0.0, 1.0, 1.0)
    _per_query(candidate_per_query, 0.0, 1.0, 1.0)

    baseline_manifest = tmp_path / "baseline_manifest.json"
    candidate_manifest = tmp_path / "candidate_manifest.json"
    _write_json(baseline_manifest, _manifest("same-gt"))
    _write_json(candidate_manifest, _manifest("same-gt"))

    result, _, _ = evaluate_experiment(
        experiment_suite_path=suite,
        experiment_id="EXP-001",
        baseline_summary_path=baseline_summary,
        candidate_summary_path=candidate_summary,
        baseline_per_query_path=baseline_per_query,
        candidate_per_query_path=candidate_per_query,
        baseline_manifest_path=baseline_manifest,
        candidate_manifest_path=candidate_manifest,
        candidate_run="candidate_003",
        k=5,
    )

    assert result["outcome"] == "inconclusive"
