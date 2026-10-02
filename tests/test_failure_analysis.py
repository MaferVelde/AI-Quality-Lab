from pathlib import Path

import pandas as pd

from ai_quality_lab.analysis.failures import analyze_failures


def test_analyze_failures_taxonomy(tmp_path: Path):
    ground_truth = pd.DataFrame(
        [
            {"query_id": "Q1", "question": "pass", "relevant_chunk_ids": "A", "category": "cat"},
            {"query_id": "Q2", "question": "weak rank", "relevant_chunk_ids": "B", "category": "cat"},
            {"query_id": "Q3", "question": "partial", "relevant_chunk_ids": "C|D", "category": "cat"},
            {"query_id": "Q4", "question": "top k miss", "relevant_chunk_ids": "E", "category": "cat"},
            {"query_id": "Q5", "question": "below cutoff", "relevant_chunk_ids": "F", "category": "cat"},
            {"query_id": "Q6", "question": "no results", "relevant_chunk_ids": "G", "category": "cat"},
        ]
    )
    retrieval = pd.DataFrame(
        [
            {"query_id": "Q1", "rank": 1, "chunk_id": "A"},
            {"query_id": "Q1", "rank": 2, "chunk_id": "X1"},
            {"query_id": "Q2", "rank": 1, "chunk_id": "X2"},
            {"query_id": "Q2", "rank": 2, "chunk_id": "B"},
            {"query_id": "Q3", "rank": 1, "chunk_id": "C"},
            {"query_id": "Q3", "rank": 2, "chunk_id": "X3"},
            {"query_id": "Q4", "rank": 1, "chunk_id": "X4"},
            {"query_id": "Q4", "rank": 2, "chunk_id": "X5"},
            {"query_id": "Q5", "rank": 1, "chunk_id": "X6"},
            {"query_id": "Q5", "rank": 2, "chunk_id": "X7"},
            {"query_id": "Q5", "rank": 3, "chunk_id": "F"},
        ]
    )

    gt_path = tmp_path / "ground_truth.csv"
    retrieval_path = tmp_path / "retrieval.csv"
    ground_truth.to_csv(gt_path, index=False)
    retrieval.to_csv(retrieval_path, index=False)

    analysis, summary = analyze_failures(gt_path, retrieval_path, k=2)
    issues = dict(zip(analysis["query_id"], analysis["primary_issue"]))

    assert issues == {
        "Q1": "pass",
        "Q2": "weak_ranking",
        "Q3": "partial_coverage",
        "Q4": "top_k_miss",
        "Q5": "relevant_below_cutoff",
        "Q6": "no_results",
    }
    assert summary["n_queries"] == 6
    assert summary["hard_failures"] == 3
    assert summary["warnings"] == 2
    assert summary["passes"] == 1
    assert summary["hard_failure_query_ids"] == ["Q4", "Q5", "Q6"]
