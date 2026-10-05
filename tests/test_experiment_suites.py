from pathlib import Path

import pandas as pd

from ai_quality_lab.analysis.experiment_suites import (
    build_experiment_suite,
    build_experiment_suite_from_plans,
)


def test_build_experiment_suite_from_plans_groups_and_orders_deterministically():
    plans = pd.DataFrame(
        [
            {
                "query_id": "Q1",
                "category": "A",
                "root_cause_candidate": "cross_document_ambiguity",
                "confidence": "high",
                "priority": "high",
                "recommended_strategy": "metadata_prefiltering",
                "experiment_hypothesis": "h1",
                "primary_metric": "HitRate@K",
                "metrics_to_watch": "HitRate@K|Recall@K",
                "success_criteria": "s1",
                "regression_checks": "r1",
                "implementation_notes": "i1",
            },
            {
                "query_id": "Q2",
                "category": "B",
                "root_cause_candidate": "cross_document_ambiguity",
                "confidence": "high",
                "priority": "high",
                "recommended_strategy": "metadata_prefiltering",
                "experiment_hypothesis": "h1",
                "primary_metric": "HitRate@K",
                "metrics_to_watch": "HitRate@K|Recall@K",
                "success_criteria": "s1",
                "regression_checks": "r1",
                "implementation_notes": "i1",
            },
            {
                "query_id": "Q3",
                "category": "C",
                "root_cause_candidate": "intradocument_competition",
                "confidence": "high",
                "priority": "high",
                "recommended_strategy": "section_aware_reranking",
                "experiment_hypothesis": "h2",
                "primary_metric": "MRR",
                "metrics_to_watch": "MRR|nDCG@K",
                "success_criteria": "s2",
                "regression_checks": "r2",
                "implementation_notes": "i2",
            },
            {
                "query_id": "Q4",
                "category": "D",
                "root_cause_candidate": "semantic_mismatch",
                "confidence": "medium",
                "priority": "medium",
                "recommended_strategy": "hybrid_retrieval_reranking_probe",
                "experiment_hypothesis": "h3",
                "primary_metric": "MRR",
                "metrics_to_watch": "MRR|Recall@K",
                "success_criteria": "s3",
                "regression_checks": "r3",
                "implementation_notes": "i3",
            },
        ]
    )

    suite, summary = build_experiment_suite_from_plans(
        plans, baseline_run="audit_001"
    )

    assert list(suite["experiment_id"]) == ["EXP-001", "EXP-002", "EXP-003"]
    assert list(suite["strategy"]) == [
        "metadata_prefiltering",
        "section_aware_reranking",
        "hybrid_retrieval_reranking_probe",
    ]
    assert suite.iloc[0]["affected_query_ids"] == "Q1|Q2"
    assert suite.iloc[0]["affected_categories"] == "A|B"
    assert suite.iloc[0]["n_affected_queries"] == 2
    assert suite.iloc[0]["baseline_run"] == "audit_001"
    assert suite.iloc[0]["status"] == "planned"

    assert summary["recommendation_plans_input"] == 4
    assert summary["planned_experiments"] == 3
    assert summary["affected_queries"] == 4
    assert summary["priority_counts"] == {"high": 2, "medium": 1}


def test_build_experiment_suite_runs_full_pipeline(tmp_path: Path):
    ground_truth = pd.DataFrame(
        [
            {
                "query_id": "Q1",
                "question": "¿Qué conocimientos previos se piden?",
                "relevant_chunk_ids": "DOC-A_CH_001",
                "category": "A",
                "source_document_id": "DOC-A",
                "expected_answer": "Conocimientos básicos.",
                "evidence_keywords": "conceptos básicos",
            },
            {
                "query_id": "Q2",
                "question": "¿Qué requisito previo se indica?",
                "relevant_chunk_ids": "DOC-B_CH_001",
                "category": "B",
                "source_document_id": "DOC-B",
                "expected_answer": "Conocimientos previos.",
                "evidence_keywords": "requisito previo",
            },
        ]
    )
    retrieval = pd.DataFrame(
        [
            {"query_id": "Q1", "rank": 1, "chunk_id": "DOC-X_CH_001", "document_id": "DOC-X"},
            {"query_id": "Q1", "rank": 2, "chunk_id": "DOC-Y_CH_001", "document_id": "DOC-Y"},
            {"query_id": "Q1", "rank": 3, "chunk_id": "DOC-Z_CH_001", "document_id": "DOC-Z"},
            {"query_id": "Q2", "rank": 1, "chunk_id": "DOC-X_CH_010", "document_id": "DOC-X"},
            {"query_id": "Q2", "rank": 2, "chunk_id": "DOC-Y_CH_010", "document_id": "DOC-Y"},
            {"query_id": "Q2", "rank": 3, "chunk_id": "DOC-Z_CH_010", "document_id": "DOC-Z"},
        ]
    )

    gt_path = tmp_path / "ground_truth.csv"
    retrieval_path = tmp_path / "retrieval.csv"
    ground_truth.to_csv(gt_path, index=False)
    retrieval.to_csv(retrieval_path, index=False)

    suite, summary = build_experiment_suite(
        gt_path,
        retrieval_path,
        k=3,
        baseline_run="baseline_test",
    )

    assert len(suite) == 1
    assert suite.iloc[0]["experiment_id"] == "EXP-001"
    assert suite.iloc[0]["strategy"] == "metadata_prefiltering"
    assert suite.iloc[0]["affected_query_ids"] == "Q1|Q2"
    assert suite.iloc[0]["n_affected_queries"] == 2
    assert summary["planned_experiments"] == 1
    assert summary["affected_queries"] == 2
    assert summary["baseline_run"] == "baseline_test"
