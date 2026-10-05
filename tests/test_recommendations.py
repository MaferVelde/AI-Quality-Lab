from pathlib import Path

import pandas as pd

from ai_quality_lab.analysis.recommendations import (
    recommend_experiments,
    recommend_experiments_from_analysis,
)


def test_recommend_experiments_from_analysis_maps_all_supported_causes():
    analysis = pd.DataFrame(
        [
            {"query_id": "Q1", "category": "A", "root_cause_candidate": "cross_document_ambiguity", "confidence": "high"},
            {"query_id": "Q2", "category": "A", "root_cause_candidate": "intradocument_competition", "confidence": "high"},
            {"query_id": "Q3", "category": "B", "root_cause_candidate": "technical_token_miss", "confidence": "medium"},
            {"query_id": "Q4", "category": "B", "root_cause_candidate": "inferential_query", "confidence": "medium"},
            {"query_id": "Q5", "category": "C", "root_cause_candidate": "semantic_mismatch", "confidence": "medium"},
            {"query_id": "Q6", "category": "C", "root_cause_candidate": "retrieval_execution_failure", "confidence": "high"},
            {"query_id": "Q7", "category": "C", "root_cause_candidate": "below_cutoff_ranking", "confidence": "high"},
        ]
    )

    plans, summary = recommend_experiments_from_analysis(analysis)
    strategies = dict(zip(plans["query_id"], plans["recommended_strategy"]))

    assert strategies["Q1"] == "metadata_prefiltering"
    assert strategies["Q2"] == "section_aware_reranking"
    assert strategies["Q3"] == "hybrid_lexical_vector_retrieval"
    assert strategies["Q4"] == "query_rewriting_or_decomposition"
    assert strategies["Q5"] == "hybrid_retrieval_reranking_probe"
    assert strategies["Q6"] == "retrieval_pipeline_reliability_check"
    assert strategies["Q7"] == "reranking_or_cutoff_tuning"

    assert summary["planned_experiments"] == 7
    assert summary["priority_counts"] == {"high": 4, "medium": 3}


def test_recommend_experiments_runs_root_cause_pipeline(tmp_path: Path):
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
            }
        ]
    )
    retrieval = pd.DataFrame(
        [
            {"query_id": "Q1", "rank": 1, "chunk_id": "DOC-X_CH_001", "document_id": "DOC-X"},
            {"query_id": "Q1", "rank": 2, "chunk_id": "DOC-Y_CH_001", "document_id": "DOC-Y"},
            {"query_id": "Q1", "rank": 3, "chunk_id": "DOC-Z_CH_001", "document_id": "DOC-Z"},
        ]
    )

    gt_path = tmp_path / "ground_truth.csv"
    retrieval_path = tmp_path / "retrieval.csv"
    ground_truth.to_csv(gt_path, index=False)
    retrieval.to_csv(retrieval_path, index=False)

    plans, summary = recommend_experiments(gt_path, retrieval_path, k=3)

    assert len(plans) == 1
    assert plans.iloc[0]["root_cause_candidate"] == "cross_document_ambiguity"
    assert plans.iloc[0]["recommended_strategy"] == "metadata_prefiltering"
    assert plans.iloc[0]["priority"] == "high"
    assert summary["n_queries_evaluated"] == 1
    assert summary["hard_failures_input"] == 1
    assert summary["planned_experiments"] == 1
