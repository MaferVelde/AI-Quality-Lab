from pathlib import Path

import pandas as pd

from ai_quality_lab.analysis.root_causes import analyze_root_causes


def test_root_cause_candidates(tmp_path: Path):
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
                "question": "¿Qué archivo contiene la configuración?",
                "relevant_chunk_ids": "DOC-B_CH_010",
                "category": "B",
                "source_document_id": "DOC-B",
                "expected_answer": "settings.py",
                "evidence_keywords": "settings.py; configuración",
            },
            {
                "query_id": "Q3",
                "question": "¿Qué aprende el lector sobre los elementos?",
                "relevant_chunk_ids": "DOC-C_CH_001",
                "category": "C",
                "source_document_id": "DOC-C",
                "expected_answer": "Cómo se estructuran.",
                "evidence_keywords": "estructuran los elementos",
            },
            {
                "query_id": "Q4",
                "question": "¿El módulo se limita únicamente a sintaxis básica?",
                "relevant_chunk_ids": "DOC-D_CH_001",
                "category": "D",
                "source_document_id": "DOC-D",
                "expected_answer": "No.",
                "evidence_keywords": "condicionales; funciones; eventos",
            },
            {
                "query_id": "Q5",
                "question": "¿Qué representa la variable app?",
                "relevant_chunk_ids": "DOC-E_CH_010",
                "category": "E",
                "source_document_id": "DOC-E",
                "expected_answer": "Una instancia.",
                "evidence_keywords": "app = FastAPI(); instancia",
            },
        ]
    )

    retrieval_rows = [
        # Q1: expected document absent -> cross-document ambiguity
        {"query_id": "Q1", "rank": 1, "chunk_id": "DOC-X_CH_001", "document_id": "DOC-X"},
        {"query_id": "Q1", "rank": 2, "chunk_id": "DOC-Y_CH_001", "document_id": "DOC-Y"},
        {"query_id": "Q1", "rank": 3, "chunk_id": "DOC-Z_CH_001", "document_id": "DOC-Z"},
        # Q2: same document dominates + technical token -> technical token miss
        {"query_id": "Q2", "rank": 1, "chunk_id": "DOC-B_CH_001", "document_id": "DOC-B"},
        {"query_id": "Q2", "rank": 2, "chunk_id": "DOC-B_CH_002", "document_id": "DOC-B"},
        {"query_id": "Q2", "rank": 3, "chunk_id": "DOC-B_CH_003", "document_id": "DOC-B"},
        # Q3: same document dominates -> intradocument competition
        {"query_id": "Q3", "rank": 1, "chunk_id": "DOC-C_CH_020", "document_id": "DOC-C"},
        {"query_id": "Q3", "rank": 2, "chunk_id": "DOC-C_CH_021", "document_id": "DOC-C"},
        {"query_id": "Q3", "rank": 3, "chunk_id": "DOC-C_CH_022", "document_id": "DOC-C"},
        # Q4: inferential wording
        {"query_id": "Q4", "rank": 1, "chunk_id": "DOC-X_CH_010", "document_id": "DOC-X"},
        {"query_id": "Q4", "rank": 2, "chunk_id": "DOC-D_CH_020", "document_id": "DOC-D"},
        {"query_id": "Q4", "rank": 3, "chunk_id": "DOC-Y_CH_010", "document_id": "DOC-Y"},
        # Q5: unresolved structural pattern -> semantic mismatch
        {"query_id": "Q5", "rank": 1, "chunk_id": "DOC-E_CH_001", "document_id": "DOC-E"},
        {"query_id": "Q5", "rank": 2, "chunk_id": "DOC-X_CH_002", "document_id": "DOC-X"},
        {"query_id": "Q5", "rank": 3, "chunk_id": "DOC-X_CH_003", "document_id": "DOC-X"},
    ]
    retrieval = pd.DataFrame(retrieval_rows)

    gt_path = tmp_path / "ground_truth.csv"
    retrieval_path = tmp_path / "retrieval.csv"
    ground_truth.to_csv(gt_path, index=False)
    retrieval.to_csv(retrieval_path, index=False)

    analysis, summary = analyze_root_causes(gt_path, retrieval_path, k=3)
    causes = dict(zip(analysis["query_id"], analysis["root_cause_candidate"]))

    assert causes["Q1"] == "cross_document_ambiguity"
    assert causes["Q2"] == "technical_token_miss"
    assert causes["Q3"] == "intradocument_competition"
    assert causes["Q4"] == "inferential_query"
    assert causes["Q5"] == "semantic_mismatch"

    indexed = analysis.set_index("query_id")
    assert indexed.loc["Q2", "technical_signal_source"] == "reference"
    assert indexed.loc["Q5", "technical_signal_source"] == "reference"

    assert summary["hard_failures_input"] == 5
    assert summary["diagnosed_cases"] == 5


def test_intradocument_competition_confidence_is_medium_when_document_only_partly_dominates(
    tmp_path: Path,
):
    ground_truth = pd.DataFrame(
        [
            {
                "query_id": "Q1",
                "question": "¿Qué temas cubre este módulo?",
                "relevant_chunk_ids": "DOC-A_CH_010",
                "category": "A",
                "source_document_id": "DOC-A",
                "expected_answer": "Tema específico.",
                "evidence_keywords": "tema específico",
            }
        ]
    )
    retrieval = pd.DataFrame(
        [
            {"query_id": "Q1", "rank": 1, "chunk_id": "DOC-A_CH_001", "document_id": "DOC-A"},
            {"query_id": "Q1", "rank": 2, "chunk_id": "DOC-A_CH_002", "document_id": "DOC-A"},
            {"query_id": "Q1", "rank": 3, "chunk_id": "DOC-A_CH_003", "document_id": "DOC-A"},
            {"query_id": "Q1", "rank": 4, "chunk_id": "DOC-B_CH_001", "document_id": "DOC-B"},
            {"query_id": "Q1", "rank": 5, "chunk_id": "DOC-C_CH_001", "document_id": "DOC-C"},
        ]
    )

    gt_path = tmp_path / "ground_truth_medium.csv"
    retrieval_path = tmp_path / "retrieval_medium.csv"
    ground_truth.to_csv(gt_path, index=False)
    retrieval.to_csv(retrieval_path, index=False)

    analysis, _ = analyze_root_causes(gt_path, retrieval_path, k=5)
    row = analysis.iloc[0]

    assert row["root_cause_candidate"] == "intradocument_competition"
    assert row["confidence"] == "medium"
