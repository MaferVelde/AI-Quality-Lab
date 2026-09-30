import pandas as pd

from ai_quality_lab.adapters.nuevamente import adapt_ground_truth


def test_adapt_ground_truth(tmp_path):
    source = tmp_path / "gt.csv"
    chunks = tmp_path / "chunks.csv"
    output = tmp_path / "canonical.csv"

    pd.DataFrame([
        {
            "case_id": "Q1",
            "pregunta": "Pregunta",
            "categoria": "IA",
            "relevant_chunk_ids": "C1;C2",
            "document_id": "D1",
            "dificultad": "Básica",
        }
    ]).to_csv(source, index=False)
    pd.DataFrame([{"chunk_id": "C1"}, {"chunk_id": "C2"}]).to_csv(chunks, index=False)

    summary = adapt_ground_truth(source, chunks, output)
    result = pd.read_csv(output)

    assert result.loc[0, "query_id"] == "Q1"
    assert result.loc[0, "question"] == "Pregunta"
    assert result.loc[0, "category"] == "IA"
    assert result.loc[0, "relevant_chunk_ids"] == "C1|C2"
    assert summary["multi_relevant_queries"] == 1
