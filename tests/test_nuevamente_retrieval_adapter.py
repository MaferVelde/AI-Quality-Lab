import json
from pathlib import Path

import pandas as pd

from ai_quality_lab.adapters.nuevamente_retrieval import adapt_retrieval_json


def test_adapt_retrieval_json(tmp_path: Path):
    chunks = pd.DataFrame({"chunk_id": ["C1", "C2"]})
    chunks_path = tmp_path / "chunks.csv"
    chunks.to_csv(chunks_path, index=False)

    source = [
        {
            "contract_version": "1.0",
            "case_id": "Q1",
            "query": "Question",
            "top_k": 2,
            "score_type": "cosine_similarity",
            "status": "success",
            "results": [
                {"rank": 1, "chunk_id": "C1", "document_id": "D1", "score": 0.9, "metadata": {"categoria": "IA"}},
                {"rank": 2, "chunk_id": "C2", "document_id": "D2", "score": 0.8, "metadata": {"categoria": "IA"}},
            ],
            "error": None,
        }
    ]
    source_path = tmp_path / "retrieval.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    output = tmp_path / "retrieval.csv"

    summary = adapt_retrieval_json(source_path, chunks_path, output)
    df = pd.read_csv(output)

    assert summary["queries"] == 1
    assert summary["rows"] == 2
    assert list(df["query_id"]) == ["Q1", "Q1"]
    assert list(df["rank"]) == [1, 2]
