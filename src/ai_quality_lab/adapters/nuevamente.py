from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_SOURCE_COLUMNS = {
    "case_id",
    "pregunta",
    "categoria",
    "relevant_chunk_ids",
}


def _normalize_chunk_ids(raw: object) -> str:
    if raw is None or pd.isna(raw):
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    # NuevaMente historically uses ';'. AI Quality Lab canonical format uses '|'.
    parts = [part.strip() for part in text.replace("|", ";").split(";") if part.strip()]
    # Stable de-duplication while preserving source order.
    return "|".join(dict.fromkeys(parts))


def adapt_ground_truth(
    source_path: str | Path,
    chunks_path: str | Path,
    output_path: str | Path,
) -> dict[str, int]:
    """Convert a frozen NuevaMente ground-truth CSV into AIQL's canonical contract.

    The source file is never modified. All referenced chunk IDs are validated
    against the supplied frozen chunks artifact before the adapted CSV is written.
    """
    source_path = Path(source_path)
    chunks_path = Path(chunks_path)
    output_path = Path(output_path)

    source = pd.read_csv(source_path)
    missing = REQUIRED_SOURCE_COLUMNS - set(source.columns)
    if missing:
        raise ValueError(f"NuevaMente ground truth missing columns: {sorted(missing)}")

    if source["case_id"].astype(str).duplicated().any():
        duplicates = sorted(
            source.loc[source["case_id"].astype(str).duplicated(), "case_id"].astype(str).unique()
        )
        raise ValueError(f"Duplicate case_id values: {duplicates}")

    chunks = pd.read_csv(chunks_path)
    if "chunk_id" not in chunks.columns:
        raise ValueError("chunks file missing required column: chunk_id")
    chunk_ids = set(chunks["chunk_id"].astype(str))

    out = pd.DataFrame(
        {
            "query_id": source["case_id"].astype(str),
            "question": source["pregunta"].astype(str),
            "relevant_chunk_ids": source["relevant_chunk_ids"].map(_normalize_chunk_ids),
            "category": source["categoria"].astype(str),
        }
    )

    # Preserve useful metadata under canonical names where available.
    optional_mapping = {
        "dificultad": "difficulty",
        "notas": "notes",
        "ground_truth_status": "ground_truth_status",
        "document_id": "source_document_id",
        "respuesta_esperada": "expected_answer",
        "seccion_evidencia": "evidence_section",
        "palabras_clave_evidencia": "evidence_keywords",
    }
    for source_col, target_col in optional_mapping.items():
        if source_col in source.columns:
            out[target_col] = source[source_col]

    empty = out["relevant_chunk_ids"].eq("")
    if empty.any():
        bad = out.loc[empty, "query_id"].tolist()
        raise ValueError(f"Queries without relevant chunks: {bad[:10]}")

    unknown: list[tuple[str, str]] = []
    for row in out[["query_id", "relevant_chunk_ids"]].to_dict(orient="records"):
        for chunk_id in row["relevant_chunk_ids"].split("|"):
            if chunk_id not in chunk_ids:
                unknown.append((row["query_id"], chunk_id))
    if unknown:
        preview = ", ".join(f"{qid}:{cid}" for qid, cid in unknown[:10])
        raise ValueError(f"Ground truth references unknown chunk IDs: {preview}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    return {
        "queries": int(len(out)),
        "chunks": int(len(chunks)),
        "multi_relevant_queries": int((out["relevant_chunk_ids"].str.contains(r"\|", regex=True)).sum()),
    }
