from __future__ import annotations

from collections import Counter
from math import ceil
from pathlib import Path
import re

import pandas as pd

from ai_quality_lab.analysis.failures import analyze_failures
from ai_quality_lab.evaluation.retrieval import load_retrieval


_CAUSE_RECOMMENDATIONS = {
    "cross_document_ambiguity": (
        "Add document/category context before retrieval, or apply metadata prefiltering "
        "when the query is only meaningful inside a known source document."
    ),
    "intradocument_competition": (
        "Use section/title-aware reranking or chunk metadata so the most specific evidence "
        "wins against nearby chunks from the same document."
    ),
    "technical_token_miss": (
        "Add lexical/hybrid retrieval (for example BM25 + vector search) and preserve exact "
        "technical tokens such as filenames, code symbols, endpoints, or identifiers."
    ),
    "inferential_query": (
        "Review whether the query should be rewritten into a more explicit factual form, "
        "or use a retrieval/generation step designed for inferential questions."
    ),
    "semantic_mismatch": (
        "Inspect query/chunk semantics and consider hybrid retrieval or reranking; the "
        "available structural signals do not isolate a more specific cause."
    ),
    "retrieval_execution_failure": (
        "Verify retriever execution, indexing, routing, and upstream availability before "
        "tuning retrieval quality."
    ),
    "below_cutoff_ranking": (
        "Tune reranking or the retrieval cutoff because relevant evidence is present but "
        "ranked below the audited K."
    ),
}

_CODELIKE_RE = re.compile(
    r"(?:\b[\w.-]+\.(?:py|js|ts|json|ya?ml|toml|html|css|java|sql|xml)\b"
    r"|@\w+|\w+\s*=\s*\w+|\w+\([^)]*\)|/\w+)",
    flags=re.IGNORECASE,
)

_INFERENTIAL_RE = re.compile(
    r"(?:\bse\s+limita\b|\búnicamente\b|\bsolamente\b|"
    r"\bonly\b|\blimited\s+to\b|\bes\s+obligatorio\b|\bis\s+required\b)",
    flags=re.IGNORECASE,
)

_OUTPUT_COLUMNS = [
    "query_id",
    "question",
    "category",
    "deterministic_issue",
    "root_cause_candidate",
    "confidence",
    "expected_document_id",
    "expected_document_hits_at_k",
    "distinct_documents_at_k",
    "retrieved_documents",
    "technical_signal",
    "technical_signal_source",
    "inferential_signal",
    "evidence",
    "recommended_action",
]


def _infer_document_id(chunk_id: str) -> str | None:
    value = str(chunk_id).strip()
    if "_CH_" in value:
        return value.split("_CH_", 1)[0]
    return None


def _read_optional_ground_truth_metadata(path: str | Path) -> dict[str, dict]:
    df = pd.read_csv(path)
    if "query_id" not in df.columns:
        return {}

    metadata: dict[str, dict] = {}
    for row in df.to_dict(orient="records"):
        qid = str(row["query_id"])
        metadata[qid] = {
            "source_document_id": None
            if pd.isna(row.get("source_document_id"))
            else str(row.get("source_document_id")),
            "expected_answer": ""
            if pd.isna(row.get("expected_answer"))
            else str(row.get("expected_answer")),
            "evidence_keywords": ""
            if pd.isna(row.get("evidence_keywords"))
            else str(row.get("evidence_keywords")),
        }
    return metadata


def _expected_document_id(
    metadata: dict,
    relevant_chunk_ids: str,
) -> str | None:
    explicit = metadata.get("source_document_id")
    if explicit:
        return str(explicit)

    inferred = {
        doc
        for doc in (
            _infer_document_id(chunk_id)
            for chunk_id in str(relevant_chunk_ids).split("|")
            if str(chunk_id).strip()
        )
        if doc
    }
    if len(inferred) == 1:
        return next(iter(inferred))
    return None


def _retrieved_documents(group: pd.DataFrame, k: int) -> list[str]:
    if group.empty:
        return []

    top = group.sort_values("rank").head(k)
    docs: list[str] = []
    for row in top.to_dict(orient="records"):
        document_id = row.get("document_id")
        if document_id is not None and not pd.isna(document_id):
            docs.append(str(document_id))
            continue
        inferred = _infer_document_id(str(row.get("chunk_id", "")))
        if inferred:
            docs.append(inferred)
    return docs


def _classify_candidate(
    deterministic_issue: str,
    question: str,
    metadata: dict,
    expected_document_id: str | None,
    documents: list[str],
    k: int,
) -> tuple[str, str, bool, bool]:
    if deterministic_issue == "no_results":
        return "retrieval_execution_failure", "high", False, False

    if deterministic_issue == "relevant_below_cutoff":
        return "below_cutoff_ranking", "high", False, False

    expected_hits = (
        sum(doc == expected_document_id for doc in documents)
        if expected_document_id
        else 0
    )
    distinct_documents = len(set(documents))

    question_technical_signal = bool(_CODELIKE_RE.search(str(question)))
    reference_text = " ".join(
        [
            str(metadata.get("expected_answer", "")),
            str(metadata.get("evidence_keywords", "")),
        ]
    )
    reference_technical_signal = bool(_CODELIKE_RE.search(reference_text))
    technical_signal = question_technical_signal or reference_technical_signal
    inferential_signal = bool(_INFERENTIAL_RE.search(str(question)))

    if inferential_signal:
        return "inferential_query", "medium", technical_signal, True

    intradocument_threshold = max(3, ceil(k * 0.6))
    strong_intradocument_threshold = max(intradocument_threshold, ceil(k * 0.8))

    # A technical token present only in reference metadata is not sufficient by
    # itself to explain a miss. Require either a technical signal in the query
    # or strong concentration of retrieved chunks in the expected document.
    if technical_signal and (
        question_technical_signal or expected_hits >= strong_intradocument_threshold
    ):
        return "technical_token_miss", "medium", True, False

    if expected_document_id and expected_hits >= intradocument_threshold:
        confidence = (
            "high"
            if expected_hits >= strong_intradocument_threshold
            or distinct_documents <= 2
            else "medium"
        )
        return "intradocument_competition", confidence, technical_signal, False

    if expected_document_id and expected_hits <= 1 and distinct_documents >= 3:
        return "cross_document_ambiguity", "high", technical_signal, False

    return "semantic_mismatch", "medium", technical_signal, False


def analyze_root_causes(
    ground_truth_path: str | Path,
    retrieval_path: str | Path,
    k: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """Generate evidence-based root-cause candidates for deterministic hard failures.

    The returned labels are hypotheses supported by observable retrieval signals,
    not proven causal explanations.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")

    failure_analysis, failure_summary = analyze_failures(
        ground_truth_path=ground_truth_path,
        retrieval_path=retrieval_path,
        k=k,
    )
    hard_failures = failure_analysis.loc[failure_analysis["hard_failure"]].copy()
    retrieval = load_retrieval(retrieval_path)
    metadata = _read_optional_ground_truth_metadata(ground_truth_path)
    grouped = {
        qid: group.copy()
        for qid, group in retrieval.groupby("query_id", sort=False)
    }

    rows: list[dict] = []
    for item in hard_failures.to_dict(orient="records"):
        qid = str(item["query_id"])
        group = grouped.get(qid, retrieval.iloc[0:0].copy())
        documents = _retrieved_documents(group, k)
        gt_metadata = metadata.get(qid, {})
        expected_document = _expected_document_id(
            gt_metadata,
            str(item.get("relevant_chunk_ids", "")),
        )
        expected_hits = (
            sum(doc == expected_document for doc in documents)
            if expected_document
            else 0
        )
        distinct_documents = len(set(documents))

        cause, confidence, technical_signal, inferential_signal = _classify_candidate(
            deterministic_issue=str(item["primary_issue"]),
            question=str(item["question"]),
            metadata=gt_metadata,
            expected_document_id=expected_document,
            documents=documents,
            k=k,
        )

        question_technical_signal = bool(_CODELIKE_RE.search(str(item["question"])))
        reference_text = " ".join(
            [
                str(gt_metadata.get("expected_answer", "")),
                str(gt_metadata.get("evidence_keywords", "")),
            ]
        )
        reference_technical_signal = bool(_CODELIKE_RE.search(reference_text))
        if question_technical_signal and reference_technical_signal:
            technical_signal_source = "question|reference"
        elif question_technical_signal:
            technical_signal_source = "question"
        elif reference_technical_signal:
            technical_signal_source = "reference"
        else:
            technical_signal_source = ""

        evidence_parts = [
            f"deterministic_issue={item['primary_issue']}",
            f"expected_document={expected_document or 'unknown'}",
            f"expected_document_hits={expected_hits}/{len(documents)}",
            f"distinct_documents={distinct_documents}",
        ]
        if technical_signal:
            evidence_parts.append(
                f"technical_signal=true:{technical_signal_source or 'unknown'}"
            )
        if inferential_signal:
            evidence_parts.append("inferential_signal=true")

        rows.append(
            {
                "query_id": qid,
                "question": item["question"],
                "category": item.get("category"),
                "deterministic_issue": item["primary_issue"],
                "root_cause_candidate": cause,
                "confidence": confidence,
                "expected_document_id": expected_document,
                "expected_document_hits_at_k": expected_hits,
                "distinct_documents_at_k": distinct_documents,
                "retrieved_documents": "|".join(documents),
                "technical_signal": technical_signal,
                "technical_signal_source": technical_signal_source,
                "inferential_signal": inferential_signal,
                "evidence": "; ".join(evidence_parts),
                "recommended_action": _CAUSE_RECOMMENDATIONS[cause],
            }
        )

    analysis = pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)
    cause_counts = (
        dict(sorted(Counter(analysis["root_cause_candidate"]).items()))
        if not analysis.empty
        else {}
    )
    confidence_counts = (
        dict(sorted(Counter(analysis["confidence"]).items()))
        if not analysis.empty
        else {}
    )

    summary: dict = {
        "n_queries_evaluated": int(failure_summary["n_queries"]),
        "k": int(k),
        "hard_failures_input": int(failure_summary["hard_failures"]),
        "diagnosed_cases": int(len(analysis)),
        "cause_counts": cause_counts,
        "confidence_counts": confidence_counts,
        "guardrail": (
            "Root-cause labels are evidence-based candidates, not proven causal claims."
        ),
    }

    if not analysis.empty and analysis["category"].notna().any():
        by_category: dict[str, dict[str, int]] = {}
        for category, group in analysis.dropna(subset=["category"]).groupby("category"):
            by_category[str(category)] = dict(
                sorted(Counter(group["root_cause_candidate"]).items())
            )
        summary["by_category"] = by_category

    return analysis, summary
