from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


_REQUIRED_SUITE_COLUMNS = {
    "experiment_id",
    "strategy",
    "affected_query_ids",
    "affected_categories",
    "primary_metric",
    "metrics_to_watch",
    "experiment_hypothesis",
    "success_criteria",
    "regression_checks",
    "implementation_notes",
    "baseline_run",
    "status",
}

_RUNTIME_REQUIREMENTS: dict[str, dict[str, object]] = {
    "metadata_prefiltering": {
        "required_any_columns": ["document_id", "category"],
        "description": (
            "Runtime context must provide at least one legitimate metadata field "
            "(document_id or category) for each affected query."
        ),
    },
    "section_aware_reranking": {
        "required_any_columns": [],
        "description": (
            "Candidate retriever must expose section/title-aware ranking signals or a reranker."
        ),
    },
    "hybrid_retrieval_reranking_probe": {
        "required_any_columns": [],
        "description": (
            "Candidate retriever must implement a controlled hybrid or reranking probe."
        ),
    },
    "hybrid_lexical_vector_retrieval": {
        "required_any_columns": [],
        "description": (
            "Candidate retriever must combine lexical and vector retrieval signals."
        ),
    },
    "query_rewriting_or_decomposition": {
        "required_any_columns": [],
        "description": (
            "Candidate retriever must preserve the original query and produce rewritten "
            "or decomposed retrieval queries."
        ),
    },
    "retrieval_pipeline_reliability_check": {
        "required_any_columns": [],
        "description": (
            "Candidate run must restore valid retrieval execution before ranking is compared."
        ),
    },
    "reranking_or_cutoff_tuning": {
        "required_any_columns": [],
        "description": (
            "Candidate run must change reranking or cutoff behavior while preserving traceability."
        ),
    },
}


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _select_experiment(suite: pd.DataFrame, experiment_id: str) -> pd.Series:
    missing = sorted(_REQUIRED_SUITE_COLUMNS - set(suite.columns))
    if missing:
        raise ValueError(f"experiment_suite is missing required columns: {missing}")

    matched = suite.loc[suite["experiment_id"].astype(str) == str(experiment_id)]
    if matched.empty:
        raise ValueError(f"experiment_id not found: {experiment_id}")
    if len(matched) > 1:
        raise ValueError(f"duplicate experiment_id in suite: {experiment_id}")
    return matched.iloc[0]


def _baseline_ground_truth_hash(manifest: dict) -> str:
    value = (
        manifest.get("artifacts", {})
        .get("ground_truth", {})
        .get("sha256")
    )
    if not value:
        raise ValueError("baseline manifest must include ground-truth sha256")
    return str(value)


def _baseline_ks(manifest: dict) -> list[int]:
    return [
        int(value)
        for value in manifest.get("evaluation", {}).get("ks", [])
    ]


def prepare_candidate_contract(
    *,
    experiment_suite_path: str | Path,
    experiment_id: str,
    baseline_manifest_path: str | Path,
    candidate_run: str,
    k: int,
    runtime_context_path: str | Path | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Prepare a candidate-run contract without using ground-truth labels as runtime input."""
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if not str(candidate_run).strip():
        raise ValueError("candidate_run must be non-empty")

    suite = pd.read_csv(experiment_suite_path)
    experiment = _select_experiment(suite, experiment_id)
    strategy = str(experiment["strategy"])

    if strategy not in _RUNTIME_REQUIREMENTS:
        raise ValueError(f"unsupported strategy: {strategy!r}")

    baseline_manifest = _load_json(baseline_manifest_path)
    baseline_gt_hash = _baseline_ground_truth_hash(baseline_manifest)
    baseline_ks = _baseline_ks(baseline_manifest)
    if k not in baseline_ks:
        raise ValueError(f"baseline manifest does not include k={k}")

    affected_query_ids = [
        value
        for value in str(experiment["affected_query_ids"]).split("|")
        if value
    ]
    affected_categories = [
        value
        for value in str(experiment["affected_categories"]).split("|")
        if value
    ]

    runtime_requirements = _RUNTIME_REQUIREMENTS[strategy]
    required_any_columns = list(runtime_requirements["required_any_columns"])

    runtime_context = pd.DataFrame({"query_id": affected_query_ids})
    runtime_context_status = "not_required"
    blocking_reasons: list[str] = []

    if required_any_columns:
        runtime_context_status = "missing"
        if runtime_context_path is None:
            blocking_reasons.append(
                "strategy requires runtime metadata, but no runtime_context file was provided"
            )
        else:
            supplied = pd.read_csv(runtime_context_path)
            if "query_id" not in supplied.columns:
                raise ValueError("runtime_context must include query_id")

            supplied = supplied.copy()
            supplied["query_id"] = supplied["query_id"].astype(str)
            duplicates = sorted(
                supplied.loc[
                    supplied["query_id"].duplicated(),
                    "query_id",
                ].unique()
            )
            if duplicates:
                raise ValueError(
                    f"runtime_context has duplicate query_id values: {duplicates}"
                )

            available_filter_columns = [
                column
                for column in required_any_columns
                if column in supplied.columns
            ]
            if not available_filter_columns:
                blocking_reasons.append(
                    "runtime_context must include at least one legitimate filter field: "
                    + ", ".join(required_any_columns)
                )
            else:
                keep_columns = ["query_id", *available_filter_columns]
                runtime_context = pd.DataFrame(
                    {"query_id": affected_query_ids}
                ).merge(
                    supplied[keep_columns],
                    on="query_id",
                    how="left",
                    validate="one_to_one",
                )

                missing_rows: list[str] = []
                for row in runtime_context.to_dict(orient="records"):
                    has_value = any(
                        column in row
                        and row[column] is not None
                        and not pd.isna(row[column])
                        and str(row[column]).strip()
                        for column in available_filter_columns
                    )
                    if not has_value:
                        missing_rows.append(str(row["query_id"]))

                if missing_rows:
                    blocking_reasons.append(
                        "runtime_context is missing usable metadata for affected queries: "
                        f"{sorted(missing_rows)}"
                    )
                else:
                    runtime_context_status = "ready"

    contract_status = "blocked" if blocking_reasons else "ready"

    contract = {
        "contract_version": "1.0",
        "experiment_id": str(experiment_id),
        "strategy": strategy,
        "candidate_run": str(candidate_run),
        "status": contract_status,
        "baseline_run": str(experiment["baseline_run"]),
        "baseline_ground_truth_sha256": baseline_gt_hash,
        "audited_k": int(k),
        "affected_query_ids": affected_query_ids,
        "affected_categories": affected_categories,
        "primary_metric": str(experiment["primary_metric"]),
        "metrics_to_watch": str(experiment["metrics_to_watch"]),
        "experiment_hypothesis": str(experiment["experiment_hypothesis"]),
        "success_criteria": str(experiment["success_criteria"]),
        "regression_checks": str(experiment["regression_checks"]),
        "implementation_notes": str(experiment["implementation_notes"]),
        "runtime_context": {
            "status": runtime_context_status,
            "required_any_columns": required_any_columns,
            "requirement": str(runtime_requirements["description"]),
        },
        "blocking_reasons": blocking_reasons,
        "candidate_output_contract": {
            "required_files": [
                "retrieval_results.csv",
            ],
            "retrieval_schema_required_columns": [
                "query_id",
                "rank",
                "chunk_id",
            ],
            "evaluation_rule": (
                "Evaluate the candidate with the same frozen ground truth and K values as "
                "the baseline before running evaluate-experiment."
            ),
        },
        "oracle_guardrail": (
            "Do not derive runtime filter values or query rewrites from ground-truth labels, "
            "relevant_chunk_ids, expected answers, or other evaluation-only fields. "
            "Runtime context must come from information legitimately available to the "
            "production retrieval system at query time."
        ),
    }

    return contract, runtime_context
