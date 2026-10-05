# Recommendation & Experiment Planning Engine — v0.2-dev

This layer converts evidence-based root-cause candidates into deterministic, testable experiment plans.

It does **not** claim that a recommendation will fix the system.

The intended workflow is:

```text
Retrieval Evaluation
        ↓
Failure Analysis
        ↓
Root Cause Analysis
        ↓
Recommendation & Experiment Planning
        ↓
Candidate implementation
        ↓
Baseline vs candidate comparison
```

## Command

```bash
aiql recommend-experiments \
  --ground-truth ground_truth.csv \
  --retrieval retrieval_results.csv \
  --output runs/experiment_plan \
  --k 5
```

Outputs:

```text
experiment_plan/
├── experiment_plan.csv
├── experiment_summary.json
└── experiment_report.md
```

## Output fields

- `query_id`
- `category`
- `root_cause_candidate`
- `confidence`
- `priority`
- `recommended_strategy`
- `experiment_hypothesis`
- `primary_metric`
- `metrics_to_watch`
- `success_criteria`
- `regression_checks`
- `implementation_notes`

## Strategy mapping

| Root-cause candidate | Recommended strategy |
|---|---|
| cross_document_ambiguity | metadata_prefiltering |
| intradocument_competition | section_aware_reranking |
| technical_token_miss | hybrid_lexical_vector_retrieval |
| inferential_query | query_rewriting_or_decomposition |
| semantic_mismatch | hybrid_retrieval_reranking_probe |
| retrieval_execution_failure | retrieval_pipeline_reliability_check |
| below_cutoff_ranking | reranking_or_cutoff_tuning |

## Priority

The first version intentionally keeps priority simple and auditable:

- high-confidence diagnosis → high-priority experiment
- medium-confidence diagnosis → medium-priority experiment

Priority is not business severity. It represents how directly the current diagnostic evidence supports trying the proposed experiment.

## Validation rule

Every recommended intervention must be evaluated against the same frozen ground truth and baseline artifacts.

A recommendation becomes an accepted improvement only after a baseline-versus-candidate comparison shows the intended gain without unacceptable regressions.

## Guardrail

Recommendations are hypotheses, not guaranteed fixes.

Do not claim that metadata filtering, BM25, reranking, query rewriting, or another intervention solved a root cause until the candidate system has been run and evaluated independently.
