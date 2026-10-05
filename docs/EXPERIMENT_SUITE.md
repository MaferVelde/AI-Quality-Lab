# Experiment Suite Builder — v0.2-dev

The Experiment Suite Builder groups per-query recommendations into a small set of reproducible experiments.

Instead of treating every failed query as a separate engineering task, queries sharing the same recommended strategy are grouped into one experiment definition.

## Command

```bash
aiql build-experiment-suite \
  --ground-truth ground_truth.csv \
  --retrieval retrieval_results.csv \
  --output runs/experiment_suite \
  --baseline-run audit_001 \
  --k 5
```

Outputs:

```text
experiment_suite/
├── experiment_suite.csv
├── experiment_suite_summary.json
└── experiment_suite_report.md
```

## Experiment fields

- `experiment_id`
- `strategy`
- `root_cause_candidates`
- `affected_query_ids`
- `affected_categories`
- `n_affected_queries`
- `priority`
- `primary_metric`
- `metrics_to_watch`
- `experiment_hypothesis`
- `success_criteria`
- `regression_checks`
- `implementation_notes`
- `baseline_run`
- `candidate_run`
- `status`

## Deterministic ordering

Experiment IDs are generated reproducibly:

1. high priority before medium/low
2. more affected queries before fewer affected queries
3. strategy name as a deterministic tie-breaker

This keeps IDs stable while the frozen benchmark and recommendation output remain unchanged.

## Initial lifecycle

A generated experiment starts with:

```text
candidate_run = ""
status = planned
```

After implementing the intervention, the candidate retrieval run should be produced separately and evaluated with the existing AI Quality Lab evaluation pipeline.

The candidate run can then be compared against the frozen baseline.

## Guardrail

An experiment definition is not evidence that the intervention works.

The recommendation becomes an accepted improvement only after a separate candidate run demonstrates the intended gain without unacceptable regressions.
