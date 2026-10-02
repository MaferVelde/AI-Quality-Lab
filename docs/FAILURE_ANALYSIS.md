# Failure Analysis Engine — v0.2-dev

AI Quality Lab v0.2-dev adds a deterministic diagnostic layer on top of retrieval evaluation.

The goal is to explain **what observable retrieval behavior failed** before attempting to infer **why** it failed.

## Command

```bash
aiql analyze-failures \
  --ground-truth ground_truth.csv \
  --retrieval retrieval_results.csv \
  --output runs/failure_analysis \
  --k 5
```

The command produces:

```text
failure_analysis/
├── failure_analysis.csv
├── failure_summary.json
└── failure_report.md
```

## Deterministic taxonomy

### `no_results`
No retrieval rows were returned for the query.

Severity: `critical`

### `top_k_miss`
Results were returned, but none of the expected relevant chunks were found in the available retrieved list.

Severity: `high`

### `relevant_below_cutoff`
Relevant evidence exists in the retrieved list but appears below the audited Top-K cutoff.

Severity: `high`

### `partial_coverage`
At least one relevant chunk was retrieved within Top-K, but not all ground-truth relevant chunks were covered.

Severity: `medium`

### `weak_ranking`
All required evidence is present within Top-K, but the first relevant chunk appears below rank 1.

Severity: `medium`

### `pass`
No deterministic retrieval issue was detected at the audited cutoff.

Severity: `none`

## Hard failures

The following labels are counted as hard failures:

- `no_results`
- `top_k_miss`
- `relevant_below_cutoff`

`partial_coverage` and `weak_ranking` are warnings rather than hard failures.

## Guardrail

This layer does **not** claim that a failure proves bad embeddings, bad chunking, incorrect ground truth, poor query rewriting, or another specific root cause.

Those are hypotheses that require additional evidence.

The next diagnostic layer can use corpus text, scores, metadata, chunk structure, and ground-truth review to investigate root cause while keeping the deterministic taxonomy as the reproducible baseline.
