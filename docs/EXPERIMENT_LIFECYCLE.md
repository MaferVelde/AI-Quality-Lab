# Experiment Lifecycle & Candidate Evaluator — v0.2-dev

This layer evaluates a candidate retrieval run against a frozen baseline for one experiment in the generated experiment suite.

It does not implement the retrieval intervention itself.

## Decision model

An experiment can end as:

- `accept`: the affected-query primary metric improves and no global retrieval metric regresses beyond tolerance.
- `reject`: at least one global metric regresses beyond tolerance, or the affected-query primary metric regresses.
- `inconclusive`: the affected-query primary metric does not change beyond tolerance and no global regression is detected.

## Compatibility guardrail

Before comparing metrics, AI Quality Lab requires:

1. baseline and candidate manifests contain a ground-truth SHA-256;
2. both hashes are identical;
3. both manifests contain the audited K.

This prevents accidentally comparing a candidate evaluated with a different Ground Truth version.

## Command

```bash
aiql evaluate-experiment \
  --experiment-suite experiment_suite.csv \
  --experiment-id EXP-001 \
  --baseline-summary baseline/summary.json \
  --candidate-summary candidate/summary.json \
  --baseline-per-query baseline/per_query_metrics.csv \
  --candidate-per-query candidate/per_query_metrics.csv \
  --baseline-manifest baseline/run_manifest.json \
  --candidate-manifest candidate/run_manifest.json \
  --candidate-run exp_001_candidate_v1 \
  --output runs/exp_001_evaluation \
  --k 5
```

Outputs:

```text
exp_001_evaluation/
├── experiment_result.json
├── affected_query_comparison.csv
├── experiment_suite_updated.csv
└── experiment_evaluation_report.md
```

## Why affected queries are evaluated separately

A global metric can hide whether the intervention helped the specific failure cases that motivated the experiment.

The evaluator therefore checks both:

- global metric changes across the full evaluation set;
- the declared primary metric over only the experiment's affected query IDs.

## Important baseline rule

The baseline must be evaluated with the same frozen Ground Truth used to create the experiment diagnostics.

For example, if the experiment suite was derived from Ground Truth v2, the baseline and candidate evaluations must both use Ground Truth v2.

## Guardrail

An accepted experiment means only that this candidate improved the declared benchmark under the current decision rules.

It does not prove that the root-cause candidate was the true causal mechanism.
