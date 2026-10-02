# Evidence-based Root Cause Analysis — v0.2-dev

This layer runs after deterministic failure analysis.

It answers a narrower question:

> Which observable retrieval pattern is the strongest candidate explanation for each hard failure?

The output is intentionally phrased as a **root-cause candidate**, not a proven cause.

## Command

```bash
aiql analyze-root-causes \
  --ground-truth ground_truth.csv \
  --retrieval retrieval_results.csv \
  --output runs/root_cause_analysis \
  --k 5
```

Outputs:

```text
root_cause_analysis/
├── root_cause_analysis.csv
├── root_cause_summary.json
└── root_cause_report.md
```

## Current candidate taxonomy

### cross_document_ambiguity
The expected source document has zero or very few Top-K hits while several other documents compete for the query.

Typical action: add document/category context or metadata prefiltering.

### intradocument_competition
The expected document dominates Top-K, but the relevant chunk is still missing.

Typical action: section/title-aware reranking or better chunk metadata.

### technical_token_miss
The expected document is represented, but the answer depends on an exact technical token such as a filename, code symbol, endpoint, or identifier.

Typical action: hybrid lexical + vector retrieval.

### inferential_query
The query wording requires an inference rather than a direct factual match.

Typical action: query rewriting, explicit factual decomposition, or an inference-aware pipeline.

### semantic_mismatch
The failure does not have enough structural evidence for a more specific label.

Typical action: inspect query/chunk semantics and test hybrid retrieval or reranking.

### retrieval_execution_failure
No results were returned.

### below_cutoff_ranking
Relevant evidence exists in the retrieval list but falls below the audited K.

## Confidence

Current rules emit `high` or `medium` confidence based on structural evidence.

Confidence describes confidence in the **pattern classification**, not proof of causality.

## Guardrail

The engine must never convert these labels into statements such as:

- "the embedding model is bad"
- "the chunking strategy caused the failure"
- "the ground truth is wrong"

unless an independent experiment or review provides that evidence.

The deterministic failure layer remains the reproducible baseline. Root-cause candidates are a second diagnostic layer intended to guide the next experiment.
