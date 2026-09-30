# Audit Workflow v0.1

1. **Intake** — define system boundary, use cases, target population and prohibited claims.
2. **Freeze artifacts** — corpus/chunks, ground truth and run configuration.
3. **Validate ground truth** — human review required.
4. **Collect retrieval results** — top-N chunks for every query.
5. **Evaluate** — compute deterministic metrics and per-query failures.
6. **Analyze errors** — classify failures by cause, not just by metric.
7. **Compare** — baseline vs candidate when changing model/chunking/retriever.
8. **Report** — evidence, limitations, recommendations and next experiment.
9. **Archive** — preserve manifests/hashes for reproducibility.
