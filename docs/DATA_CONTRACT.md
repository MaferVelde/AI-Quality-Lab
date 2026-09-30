# Data Contract v0.1

## Ground truth

Required columns:
- `query_id`: stable unique identifier.
- `question`: natural-language evaluation query.
- `relevant_chunk_ids`: one or more valid chunk IDs separated by `|`.

Optional:
- `category`
- `difficulty`
- `language`
- `notes`
- `ground_truth_status`

Rules:
- A frozen version is immutable.
- Corrections create a new version (`v2`, `v3`, ...).
- Every released ground-truth artifact should be referenced by a manifest with a content hash.

## Retrieval results

Required:
- `query_id`
- `rank`
- `chunk_id`

Optional:
- `score`
- `latency_ms`
- `retriever`
- `embedding_model`
- `index_version`

## Run naming

Recommended:
`YYYYMMDD_<client>_<system>_<run_label>`

Example:
`20260925_nuevamente_rag_retrieval_v1`

## NuevaMente adapter (v0.1.1)

Frozen NuevaMente files are not edited. Convert them into the canonical AIQL contract with:

```powershell
aiql adapt-nuevamente --source data/clients/nuevamente/ground_truth/ground_truth_v1.csv --chunks data/clients/nuevamente/corpus/chunks_v1.csv --output data/clients/nuevamente/adapted/ground_truth_v1_aiql.csv
```

Mapping:
- `case_id` -> `query_id`
- `pregunta` -> `question`
- `categoria` -> `category`
- `relevant_chunk_ids` remains the same concept, but `;` is normalized to AIQL's canonical `|` delimiter.

The adapter validates every referenced chunk ID against `chunks_v1.csv` before writing the adapted artifact.
