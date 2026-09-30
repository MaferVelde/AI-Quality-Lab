# AI Quality Lab v0.1

Motor reproducible para auditar sistemas RAG/agentes antes y después de cambios en retrieval.

## Qué resuelve v0.1

- Evalúa retrieval con métricas deterministas: Recall@K, Precision@K, Hit Rate@K, MRR y nDCG@K.
- Genera métricas por consulta y un resumen agregado.
- Versiona cada corrida mediante un `run_manifest.json` con hashes SHA-256.
- Compara una corrida candidata contra un baseline y detecta regresiones.
- Genera un reporte Markdown listo para convertir en entregable al cliente.
- Reserva una interfaz para métricas de generación (faithfulness, answer relevance, hallucination) sin acoplar el core a un proveedor LLM.

## Filosofía

1. Nunca sobrescribir artefactos congelados.
2. Toda métrica debe poder reproducirse.
3. Separar datos, evaluación, reporte y versionado.
4. El motor debe aceptar resultados de cualquier stack RAG.
5. Primero medir; después recomendar.

## Entrada mínima

### `ground_truth.csv`

```csv
query_id,question,relevant_chunk_ids,category
a1,"¿Qué es X?","chunk_001|chunk_014",ai
```

### `retrieval_results.csv`

```csv
query_id,rank,chunk_id,score,latency_ms
a1,1,chunk_014,0.91,118
a1,2,chunk_077,0.82,118
```

## Uso

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -e .

aiql evaluate \
  --ground-truth examples/minimal/ground_truth.csv \
  --retrieval examples/minimal/retrieval_results.csv \
  --output runs/example_v1 \
  --ks 3,5
```

Comparar dos corridas:

```bash
aiql compare \
  --baseline runs/baseline/summary.json \
  --candidate runs/candidate/summary.json \
  --output runs/comparison.json
```

## Salidas

Una evaluación produce:

```text
runs/example_v1/
├── per_query_metrics.csv
├── summary.json
├── run_manifest.json
└── report.md
```

## Roadmap

- v0.1: retrieval + reproducibilidad + regresión.
- v0.2: generación, citas, faithfulness y answer relevance.
- v0.3: LLM-as-a-judge configurable + casos adversariales.
- v0.4: dashboard web y ejecución por API.
- v1.0: evaluación continua y planes white-label para agencias.

## NuevaMente — Audit Run #001

NuevaMente is used as the first real validation case for AI Quality Lab.

For privacy and reproducibility, client/project datasets and generated audit runs are not stored in this repository. They are kept outside the Git repository under the local `client-data/` workspace.

AI Quality Lab v0.1.2 includes adapters capable of converting the frozen NuevaMente ground truth and retrieval contract into the canonical AIQL format.

### 1. Adapt frozen ground truth

```powershell
aiql adapt-nuevamente `
  --source data/clients/nuevamente/ground_truth/ground_truth_v1.csv `
  --chunks data/clients/nuevamente/corpus/chunks_v1.csv `
  --output data/clients/nuevamente/adapted/ground_truth_v1_aiql.csv
```

### 2. Adapt retrieval JSON

```powershell
aiql adapt-nuevamente-retrieval `
  --source data/clients/nuevamente/retrieval/raw/retrieval_results_agentes_v1.json `
  --chunks data/clients/nuevamente/corpus/chunks_v1.csv `
  --output data/clients/nuevamente/retrieval/adapted/retrieval_results_v1_aiql.csv
```

The original JSON remains unchanged. The adapter validates successful cases, Top-K size, unique ranks, and all retrieved chunk IDs against the frozen corpus.

### 3. Run the frozen v1 baseline audit

```powershell
aiql evaluate `
  --ground-truth data/clients/nuevamente/adapted/ground_truth_v1_aiql.csv `
  --retrieval data/clients/nuevamente/retrieval/adapted/retrieval_results_v1_aiql.csv `
  --output runs/nuevamente/audit_001_baseline_v1 `
  --ks 3,5
```

Audit Run #001 outputs:

- `per_query_metrics.csv`
- `summary.json`
- `run_manifest.json`
- `report.md`

The repository snapshot includes the validated Audit Run #001 generated from the supplied NuevaMente artifacts.
