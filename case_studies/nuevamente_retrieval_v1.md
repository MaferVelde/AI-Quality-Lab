# NuevaMente — historical retrieval baseline

This file records the initial internal case that motivates AI Quality Lab v0.1.

## Frozen evaluation context
- 10 Spanish-language documents.
- 5 categories.
- Ground truth: 50/50 validated questions.
- Retrieval baseline evaluated at K=3 and K=5.

## Historical metrics
- Recall@3: 0.5733
- Recall@5: 0.7533
- Precision@3: 0.2133
- Precision@5: 0.1640

After reviewing the ground truth and broadening three cases to include multiple valid chunks, 11 critical cases remained with Recall@5 = 0.

## Error taxonomy observed
- AMBIGUEDAD_CONTEXTO: 4
- RANKING_SEMANTICO: 3
- COMPETENCIA_INTRADOCUMENTO: 2
- ENTIDAD_LEXICAL: 1
- PREGUNTA_INFERENCIAL: 1

## Versioning constraint
The historical `ground_truth_v1` must remain immutable. Any corrected reference set must be released as a new ground-truth version with its own manifest, then re-evaluated without deleting the v1 baseline.
