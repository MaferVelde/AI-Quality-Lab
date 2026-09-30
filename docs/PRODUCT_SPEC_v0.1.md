# Product Spec — AI Quality Lab v0.1

## Customer

Initial customer: AI agencies, freelancers, startups and internal teams that already have a RAG/agent prototype or production system.

## Job to be done

"Tell me, with reproducible evidence, whether the retrieval layer improved or regressed after a change."

## Scope

### Included
- Ground-truth validation contract.
- Retrieval metrics at configurable K.
- Per-query error surface.
- Category-level metrics.
- Latency summary when provided.
- Run manifest + SHA-256 hashes.
- Baseline/candidate regression comparison.
- Markdown audit report.

### Explicitly out of scope for v0.1
- Automatic legal/compliance certification.
- Universal "AI score".
- Autonomous generation of ground truth without human validation.
- Provider-specific LLM judging in the deterministic retrieval core.
- Automatic deployment changes to a customer's production system.

## Acceptance criteria

1. Same inputs produce the same retrieval metrics.
2. Every run stores hashes for the supplied evaluation artifacts.
3. Ground truth with missing relevant chunks fails validation.
4. Duplicate `(query_id, rank)` rows fail validation.
5. Candidate vs baseline comparison identifies metric regressions.
6. The core accepts retrieval results produced by any vector database or framework.

## Commercial packaging

### Quick Audit — beta
- Up to 50 evaluation questions.
- Retrieval metrics @3 and @5.
- Error analysis.
- One written report.
- One review meeting.

Target beta price: **US$500**.

### Expansion after v0.1
- Full audit: generation + citations + hallucination.
- Continuous regression monitoring.
- White-label agency plan.
