from __future__ import annotations

from collections.abc import Mapping


def build_root_cause_report(
    summary: Mapping,
    title: str = "AI Quality Lab — Root Cause Analysis",
) -> str:
    lines = [
        f"# {title}",
        "",
        "## Executive summary",
        "",
        f"Queries evaluated: **{summary.get('n_queries_evaluated', 0)}**",
        f"Audited cutoff: **Top-{summary.get('k', 0)}**",
        f"Hard failures received: **{summary.get('hard_failures_input', 0)}**",
        f"Cases diagnosed: **{summary.get('diagnosed_cases', 0)}**",
        "",
        "## Root-cause candidates",
        "",
        "| Candidate | Count |",
        "|---|---:|",
    ]

    for cause, count in sorted(summary.get("cause_counts", {}).items()):
        lines.append(f"| {cause} | {count} |")

    lines += [
        "",
        "## Confidence",
        "",
        "| Confidence | Count |",
        "|---|---:|",
    ]
    for confidence, count in sorted(summary.get("confidence_counts", {}).items()):
        lines.append(f"| {confidence} | {count} |")

    lines += [
        "",
        "## Interpretation guardrail",
        "",
        str(
            summary.get(
                "guardrail",
                "Root-cause labels are evidence-based candidates, not proven causal claims.",
            )
        ),
        "",
        "A candidate such as `cross_document_ambiguity` or "
        "`intradocument_competition` describes a reproducible retrieval pattern. "
        "Confirming the true cause may still require corpus inspection, query rewriting "
        "experiments, reranking tests, or ground-truth review.",
        "",
    ]
    return "\n".join(lines)
