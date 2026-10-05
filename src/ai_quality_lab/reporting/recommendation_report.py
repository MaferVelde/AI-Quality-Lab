from __future__ import annotations

from collections.abc import Mapping


def build_recommendation_report(
    summary: Mapping,
    title: str = "AI Quality Lab — Recommendation & Experiment Plan",
) -> str:
    lines = [
        f"# {title}",
        "",
        "## Executive summary",
        "",
        f"Queries evaluated: **{summary.get('n_queries_evaluated', 0)}**",
        f"Audited cutoff: **Top-{summary.get('k', 0)}**",
        f"Hard failures received: **{summary.get('hard_failures_input', 0)}**",
        f"Experiment plans generated: **{summary.get('planned_experiments', 0)}**",
        "",
        "## Recommended strategies",
        "",
        "| Strategy | Cases |",
        "|---|---:|",
    ]

    for strategy, count in sorted(summary.get("strategy_counts", {}).items()):
        lines.append(f"| {strategy} | {count} |")

    lines += [
        "",
        "## Priority",
        "",
        "| Priority | Cases |",
        "|---|---:|",
    ]
    for priority, count in sorted(summary.get("priority_counts", {}).items()):
        lines.append(f"| {priority} | {count} |")

    lines += [
        "",
        "## Experiment rule",
        "",
        "Every recommendation is a testable hypothesis. Keep the ground truth and baseline "
        "retrieval artifacts frozen, change one intervention at a time when possible, rerun "
        "the same evaluation, and compare baseline versus candidate metrics.",
        "",
        "## Interpretation guardrail",
        "",
        str(
            summary.get(
                "guardrail",
                "Recommendations are experiment hypotheses, not guaranteed fixes.",
            )
        ),
        "",
    ]
    return "\n".join(lines)
