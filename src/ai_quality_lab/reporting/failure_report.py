from __future__ import annotations

from collections.abc import Mapping


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_failure_report(
    summary: Mapping,
    title: str = "AI Quality Lab — Retrieval Failure Analysis",
) -> str:
    lines = [
        f"# {title}",
        "",
        "## Executive summary",
        "",
        f"Queries analyzed: **{summary.get('n_queries', 0)}**",
        f"Audited cutoff: **Top-{summary.get('k', 0)}**",
        f"Hard failures: **{summary.get('hard_failures', 0)}** "
        f"({_pct(float(summary.get('hard_failure_rate', 0.0)))})",
        f"Warnings: **{summary.get('warnings', 0)}**",
        f"Passes: **{summary.get('passes', 0)}**",
        "",
        "## Issue taxonomy",
        "",
        "| Issue | Count |",
        "|---|---:|",
    ]

    for issue, count in sorted(summary.get("issue_counts", {}).items()):
        lines.append(f"| {issue} | {count} |")

    by_category = summary.get("by_category") or {}
    if by_category:
        lines += [
            "",
            "## Hard failures by category",
            "",
            "| Category | Queries | Hard failures | Rate |",
            "|---|---:|---:|---:|",
        ]
        for category, values in sorted(by_category.items()):
            lines.append(
                f"| {category} | {values['n_queries']} | {values['hard_failures']} | "
                f"{_pct(float(values['hard_failure_rate']))} |"
            )

    lines += [
        "",
        "## Interpretation guardrail",
        "",
        "This report classifies observable retrieval behavior. Labels such as `top_k_miss`, "
        "`partial_coverage`, and `weak_ranking` do not by themselves prove a specific root cause "
        "such as bad embeddings, bad chunking, or an incorrect ground truth. Root-cause claims "
        "require additional evidence.",
        "",
    ]
    return "\n".join(lines)
