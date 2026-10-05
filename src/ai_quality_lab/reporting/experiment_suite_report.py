from __future__ import annotations

from collections.abc import Mapping


def build_experiment_suite_report(
    summary: Mapping,
    title: str = "AI Quality Lab — Experiment Suite",
) -> str:
    lines = [
        f"# {title}",
        "",
        "## Executive summary",
        "",
        f"Queries evaluated: **{summary.get('n_queries_evaluated', 0)}**",
        f"Audited cutoff: **Top-{summary.get('k', 0)}**",
        f"Hard failures received: **{summary.get('hard_failures_input', 0)}**",
        f"Recommendation plans received: **{summary.get('recommendation_plans_input', 0)}**",
        f"Unique affected queries: **{summary.get('affected_queries', 0)}**",
        f"Experiments planned: **{summary.get('planned_experiments', 0)}**",
        f"Baseline run: **{summary.get('baseline_run', 'baseline')}**",
        "",
        "## Experiment priorities",
        "",
        "| Priority | Experiments |",
        "|---|---:|",
    ]

    for priority, count in sorted(summary.get("priority_counts", {}).items()):
        lines.append(f"| {priority} | {count} |")

    lines += [
        "",
        "## Execution rule",
        "",
        "Implement one candidate intervention per experiment when possible. Keep the frozen "
        "ground truth and baseline artifacts unchanged, produce a separate candidate run, "
        "then compare the candidate against the baseline using the declared metrics and "
        "regression checks.",
        "",
        "## Interpretation guardrail",
        "",
        str(
            summary.get(
                "guardrail",
                "Experiment definitions are plans only; they are not accepted improvements.",
            )
        ),
        "",
    ]
    return "\n".join(lines)
