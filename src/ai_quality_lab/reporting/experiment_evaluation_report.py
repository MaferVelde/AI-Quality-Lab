from __future__ import annotations

from collections.abc import Mapping


def build_experiment_evaluation_report(
    result: Mapping,
    title: str = "AI Quality Lab — Experiment Evaluation",
) -> str:
    affected = result.get("affected_queries", {})
    global_primary = result.get("global_primary_metric", {})

    lines = [
        f"# {title}",
        "",
        "## Decision",
        "",
        f"Experiment: **{result.get('experiment_id', '')}**",
        f"Strategy: **{result.get('strategy', '')}**",
        f"Outcome: **{str(result.get('outcome', '')).upper()}**",
        f"Baseline run: **{result.get('baseline_run', '')}**",
        f"Candidate run: **{result.get('candidate_run', '')}**",
        f"Primary metric: **{result.get('primary_metric', '')}**",
        "",
        "## Affected queries",
        "",
        f"Queries: **{affected.get('n_affected_queries', 0)}**",
        f"Baseline mean: **{float(affected.get('baseline_mean', 0.0)):.4f}**",
        f"Candidate mean: **{float(affected.get('candidate_mean', 0.0)):.4f}**",
        f"Delta: **{float(affected.get('delta', 0.0)):+.4f}**",
        f"Status: **{affected.get('status', '')}**",
        "",
        "## Global primary metric",
        "",
        f"Baseline: **{float(global_primary.get('baseline', 0.0)):.4f}**",
        f"Candidate: **{float(global_primary.get('candidate', 0.0)):.4f}**",
        f"Delta: **{float(global_primary.get('delta', 0.0)):+.4f}**",
        "",
        "## Decision reasons",
        "",
    ]
    for reason in result.get("decision_reasons", []):
        lines.append(f"- {reason}")

    lines += [
        "",
        "## Interpretation guardrail",
        "",
        str(result.get("guardrail", "")),
        "",
    ]
    return "\n".join(lines)
