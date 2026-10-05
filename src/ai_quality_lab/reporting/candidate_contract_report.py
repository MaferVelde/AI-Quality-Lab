from __future__ import annotations

from collections.abc import Mapping


def build_candidate_contract_report(
    contract: Mapping,
    title: str = "AI Quality Lab — Candidate Run Contract",
) -> str:
    lines = [
        f"# {title}",
        "",
        "## Candidate",
        "",
        f"Experiment: **{contract.get('experiment_id', '')}**",
        f"Strategy: **{contract.get('strategy', '')}**",
        f"Candidate run: **{contract.get('candidate_run', '')}**",
        f"Contract status: **{str(contract.get('status', '')).upper()}**",
        f"Baseline run: **{contract.get('baseline_run', '')}**",
        f"Audited K: **{contract.get('audited_k', '')}**",
        "",
        "## Evaluation target",
        "",
        f"Primary metric: **{contract.get('primary_metric', '')}**",
        f"Affected queries: **{len(contract.get('affected_query_ids', []))}**",
        "",
        "## Runtime context",
        "",
        str(contract.get("runtime_context", {}).get("requirement", "")),
        "",
    ]

    blocking = list(contract.get("blocking_reasons", []))
    if blocking:
        lines += ["## Blocking reasons", ""]
        lines.extend(f"- {reason}" for reason in blocking)
        lines.append("")

    lines += [
        "## Oracle-leakage guardrail",
        "",
        str(contract.get("oracle_guardrail", "")),
        "",
        "## Next step",
        "",
        "Produce the candidate retrieval artifact outside the evaluator, then evaluate it "
        "against the same frozen ground truth and baseline manifest.",
        "",
    ]
    return "\n".join(lines)
