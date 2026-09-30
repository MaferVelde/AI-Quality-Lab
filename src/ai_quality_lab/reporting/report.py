from __future__ import annotations

from collections.abc import Mapping


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_markdown_report(summary: Mapping, title: str = "AI Quality Lab — Retrieval Audit") -> str:
    metrics = summary.get("metrics", {})
    lines = [
        f"# {title}",
        "",
        "## Executive summary",
        "",
        f"Queries evaluated: **{summary.get('n_queries', 0)}**",
        "",
        "## Retrieval metrics",
        "",
        "| Metric | Result |",
        "|---|---:|",
    ]
    for key, value in sorted(metrics.items()):
        lines.append(f"| {key} | {_pct(float(value))} |")

    latency = summary.get("latency_ms")
    if latency:
        lines += [
            "",
            "## Latency",
            "",
            f"- Mean: **{latency['mean']:.1f} ms**",
            f"- P50: **{latency['p50']:.1f} ms**",
            f"- P95: **{latency['p95']:.1f} ms**",
        ]

    lines += [
        "",
        "## Interpretation guardrail",
        "",
        "These metrics quantify retrieval behavior for the supplied ground truth and run artifacts. "
        "They should not be presented as a universal quality score or legal/compliance certification.",
        "",
        "## Next layer",
        "",
        "Generation quality (faithfulness, answer relevance, citation correctness and hallucination) "
        "is intentionally separated from retrieval quality and will be evaluated with its own evidence.",
        "",
    ]
    return "\n".join(lines)
