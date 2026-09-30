from __future__ import annotations


def compare_summaries(baseline: dict, candidate: dict, tolerance: float = 1e-12) -> dict:
    base_metrics = baseline.get("metrics", {})
    cand_metrics = candidate.get("metrics", {})
    common = sorted(set(base_metrics) & set(cand_metrics))

    metrics: dict[str, dict[str, float | str]] = {}
    regressions: list[str] = []
    improvements: list[str] = []

    for name in common:
        b = float(base_metrics[name])
        c = float(cand_metrics[name])
        delta = c - b
        status = "unchanged"
        if delta < -tolerance:
            status = "regression"
            regressions.append(name)
        elif delta > tolerance:
            status = "improvement"
            improvements.append(name)
        metrics[name] = {
            "baseline": b,
            "candidate": c,
            "delta": delta,
            "status": status,
        }

    return {
        "metrics": metrics,
        "regressions": regressions,
        "improvements": improvements,
        "has_regression": bool(regressions),
    }
