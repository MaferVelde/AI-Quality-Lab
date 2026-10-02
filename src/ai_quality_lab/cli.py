from __future__ import annotations

import json
from pathlib import Path

import typer

from ai_quality_lab import __version__
from ai_quality_lab.analysis.failures import analyze_failures
from ai_quality_lab.analysis.root_causes import analyze_root_causes
from ai_quality_lab.comparison import compare_summaries
from ai_quality_lab.adapters.nuevamente import adapt_ground_truth as adapt_nuevamente_ground_truth
from ai_quality_lab.adapters.nuevamente_retrieval import adapt_retrieval_json as adapt_nuevamente_retrieval_json
from ai_quality_lab.evaluation.retrieval import evaluate_retrieval
from ai_quality_lab.reporting.failure_report import build_failure_report
from ai_quality_lab.reporting.root_cause_report import build_root_cause_report
from ai_quality_lab.reporting.report import build_markdown_report
from ai_quality_lab.versioning.manifest import build_manifest

app = typer.Typer(no_args_is_help=True, help=f"AI Quality Lab v{__version__}")


def _parse_ks(raw: str) -> tuple[int, ...]:
    values = tuple(sorted({int(x.strip()) for x in raw.split(",") if x.strip()}))
    if not values or any(k <= 0 for k in values):
        raise typer.BadParameter("ks must be positive integers, e.g. 3,5")
    return values


@app.command()
def evaluate(
    ground_truth: Path = typer.Option(..., exists=True, readable=True),
    retrieval: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(...),
    ks: str = typer.Option("3,5", help="Comma-separated K values"),
) -> None:
    parsed_ks = _parse_ks(ks)
    output.mkdir(parents=True, exist_ok=True)

    per_query, summary = evaluate_retrieval(ground_truth, retrieval, parsed_ks)
    per_query.to_csv(output / "per_query_metrics.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = build_manifest(
        ground_truth_path=ground_truth,
        retrieval_path=retrieval,
        ks=parsed_ks,
        tool_version=__version__,
    )
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "report.md").write_text(build_markdown_report(summary), encoding="utf-8")

    typer.echo(f"Evaluation complete: {output}")


@app.command("adapt-nuevamente")
def adapt_nuevamente(
    source: Path = typer.Option(..., exists=True, readable=True, help="Frozen NuevaMente ground-truth CSV"),
    chunks: Path = typer.Option(..., exists=True, readable=True, help="Frozen NuevaMente chunks CSV"),
    output: Path = typer.Option(..., help="Canonical AIQL ground-truth CSV to create"),
) -> None:
    summary = adapt_nuevamente_ground_truth(source, chunks, output)
    typer.echo(
        f"Adaptation complete: {output} "
        f"({summary['queries']} queries, {summary['chunks']} chunks, "
        f"{summary['multi_relevant_queries']} multi-relevant queries)"
    )


@app.command("adapt-nuevamente-retrieval")
def adapt_nuevamente_retrieval(
    source: Path = typer.Option(..., exists=True, readable=True, help="Frozen NuevaMente retrieval JSON"),
    chunks: Path = typer.Option(..., exists=True, readable=True, help="Frozen NuevaMente chunks CSV"),
    output: Path = typer.Option(..., help="Canonical AIQL retrieval CSV to create"),
) -> None:
    summary = adapt_nuevamente_retrieval_json(source, chunks, output)
    typer.echo(
        f"Retrieval adaptation complete: {output} "
        f"({summary['queries']} queries, {summary['rows']} rows, "
        f"top_k={summary['top_k']}, score_type={summary['score_types']})"
    )


@app.command("analyze-failures")
def analyze_failure_patterns(
    ground_truth: Path = typer.Option(..., exists=True, readable=True),
    retrieval: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(...),
    k: int = typer.Option(5, min=1, help="Top-K cutoff to diagnose"),
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    analysis, summary = analyze_failures(ground_truth, retrieval, k=k)
    analysis.to_csv(output / "failure_analysis.csv", index=False)
    (output / "failure_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "failure_report.md").write_text(
        build_failure_report(summary), encoding="utf-8"
    )
    typer.echo(
        f"Failure analysis complete: {output} "
        f"({summary['hard_failures']} hard failures, {summary['warnings']} warnings)"
    )


@app.command("analyze-root-causes")
def analyze_root_cause_candidates(
    ground_truth: Path = typer.Option(..., exists=True, readable=True),
    retrieval: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(...),
    k: int = typer.Option(5, min=1, help="Top-K cutoff to diagnose"),
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    analysis, summary = analyze_root_causes(ground_truth, retrieval, k=k)
    analysis.to_csv(output / "root_cause_analysis.csv", index=False)
    (output / "root_cause_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "root_cause_report.md").write_text(
        build_root_cause_report(summary), encoding="utf-8"
    )
    typer.echo(
        f"Root cause analysis complete: {output} "
        f"({summary['diagnosed_cases']} diagnosed hard failures)"
    )


@app.command()
def compare(
    baseline: Path = typer.Option(..., exists=True, readable=True),
    candidate: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(...),
) -> None:
    base = json.loads(baseline.read_text(encoding="utf-8"))
    cand = json.loads(candidate.read_text(encoding="utf-8"))
    result = compare_summaries(base, cand)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    typer.echo(f"Comparison complete: {output}")


if __name__ == "__main__":
    app()
