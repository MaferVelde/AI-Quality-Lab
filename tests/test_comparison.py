from ai_quality_lab.comparison import compare_summaries


def test_compare_detects_regression():
    baseline = {"metrics": {"recall@5": 0.80, "mrr": 0.70}}
    candidate = {"metrics": {"recall@5": 0.75, "mrr": 0.72}}
    result = compare_summaries(baseline, candidate)
    assert "recall@5" in result["regressions"]
    assert "mrr" in result["improvements"]
    assert result["has_regression"] is True
