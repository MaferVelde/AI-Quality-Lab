from ai_quality_lab.evaluation.metrics import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_metrics_basic():
    relevant = {"a", "c"}
    retrieved = ["x", "a", "c", "y"]
    assert recall_at_k(relevant, retrieved, 3) == 1.0
    assert precision_at_k(relevant, retrieved, 3) == 2 / 3
    assert hit_rate_at_k(relevant, retrieved, 1) == 0.0
    assert hit_rate_at_k(relevant, retrieved, 2) == 1.0
    assert reciprocal_rank(relevant, retrieved) == 0.5
    assert 0.0 < ndcg_at_k(relevant, retrieved, 3) <= 1.0
