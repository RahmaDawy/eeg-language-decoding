import numpy as np
import pytest

from eegdecode.evaluation.metrics import (
    bootstrap_ci, candidate_index, cosine_matrix, ranks_from_similarity, retrieval_metrics,
)


def test_perfect_retrieval():
    r = retrieval_metrics(np.eye(5), np.arange(5))
    assert r["top1"] == 1.0 and r["mrr"] == 1.0 and r["recall@5"] == 1.0
    assert r["chance_top1"] == pytest.approx(0.2)


def test_worst_case_rank():
    sim = np.array([[1.0, 2.0, 3.0]])
    assert ranks_from_similarity(sim, [0]).tolist() == [3.0]
    r = retrieval_metrics(sim, [0], ks=(1, 2))
    assert r["top1"] == 0.0 and r["recall@2"] == 0.0 and r["mrr"] == pytest.approx(1 / 3)


def test_ties_are_not_rewarded():
    sim = np.ones((1, 4))
    assert ranks_from_similarity(sim, [2]).tolist() == [2.5]
    assert retrieval_metrics(sim, [2])["top1"] == 0.0


def test_chance_mrr_is_harmonic_mean():
    r = retrieval_metrics(np.eye(4), np.arange(4))
    assert r["chance_mrr"] == pytest.approx((1 + 1 / 2 + 1 / 3 + 1 / 4) / 4)


def test_random_scores_are_near_chance():
    rng = np.random.default_rng(0)
    sim = rng.standard_normal((4000, 20))
    r = retrieval_metrics(sim, rng.integers(0, 20, 4000))
    assert abs(r["top1"] - 0.05) < 0.02
    assert abs(r["mrr"] - r["chance_mrr"]) < 0.03


def test_cosine_matrix_identity():
    a = np.array([[1.0, 0.0], [0.0, 2.0]])
    assert np.allclose(cosine_matrix(a, a), np.eye(2))


def test_candidate_index_maps_to_sorted_unique():
    uniq, idx = candidate_index(["b", "a", "b", "c"])
    assert uniq == ["a", "b", "c"] and idx.tolist() == [1, 0, 1, 2]


def test_out_of_range_index_raises():
    with pytest.raises(ValueError):
        ranks_from_similarity(np.eye(3), [0, 1, 5])


def test_bootstrap_ci_contains_mean():
    m, lo, hi = bootstrap_ci([0, 1, 1, 0, 1, 1, 1, 0], seed=1)
    assert lo <= m <= hi