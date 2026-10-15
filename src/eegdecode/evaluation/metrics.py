"""Retrieval metrics for EEG-to-sentence decoding.

Protocol: every query (one EEG window) is scored against ONE candidate per UNIQUE sentence in the
evaluated split. Chance level for top-1 is therefore 1 / n_candidates.
Exact retrieval (top-1, recall@K, MRR) asks "is the right sentence ranked first?".
sim_margin (cosine only) asks "is the right sentence more similar than the others on average?",
which can improve even when exact retrieval is still near chance.
"""
import numpy as np


def l2_normalize(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float64)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def cosine_matrix(queries, candidates):
    return l2_normalize(queries) @ l2_normalize(candidates).T


def candidate_index(sentence_ids):
    """Unique sorted candidate ids and, for each query, the index of its true candidate."""
    uniq = sorted(set(sentence_ids))
    pos = {s: i for i, s in enumerate(uniq)}
    return uniq, np.array([pos[s] for s in sentence_ids], dtype=int)


def ranks_from_similarity(sim, true_idx):
    """Rank (1 = best) of the true candidate. Ties share the average rank, so constant scores earn no credit."""
    sim = np.asarray(sim, dtype=np.float64)
    true_idx = np.asarray(true_idx, dtype=int)
    if sim.ndim != 2 or sim.shape[0] != true_idx.shape[0]:
        raise ValueError(f"sim {sim.shape} and true_idx {true_idx.shape} do not match")
    if true_idx.min() < 0 or true_idx.max() >= sim.shape[1]:
        raise ValueError("true_idx out of range")
    true_scores = sim[np.arange(sim.shape[0]), true_idx][:, None]
    greater = (sim > true_scores).sum(axis=1)
    equal = (sim == true_scores).sum(axis=1) - 1  # exclude the true candidate itself
    return 1.0 + greater + equal / 2.0


def retrieval_metrics(sim, true_idx, ks=(1, 5, 10)):
    sim = np.asarray(sim, dtype=np.float64)
    ranks = ranks_from_similarity(sim, true_idx)
    n_q, n_c = sim.shape
    out = {
        "n_queries": int(n_q),
        "n_candidates": int(n_c),
        "top1": float(np.mean(ranks <= 1)),
        "chance_top1": 1.0 / n_c,
        "mrr": float(np.mean(1.0 / ranks)),
        "chance_mrr": float(np.mean(1.0 / np.arange(1, n_c + 1))),
        "mean_rank": float(ranks.mean()),
        "median_rank": float(np.median(ranks)),
    }
    for k in ks:
        out[f"recall@{k}"] = float(np.mean(ranks <= k))
        out[f"chance_recall@{k}"] = min(k, n_c) / n_c
    true_scores = sim[np.arange(n_q), np.asarray(true_idx, dtype=int)]
    others = (sim.sum(axis=1) - true_scores) / max(n_c - 1, 1)
    out["mean_true_sim"] = float(true_scores.mean())
    out["mean_other_sim"] = float(others.mean())
    out["sim_margin"] = out["mean_true_sim"] - out["mean_other_sim"]
    return out


def bootstrap_ci(values, n_boot=1000, seed=0, alpha=0.05):
    """CI of the mean over queries. Queries from one subject/sentence are correlated, so this is optimistic."""
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)