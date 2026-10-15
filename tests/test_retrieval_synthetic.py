import numpy as np
import pandas as pd

from eegdecode.data.splits import assign_split, check_no_leakage, split_items
from eegdecode.decoding.alignment import run_experiment

ALPHAS = (0.1, 1.0, 10.0)


def make_world(n_sub=6, n_sent=60, d=16, n_feat=40, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n_sent, d))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    W = rng.standard_normal((d, n_feat))
    rows, X = [], []
    for s in range(n_sub):
        for t in range(n_sent):
            rows.append(dict(subject=f"S{s}", sentence_id=f"T{t:03d}"))
            X.append(emb[t] @ W + noise * rng.standard_normal(n_feat))
    return pd.DataFrame(rows), np.array(X), {f"T{t:03d}": emb[t] for t in range(n_sent)}


def _split(df):
    subj = split_items(df.subject, 1, 2, seed=0)
    sent = split_items(df.sentence_id, 8, 10, seed=0)
    split = assign_split(df, subj, sent)
    check_no_leakage(df, split)
    return split.to_numpy()


def test_planted_signal_is_retrieved_above_chance():
    df, X, emb = make_world()
    res = run_experiment(X, df.sentence_id.to_numpy(), emb, _split(df), alphas=ALPHAS)
    assert res["alpha"] in ALPHAS
    assert res["test"]["n_candidates"] == 10 and res["test"]["n_queries"] == 20
    assert res["test"]["top1"] > 0.6  # chance = 0.1
    assert res["test"]["mrr"] > res["test"]["chance_mrr"] + 0.3


def test_noise_input_control_is_near_chance():
    df, X, emb = make_world()
    X_noise = np.random.default_rng(123).standard_normal(X.shape)
    res = run_experiment(X_noise, df.sentence_id.to_numpy(), emb, _split(df), alphas=ALPHAS)
    assert res["test"]["top1"] < 0.4  # chance = 0.1