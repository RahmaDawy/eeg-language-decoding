"""Day 1: band-power baseline, EEG -> sentence retrieval, with controls.

Run:  python scripts\run_baseline.py            (both datasets)
      python scripts\run_baseline.py --dataset zuco1

Design (strict): every test fold uses subjects AND sentences never seen in training.
  EEG features --ridge--> sentence embedding (MiniLM) --cosine--> rank among the test sentences.
Conditions: band power (real), noise input, duration only, shuffled labels.
Chance level is computed exactly from the number of candidate sentences in each fold.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def ridge_fit(X, Y, alpha_rel):
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    A = Xc.T @ Xc
    alpha = alpha_rel * np.trace(A) / A.shape[0]
    W = np.linalg.solve(A + alpha * np.eye(A.shape[0]), Xc.T @ Yc)
    return W, mx, my


def unit(a):
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)


def retrieval_metrics(pred, cand, true_idx):
    sims = unit(pred) @ unit(cand).T
    true_sim = sims[np.arange(len(sims)), true_idx][:, None]
    rank = 1 + (sims > true_sim).sum(1)
    m = cand.shape[0]
    return dict(top1=(rank == 1).mean(), top5=(rank <= 5).mean(), mrr=(1.0 / rank).mean(),
                chance_top1=1 / m, chance_top5=min(5, m) / m, chance_mrr=(1.0 / np.arange(1, m + 1)).mean(),
                n_test=len(rank), n_cand=m)


def fold_ids(keys, K, rng):
    u = np.array(sorted(set(keys)))
    rng.shuffle(u)
    return {k: i % K for i, k in enumerate(u)}


def run_dataset(ds, root, K, alpha_rel, seed):
    w = pd.read_csv(root / "windows.csv")
    X = np.load(root / "features.npy")
    S = pd.read_csv(root / "sentences.csv")
    E = np.load(root / "sent_emb.npy")
    sid2row = {s: i for i, s in enumerate(S.sent_id)}
    m = ((w.dataset == ds) & w.aligned & w.valid).values
    w, X = w[m].reset_index(drop=True), X[m]
    Y = E[[sid2row[s] for s in w.sent_id]]
    K = min(K, w.subject.nunique())
    rng = np.random.default_rng(seed)
    sf = w.subject.map(fold_ids(w.subject, K, rng)).values
    tf = w.sent_id.map(fold_ids(w.sent_id, K, rng)).values

    ld = np.log(w.duration_s.values)
    D = np.c_[ld, ld ** 2]
    for (_, g) in w.groupby("subject"):  # per-subject z-score of the duration feature
        ix = g.index.values
        D[ix] = (D[ix] - D[ix].mean(0)) / (D[ix].std(0) + 1e-6)
    conds = {"band power (real EEG)": (X, False),
             "noise input": (rng.normal(size=X.shape).astype(np.float32), False),
             "duration only": (D, False),
             "shuffled labels": (X, True)}

    rows = []
    for f in range(K):
        te = (sf == f) & (tf == f)
        tr = (sf != f) & (tf != f)
        if te.sum() == 0 or tr.sum() == 0:
            continue
        cand_ids = sorted(set(w.sent_id[te]))
        cand_row = {s: i for i, s in enumerate(cand_ids)}
        cand = E[[sid2row[s] for s in cand_ids]]
        true_idx = np.array([cand_row[s] for s in w.sent_id[te]])
        for name, (F, shuffle) in conds.items():
            Ytr = Y[tr][rng.permutation(tr.sum())] if shuffle else Y[tr]
            W, mx, my = ridge_fit(F[tr], Ytr, alpha_rel)
            pred = (F[te] - mx) @ W
            r = retrieval_metrics(pred, cand - my, true_idx)
            r.update(condition=name, fold=f, n_train=int(tr.sum()))
            rows.append(r)
    res = pd.DataFrame(rows)
    summ = res.groupby("condition", sort=False)[["top1", "top5", "mrr", "chance_top1", "chance_top5",
                                                 "chance_mrr", "n_test", "n_cand"]].mean()
    summ["top1_std"] = res.groupby("condition", sort=False).top1.std()
    summ["top5_std"] = res.groupby("condition", sort=False).top5.std()
    print(f"\n=== {ds}: {len(w)} windows, {w.subject.nunique()} subjects, {w.sent_id.nunique()} sentences, "
          f"{K} folds (unseen subjects AND unseen sentences) ===")
    print(summ.round(3).to_string())
    res["dataset"] = ds
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["zuco1", "zuco2", "both"], default="both")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alpha-rel", type=float, default=5.0,
                    help="ridge strength relative to mean eigenvalue; one fixed value for ALL conditions")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = []
    for ds in (["zuco1", "zuco2"] if a.dataset == "both" else [a.dataset]):
        out.append(run_dataset(ds, Path(a.processed), a.folds, a.alpha_rel, a.seed))
    Path("reports").mkdir(exist_ok=True)
    pd.concat(out).to_csv("reports/baseline_results.csv", index=False)
    print("\nSaved reports/baseline_results.csv")