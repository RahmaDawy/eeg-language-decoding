"""Unseen-subject retrieval from EEG band-power features, with controls.

Regimes:
  double        test subjects AND test sentences unseen (main, strict result)
  subject_only  test subjects unseen, test sentences may also appear in training (weaker; diagnostic)
Conditions: eeg_bandpower | duration_only (shortcut control) | random_input (noise control, several seeds).
Candidates = unique sentences of the test split; chance = 1 / n_candidates. sim_margin chance = 0.
--subject-norm z-scores each subject's features using that subject's own EEG only (no labels);
it uses unlabeled test-subject EEG, so it is TRANSDUCTIVE and must be reported as such.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from eegdecode.data.splits import assign_split, check_no_leakage, sentence_key, split_items, subject_folds
from eegdecode.decoding.alignment import run_experiment

P = Path("data/processed")
OUT = Path("experiments/results/metrics")
ALPHAS = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)
KEEP = ["top1", "chance_top1", "recall@5", "chance_recall@5", "recall@10", "mrr", "chance_mrr",
        "median_rank", "sim_margin"]


def load():
    w = pd.read_csv(P / "windows_aligned.csv")
    s = pd.read_csv(P / "sentences.csv")
    E = np.load(P / "sent_emb.npy")
    if len(s) != len(E):
        raise SystemExit(f"sentences.csv ({len(s)}) and sent_emb.npy ({len(E)}) differ")
    X = None
    # rebuild features from the per-recording cache so the row order is guaranteed to match windows
    for (ds, sub, blk), g in w.groupby(["dataset", "subject", "block"]):
        f = np.load(P / "cache" / f"{ds}_{sub}_{blk}.npz")["feats"]
        g = g.sort_values("win_idx")
        if f.shape[0] != len(g):
            raise SystemExit(f"{ds}_{sub}_{blk}: cache has {f.shape[0]} windows, windows.csv has {len(g)}")
        f = f.reshape(len(g), -1)
        if X is None:
            X = np.full((len(w), f.shape[1]), np.nan)
        X[g.index.values] = f
    return w, s, E, X


def subject_standardize(X, subjects):
    X = X.copy()
    for sub in np.unique(subjects):
        m = subjects == sub
        X[m] = (X[m] - X[m].mean(axis=0)) / (X[m].std(axis=0) + 1e-8)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random-seeds", type=int, default=5)
    ap.add_argument("--regime", choices=["double", "subject_only"], default="double")
    ap.add_argument("--subject-norm", action="store_true")
    a = ap.parse_args()

    w, s, E, X = load()
    s["key"] = s.text.map(sentence_key)  # identical sentences share one key (never split across train/test)
    emb = {}
    for k, row in zip(s.key, range(len(s))):
        emb.setdefault(k, E[row])
    ok = w.aligned & w.valid & np.isfinite(X).all(axis=1)
    w["key"] = None
    w.loc[w.aligned, "key"] = s.key.values[w.loc[w.aligned, "sent_row"]]

    rows = []
    for ds in sorted(w.dataset.unique()):
        d = w[(w.dataset == ds) & ok]
        Xd = X[d.index.values]
        if a.subject_norm:
            Xd = subject_standardize(Xd, d.subject.to_numpy())
        dur = np.log(d.duration_s.to_numpy())[:, None]
        subs, n_sent = sorted(d.subject.unique()), d.key.nunique()
        n_hold = max(2, int(0.15 * n_sent))
        sent_split = split_items(d.key, n_val=n_hold, n_test=n_hold, seed=a.seed) if a.regime == "double" else None
        held = f"val/test sentences held out: {n_hold}/{n_hold}" if a.regime == "double" else "sentences NOT held out"
        print(f"\n{ds}: {len(d)} usable windows, {len(subs)} subjects, {n_sent} unique sentences ({held})")
        for fi, fold in enumerate(subject_folds(subs, a.folds, a.seed)):
            split = assign_split(d, fold, sent_split, "subject", "key")
            counts = check_no_leakage(d, split, "subject", "key", sentence_disjoint=(a.regime == "double"))
            sp, ids = split.to_numpy(), d.key.to_numpy()
            conds = {
                "eeg_bandpower": [Xd],
                "duration_only": [dur],
                "random_input": [np.random.default_rng(1000 + r).standard_normal(Xd.shape) for r in range(a.random_seeds)],
            }
            for name, variants in conds.items():
                res = [run_experiment(v, ids, emb, sp, ALPHAS) for v in variants]
                row = dict(dataset=ds, fold=fi, condition=name, test_subjects=",".join(fold["test"]),
                           n_train=counts["train"], n_test=counts["test"],
                           n_candidates=res[0]["test"]["n_candidates"])
                for k in KEEP:
                    row[k] = float(np.mean([r["test"][k] for r in res]))
                rows.append(row)
            print(f"  fold {fi} done (test subjects: {fold['test']}, test queries: {counts['test']})")

    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{a.regime}{'_subjnorm' if a.subject_norm else ''}"
    res.to_csv(OUT / f"results_{tag}.csv", index=False)
    print(f"\nMEAN OVER FOLDS (regime: {a.regime}, subject-norm: {a.subject_norm})")
    print(res.groupby(["dataset", "condition"], sort=False)[["n_candidates"] + KEEP].mean().round(4).to_string())
    print("\nSTD ACROSS FOLDS")
    print(res.groupby(["dataset", "condition"], sort=False)[["top1", "sim_margin"]].std().round(4).to_string())
    print(f"\nsaved {OUT / f'results_{tag}.csv'}")


if __name__ == "__main__":
    main()