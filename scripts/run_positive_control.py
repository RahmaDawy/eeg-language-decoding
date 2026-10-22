"""Positive control: do the authors' per-word EEG band features carry ANY word-level information?
Targets: word length (characters) and log first-fixation duration. Metric: Pearson r on held-out rows.
Settings
  cross_subject         train on other subjects; test on unseen subjects AND unseen sentences (as in run_words.py)
  cross_subject, eeg_subjnorm   same, but each subject's features z-scored with that subject's own EEG
                        (label-free, but TRANSDUCTIVE: it uses unlabeled test-subject EEG)
  within_subject        for each subject: train on its train sentences, test on its unseen test sentences
Control: random features through the same pipeline. Ridge alpha is chosen on validation rows only.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from eegdecode.data.splits import assign_split, check_no_leakage, split_items, subject_folds
from run_words import load, safe_r

OUT = Path("experiments/results/metrics")
ALPHAS = (100.0, 1e3, 1e4, 1e5)


def subject_z(X, subj):
    X = X.copy()
    for s in np.unique(subj):
        m = subj == s
        X[m] = (X[m] - X[m].mean(axis=0)) / (X[m].std(axis=0) + 1e-8)
    return X


def fit_pred(Xtr, ytr, Xva, yva, Xte):
    sc = StandardScaler().fit(Xtr)  # statistics from training rows only
    Ztr, Zva, Zte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    best, best_r = None, -np.inf
    for al in ALPHAS:
        m = Ridge(alpha=al).fit(Ztr, ytr)
        r = safe_r(m.predict(Zva), yva)
        if r > best_r:
            best, best_r = m, r
    return best.predict(Zte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    d, X = load()
    subj, keys = d.subject.to_numpy(), d.key.to_numpy()
    T = {"nchar": d.nchar.to_numpy(float), "log_ffd": np.log(d.ffd.to_numpy())}
    feats = {"eeg": X, "eeg_subjnorm": subject_z(X, subj),
             "random": np.random.default_rng(0).standard_normal(X.shape, dtype=np.float32)}
    subs = sorted(d.subject.unique())
    n_hold = max(2, int(0.15 * d.key.nunique()))
    sent_split = split_items(d.key, n_hold, n_hold, a.seed)
    print(f"{len(d)} word rows, {len(subs)} subjects, {d.key.nunique()} sentences")

    rows = []
    for fi, fold in enumerate(subject_folds(subs, min(a.folds, len(subs)), a.seed)):
        split = assign_split(d, fold, sent_split, "subject", "key")
        check_no_leakage(d, split, "subject", "key")  # raises if subjects or sentences are shared
        sp = split.to_numpy()
        tr, va, te = (np.where(sp == nm)[0] for nm in ("train", "val", "test"))
        for fn, F in feats.items():
            for tname, y in T.items():
                p = fit_pred(F[tr], y[tr], F[va], y[va], F[te])
                rows.append(dict(setting="cross_subject", features=fn, target=tname, unit=f"fold{fi}", r=safe_r(p, y[te])))
        print(f"  cross-subject fold {fi} done", flush=True)

    sets = {nm: np.isin(keys, sent_split[nm]) for nm in ("train", "val", "test")}
    for s in subs:
        ms = subj == s
        tr, va, te = (np.where(ms & sets[nm])[0] for nm in ("train", "val", "test"))
        for fn in ("eeg", "random"):
            for tname, y in T.items():
                p = fit_pred(feats[fn][tr], y[tr], feats[fn][va], y[va], feats[fn][te])
                rows.append(dict(setting="within_subject", features=fn, target=tname, unit=s, r=safe_r(p, y[te])))
    print("  within-subject done", flush=True)

    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "results_positive_control.csv", index=False)
    g = res.groupby(["setting", "features", "target"], sort=False).r
    print("\nPEARSON r ON HELD-OUT ROWS (0 = no information; random features should be about 0)")
    print(pd.DataFrame({"mean_r": g.mean(), "std": g.std(), "n_units": g.size()}).round(4).to_string())
    print(f"\nsaved {OUT / 'results_positive_control.csv'}")


if __name__ == "__main__":
    main()