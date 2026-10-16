"""Cross-dataset transfer (source -> target) with a SOURCE-ONLY decoder.

The ridge decoder and its scaler are fitted on SOURCE rows only; alpha is chosen on two held-out SOURCE
subjects; no target labels are used for fitting or tuning. Target rows are only test queries.
  unseen_sentences : target test restricted to sentences NOT in the source training set (primary)
  all_sentences    : all target windows (includes ~100 sentences shared between the datasets)
--subject-norm z-scores each subject's features with that subject's own EEG (label-free, but TRANSDUCTIVE).
"""
import argparse

import numpy as np
import pandas as pd

from run_experiments import ALPHAS, KEEP, OUT, load, subject_standardize
from eegdecode.data.splits import sentence_key
from eegdecode.decoding.alignment import run_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random-seeds", type=int, default=5)
    ap.add_argument("--subject-norm", action="store_true")
    a = ap.parse_args()

    w, s, E, X = load()
    s["key"] = s.text.map(sentence_key)
    emb = {}
    for k, row in zip(s.key, range(len(s))):
        emb.setdefault(k, E[row])
    ok = (w.aligned & w.valid & np.isfinite(X).all(axis=1)).to_numpy()
    w["key"] = None
    w.loc[w.aligned, "key"] = s.key.values[w.loc[w.aligned, "sent_row"]]

    rows = []
    for src, tgt in [("zuco1", "zuco2"), ("zuco2", "zuco1")]:
        S = w[(w.dataset == src) & ok]
        T = w[(w.dataset == tgt) & ok]
        XS, XT = X[S.index.values], X[T.index.values]
        if a.subject_norm:
            XS = subject_standardize(XS, S.subject.to_numpy())
            XT = subject_standardize(XT, T.subject.to_numpy())
        subs = sorted(S.subject.unique())
        val_subs = [subs[i] for i in np.random.RandomState(a.seed).permutation(len(subs))[:2]]
        src_split = np.where(S.subject.isin(val_subs), "val", "train")
        train_keys = set(S.key[src_split == "train"])
        print(f"\n{src} -> {tgt}: source windows {len(S)} (val subjects {val_subs}), target windows {len(T)}")
        for subset in ("unseen_sentences", "all_sentences"):
            keep = (~T.key.isin(train_keys)).to_numpy() if subset == "unseen_sentences" else np.ones(len(T), bool)
            Tk = T[keep]
            ids = np.concatenate([S.key.to_numpy(), Tk.key.to_numpy()])
            split = np.concatenate([src_split, np.full(len(Tk), "test")])
            dur = np.log(np.concatenate([S.duration_s.to_numpy(), Tk.duration_s.to_numpy()]))[:, None]
            feats = np.vstack([XS, XT[keep]])
            conds = {
                "eeg_bandpower": [feats],
                "duration_only": [dur],
                "random_input": [np.random.default_rng(1000 + r).standard_normal(feats.shape)
                                 for r in range(a.random_seeds)],
            }
            for name, variants in conds.items():
                res = [run_experiment(v, ids, emb, split, ALPHAS) for v in variants]
                row = dict(source=src, target=tgt, target_subset=subset, condition=name,
                           n_target_queries=len(Tk), n_candidates=res[0]["test"]["n_candidates"])
                for k in KEEP:
                    row[k] = float(np.mean([r["test"][k] for r in res]))
                rows.append(row)

    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "_subjnorm" if a.subject_norm else ""
    res.to_csv(OUT / f"results_transfer{tag}.csv", index=False)
    print(res.round(4).to_string(index=False))
    print(f"\nsaved {OUT / f'results_transfer{tag}.csv'}")


if __name__ == "__main__":
    main()