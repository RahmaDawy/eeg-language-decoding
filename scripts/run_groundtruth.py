"""Ground-truth test on the ZuCo 1.0 'Matlab files': the true sentence is given for every EEG row.

Features
  official_bandpower  the authors' sentence-level band means (8 bands x 105 channels)
  our_bandpower       our log band power computed from the sentence's rawData
Controls: length_only (log samples, log words = reading-time/length shortcut), random_input (5 seeds).
Regimes
  double        test subject unseen AND test sentences unseen (val sentences from the training subjects)
  subject_only  test subject unseen; test sentences may appear in training; alpha chosen on another training subject
Candidates = unique sentences of the test rows; chance = 1 / n_candidates.
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import io as sio

from eegdecode.data.splits import sentence_key
from eegdecode.decoding.alignment import DEFAULT_ALPHAS, run_experiment
from eegdecode.decoding.text_encoder import embed_texts
from eegdecode.models.baseline import band_power_features

BANDS = ["t1", "t2", "a1", "a2", "b1", "b2", "g1", "g2"]
KEEP = ["top1", "chance_top1", "recall@5", "chance_recall@5", "recall@10", "mrr", "chance_mrr",
        "median_rank", "sim_margin"]
OUT = Path("experiments/results/metrics")


def load_subject(path):
    sd = sio.loadmat(path, struct_as_record=False, squeeze_me=True, variable_names=["sentenceData"])["sentenceData"]
    texts, rows = [], []
    for s in np.atleast_1d(sd):
        texts.append(str(s.content))
        try:
            raw = np.asarray(s.rawData, dtype=np.float64)  # (105 channels, samples)
            if raw.ndim != 2 or raw.shape[0] != 105 or raw.shape[1] < 500 or not np.isfinite(raw).all():
                rows.append(None)
                continue
            official = np.concatenate([np.asarray(getattr(s, f"mean_{b}"), dtype=np.float64).ravel() for b in BANDS])
            if official.size != 8 * 105 or not np.isfinite(official).all():
                rows.append(None)
                continue
            rows.append(dict(text=str(s.content), official=official, ours=band_power_features(raw.T, 500.0),
                             nsamp=raw.shape[1], nwords=int(np.size(s.word))))
        except AttributeError:
            rows.append(None)
    return texts, rows


def make_splits(subj, keys, test_sub, others, regime, seed):
    n = len(subj)
    if regime == "double":
        uniq = sorted(set(keys))
        perm = [uniq[i] for i in np.random.RandomState(seed).permutation(len(uniq))]
        n_h = max(2, int(0.15 * len(uniq)))
        test_s, val_s = perm[:n_h], perm[n_h:2 * n_h]
        sp = np.full(n, "drop", dtype=object)
        sp[np.isin(subj, others) & ~np.isin(keys, test_s + val_s)] = "train"
        sp[np.isin(subj, others) & np.isin(keys, val_s)] = "val"
        sp[(subj == test_sub) & np.isin(keys, test_s)] = "test"
        return [sp]
    out = []
    for v in others:  # validation subject = one of the training subjects; train on the rest
        sp = np.full(n, "drop", dtype=object)
        sp[np.isin(subj, [s for s in others if s != v])] = "train"
        sp[subj == v] = "val"
        sp[subj == test_sub] = "test"
        out.append(sp)
    return out


def assert_clean(sp, subj, keys, regime):
    tr, va, te = (sp == nm for nm in ("train", "val", "test"))
    assert not (set(subj[te]) & (set(subj[tr]) | set(subj[va]))), "test subject leaked"
    if regime == "double":
        assert not (set(keys[te]) & (set(keys[tr]) | set(keys[va]))), "test sentence leaked"
        assert not (set(keys[va]) & set(keys[tr])), "val sentence leaked"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+",
                    default=sorted(glob.glob("data/raw/zuco1/task2 - NR/Matlab files/results*_NR.mat")))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--random-seeds", type=int, default=5)
    a = ap.parse_args()
    if len(a.files) < 3:
        raise SystemExit("need at least 3 subject files")

    frames, all_texts = [], {}
    for p in a.files:
        name = Path(p).stem.replace("results", "").replace("_NR", "")
        print(f"loading {name} (about a minute) ...", flush=True)
        texts, rows = load_subject(p)
        ok = [r for r in rows if r is not None]
        print(f"  {name}: {len(texts)} sentences, {len(ok)} usable rows", flush=True)
        all_texts[name] = texts
        frames += [dict(subject=name, **r) for r in ok]
    first = next(iter(all_texts.values()))
    print("sentence order identical across subjects:", all(t == first for t in all_texts.values()))

    d = pd.DataFrame(frames)
    d["key"] = d.text.map(sentence_key)
    subj, keys = d.subject.to_numpy(), d.key.to_numpy()
    XA, XB = np.vstack(d.official.to_list()), np.vstack(d.ours.to_list())
    L = np.column_stack([np.log(d.nsamp.to_numpy()), np.log(d.nwords.clip(lower=1).to_numpy())])
    uniq = d.drop_duplicates("key")
    emb = dict(zip(uniq.key, embed_texts(uniq.text.tolist(), cache_path="data/processed/gt_sent_emb")))
    feats = {
        "official_bandpower": [XA],
        "our_bandpower": [XB],
        "length_only": [L],
        "random_input": [np.random.default_rng(1000 + r).standard_normal(XA.shape) for r in range(a.random_seeds)],
    }
    subs = sorted(d.subject.unique())
    print(f"\n{len(d)} rows, {len(subs)} subjects, {len(uniq)} unique sentences")

    rows = []
    for regime in ("double", "subject_only"):
        for t in subs:
            others = [s for s in subs if s != t]
            for seed in (range(a.seeds) if regime == "double" else [0]):
                splits = make_splits(subj, keys, t, others, regime, seed)
                for sp in splits:
                    assert_clean(sp, subj, keys, regime)
                for name, variants in feats.items():
                    res = [run_experiment(v, keys, emb, sp, DEFAULT_ALPHAS)["test"] for v in variants for sp in splits]
                    row = dict(regime=regime, test_subject=t, seed=seed, condition=name,
                               n_test=int((splits[0] == "test").sum()), n_candidates=res[0]["n_candidates"])
                    for k in KEEP:
                        row[k] = float(np.mean([r[k] for r in res]))
                    rows.append(row)
        print(f"  regime {regime} done", flush=True)

    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "results_groundtruth.csv", index=False)
    print("\nMEAN OVER TEST SUBJECTS x SPLITS (true sentence labels, 3 subjects)")
    print(res.groupby(["regime", "condition"], sort=False)[["n_test", "n_candidates"] + KEEP].mean().round(4).to_string())
    print("\nSTD ACROSS RUNS")
    print(res.groupby(["regime", "condition"], sort=False)[["top1", "sim_margin"]].std().round(4).to_string())
    print(f"\nsaved {OUT / 'results_groundtruth.csv'}")


if __name__ == "__main__":
    main()