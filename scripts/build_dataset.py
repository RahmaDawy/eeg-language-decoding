r"""Day 1: align EEG windows to sentences, extract band-power features, embed sentences.

Run from the project root:   python scripts\build_dataset.py
Writes to data\processed\ : windows.csv, features.npy, sentences.csv, sent_emb.npy
and reports\alignment_check.csv.  Safe to re-run: per-recording results are cached.

Alignment rules (verified on your files):
  * ZuCo 1: relations_labels_task2.csv has the 300 sentences in presentation order.
            Blocks 1..6 hold 49,50,50,50,50,51 sentences -> offsets by block.
  * ZuCo 2: nr_k.csv has 3 extra rows at the start; EEG windows = rows [3:].
A recording whose window count does not match its block size is NOT aligned (reported, skipped).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

from eegdecode.data.load_data import channel_keep_idx, list_recordings, load_recording, qc_table
from eegdecode.decoding.text_encoder import embed_texts

OUT = Path("data/processed")
CACHE = OUT / "cache"
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
Z1_BLOCK_SIZES = {1: 49, 2: 50, 3: 50, 4: 50, 5: 50, 6: 51}
Z1_OFFSETS = {k: sum(v for b, v in Z1_BLOCK_SIZES.items() if b < k) for k in Z1_BLOCK_SIZES}
Z2_SKIP = 3


# ----------------------------------------------------------------- text
def zuco1_texts():
    df = pd.read_csv("data/raw/zuco1/task_materials/relations_labels_task2.csv", encoding="latin-1")
    return [str(s).strip() for s in df["sentence"]]


def zuco2_texts(block):
    t = pd.read_csv(f"data/raw/zuco2/task_materials/nr_{block}.csv", sep=";", header=None,
                    encoding="latin-1").dropna(subset=[2])
    return [str(s).strip() for s in t[2]]


def n_words(s):
    return len(str(s).split())


# ----------------------------------------------------------------- features
def band_power(x, fs):
    """x: (T, C) -> (C, n_bands) log10 band power from Welch PSD."""
    f, p = welch(x.T.astype(np.float64), fs=fs, nperseg=min(x.shape[0], 1000), axis=-1)
    out = np.empty((x.shape[1], len(BANDS)), np.float32)
    for j, (lo, hi) in enumerate(BANDS.values()):
        m = (f >= lo) & (f < hi)
        out[:, j] = np.log10(p[:, m].mean(axis=1) + 1e-12)
    return out


def process_recording(item, dataset):
    cache = CACHE / f"{dataset}_{item['subject']}_{item['block']}.npz"
    if not cache.exists():
        rec = load_recording(item["path"], dataset)
        keep = channel_keep_idx(rec.channels)
        names = np.array([str(rec.channels[i]) for i in keep])
        qc = qc_table(rec)
        feats = np.zeros((len(rec.windows), len(keep), len(BANDS)), np.float32)
        rms = np.zeros(len(rec.windows), np.float32)
        for i, w in enumerate(rec.windows):
            if qc.valid[i]:
                x = w[:, keep]
                feats[i] = band_power(x, rec.srate)
                rms[i] = np.log10(np.sqrt((x.astype(np.float64) ** 2).mean()) + 1e-12)
        np.savez(cache, feats=feats, names=names, duration=qc.duration_s.values.astype(np.float32),
                 valid=qc.valid.values, rms=rms, prefix=np.array(rec.prefix), srate=np.array(rec.srate))
    return dict(np.load(cache))


def corr(a, b):
    if len(a) < 10 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


# ----------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    # sentence table
    z1 = zuco1_texts()
    rows = [dict(sent_id=f"z1_{i:03d}", dataset="zuco1", block=0, text=t) for i, t in enumerate(z1)]
    z2_raw, z2_ids = {}, {}
    for k in range(1, 8):
        z2_raw[k] = zuco2_texts(k)
        z2_ids[k] = [f"z2_b{k}_{j:02d}" for j in range(len(z2_raw[k]) - Z2_SKIP)]
        for sid, t in zip(z2_ids[k], z2_raw[k][Z2_SKIP:]):
            rows.append(dict(sent_id=sid, dataset="zuco2", block=k, text=t))
    sents = pd.DataFrame(rows)
    sents["n_words"] = sents.text.map(n_words)
    wc = dict(zip(sents.sent_id, sents.n_words))
    print(f"sentences: {len(sents)} (zuco1 {sum(sents.dataset == 'zuco1')}, zuco2 {sum(sents.dataset == 'zuco2')})")

    # recordings -> features (cached)
    recs = []
    for ds, root in (("zuco1", "data/raw/zuco1"), ("zuco2", "data/raw/zuco2")):
        items = [it for it in list_recordings(root) if "NR" in it["path"].name]
        for k, it in enumerate(items, 1):
            print(f"[{ds}] {k}/{len(items)} {it['path'].name}", flush=True)
            try:
                recs.append((ds, it, process_recording(it, ds)))
            except Exception as e:  # keep going, report at the end
                print("   FAILED:", e)

    common = None
    for _, _, d in recs:
        names = list(d["names"])
        common = names if common is None else [c for c in common if c in set(names)]
    print(f"common channels: {len(common)}")

    # alignment
    meta, feats, skipped, check = [], [], [], []
    for ds, it, d in recs:
        n, k = len(d["duration"]), it["block"]
        if ds == "zuco1":
            ids = [f"z1_{Z1_OFFSETS[k] + i:03d}" for i in range(n)] if Z1_BLOCK_SIZES.get(k) == n else None
        else:
            ids = z2_ids.get(k) if len(z2_ids.get(k, [])) == n else None
        if ids is None:
            skipped.append(f"{ds} {it['subject']} block {k}: {n} windows")
            ids = [""] * n
        names = list(d["names"])
        idx = [names.index(c) for c in common]
        feats.append(d["feats"][:, idx, :].reshape(n, -1))
        for i in range(n):
            meta.append(dict(dataset=ds, subject=it["subject"], block=k, prefix=str(d["prefix"]), win_idx=i,
                             sent_id=ids[i], aligned=bool(ids[i]), duration_s=float(d["duration"][i]),
                             valid=bool(d["valid"][i]), rms=float(d["rms"][i]),
                             n_words=wc.get(ids[i], np.nan)))
        v = d["valid"]
        if ids[0]:
            check.append(dict(dataset=ds, subject=it["subject"], block=k, n=n,
                              corr_dur_words=corr(d["duration"][v], np.array([wc[s] for s in ids])[v])))
    meta = pd.DataFrame(meta)
    X = np.concatenate(feats).astype(np.float32)

    # per-subject z-scoring (uses only that subject's own unlabeled windows)
    for (ds, sub), g in meta.groupby(["dataset", "subject"]):
        ix = g.index.values
        ok = ix[g.valid.values]
        mu, sd = X[ok].mean(0), X[ok].std(0) + 1e-6
        X[ix] = (X[ix] - mu) / sd

    # sentence embeddings
    print("embedding sentences ...")
    E = embed_texts(list(sents.text)).astype(np.float32)

    meta.to_csv(OUT / "windows.csv", index=False)
    np.save(OUT / "features.npy", X)
    sents.to_csv(OUT / "sentences.csv", index=False)
    np.save(OUT / "sent_emb.npy", E)
    chk = pd.DataFrame(check)
    chk.to_csv("reports/alignment_check.csv", index=False)

    # ---- report
    print("\n=== SUMMARY ===")
    print(f"features {X.shape}, windows {len(meta)}, aligned {int(meta.aligned.sum())}, "
          f"aligned+valid {int((meta.aligned & meta.valid).sum())}")
    print(meta[meta.aligned & meta.valid].groupby("dataset").agg(
        windows=("win_idx", "size"), subjects=("subject", "nunique")))
    print("\nrecordings NOT aligned (window count != block size):", len(skipped))
    for s in skipped:
        print("  ", s)
    print("\nduration-vs-word-count correlation per dataset (should be clearly positive, ~0.5+):")
    print(chk.groupby("dataset").corr_dur_words.describe()[["count", "mean", "min", "max"]])

    # ZuCo2 offset sanity check: is skipping 3 rows the best choice?
    print("\nZuCo2 offset check: mean corr(duration, word count) if the first o CSV rows are dropped")
    res = {}
    for ds, it, d in recs:
        if ds != "zuco2":
            continue
        n, k = len(d["duration"]), it["block"]
        w = np.array([n_words(s) for s in z2_raw[k]])
        extra = len(w) - n
        if 0 <= extra <= 6:
            for o in range(extra + 1):
                res.setdefault(o, []).append(corr(d["duration"][d["valid"]], w[o:o + n][d["valid"]]))
    for o in sorted(res):
        print(f"  skip {o}: {np.nanmean(res[o]):.2f}  (over {len(res[o])} recordings)")
    print("\nDONE. Next: python scripts\\run_baseline.py")


if __name__ == "__main__":
    main()