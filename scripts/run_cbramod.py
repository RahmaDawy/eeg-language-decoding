"""Frozen pretrained CBraMod features vs the same network randomly initialised vs band power.
Same folds / decoder / controls as run_final.py (unseen subjects AND unseen sentences, leakage checked per fold).

extract : per ZuCo 1.0 'Matlab files' subject and sentence: drop the flat reference channel (-> 104), resample 500 -> 200 Hz,
          up to 3 non-overlapping 2 s chunks from the sentence start, z-score each chunk per channel (ASSUMPTION: the unit
          scaling used in pretraining is not verified). FROZEN pretrained model and an identically shaped RANDOMLY
          INITIALISED copy; feature = [mean, std over channels] of the patch-averaged encoder output (400 values);
          sentence feature = mean over its chunks. Saved to data/processed/cbramod/<SUBJ>.npz.
eval    : ridge decoder (alpha on validation subjects) + controls. '+subjnorm' variants z-score each subject's features
          with that subject's own EEG (label-free but TRANSDUCTIVE).
Settings are fixed in advance; one run.
"""
import argparse
import copy
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import io as sio
from scipy.signal import resample_poly

from eegdecode.data.splits import assign_split, check_no_leakage, sentence_key, split_items, subject_folds
from eegdecode.decoding.text_encoder import embed_texts
from run_experiments import subject_standardize
from run_final import average, candidates, fit_eval, record

GT, CB = Path("data/processed/gt"), Path("data/processed/cbramod")
OUT = Path("experiments/results/metrics")
REPO = "braindecode/cbramod-pretrained"
CHUNK, MAX_CHUNKS = 400, 3  # 2 s at 200 Hz


def load_pretrained():
    from braindecode.models import CBraMod
    return CBraMod.from_pretrained(REPO, n_chans=104, n_times=CHUNK, sfreq=200, return_encoder_output=True).eval()


def reinit(model, seed=0):
    torch.manual_seed(seed)
    for m in model.modules():
        if m is model:
            continue
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
        elif isinstance(m, torch.nn.MultiheadAttention):
            m._reset_parameters()
    return model.eval()


def pooled(model, x, bs=32):
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            y = model(x[i:i + bs])  # (B, 104, 2 patches, 200)
            z = y.mean(dim=2)  # average the two 1 s patches -> (B, 104, 200)
            outs.append(torch.cat([z.mean(1), z.std(1)], dim=1))
    return torch.cat(outs).numpy()


def extract(files, force):
    CB.mkdir(parents=True, exist_ok=True)
    pre = load_pretrained()
    rnd = reinit(copy.deepcopy(pre))
    drop = None
    for p in files:
        sub = Path(p).stem.replace("results", "").replace("_NR", "")
        dest = CB / f"{sub}.npz"
        if dest.exists() and not force:
            print(f"{sub}: exists, skipping")
            continue
        print(f"{sub}: loading ...", flush=True)
        sd = sio.loadmat(p, struct_as_record=False, squeeze_me=True, variable_names=["sentenceData"])["sentenceData"]
        texts, chunks, owner = [], [], []
        for s in np.atleast_1d(sd):
            try:
                raw = np.asarray(s.rawData, dtype=np.float64)
            except AttributeError:
                continue
            if raw.ndim != 2 or raw.shape[0] != 105 or raw.shape[1] < 500 or not np.isfinite(raw).all():
                continue
            if drop is None:
                flat = np.where(raw.std(axis=1) == 0)[0]
                if len(flat) != 1:
                    raise SystemExit(f"expected exactly one flat channel, found {flat.tolist()}")
                drop = int(flat[0])
                print("flat reference channel index:", drop)
            elif raw[drop].std() > 1e-12:
                raise SystemExit(f"channel {drop} is not flat in {sub}")
            x = resample_poly(np.delete(raw, drop, axis=0), 2, 5, axis=1)  # 500 -> 200 Hz
            n = min(MAX_CHUNKS, x.shape[1] // CHUNK)
            for c in range(n):
                seg = x[:, c * CHUNK:(c + 1) * CHUNK]
                seg = (seg - seg.mean(1, keepdims=True)) / (seg.std(1, keepdims=True) + 1e-6)
                chunks.append(seg.astype(np.float32))
                owner.append(len(texts))
            if n > 0:
                texts.append(str(s.content))
        X = torch.from_numpy(np.stack(chunks))
        print(f"  {sub}: {len(texts)} sentences, {len(chunks)} chunks; running the models ...", flush=True)
        fp, fr = pooled(pre, X), pooled(rnd, X)
        own = np.array(owner)

        def sent_mean(F):
            df = pd.DataFrame(F)
            df["o"] = own
            return df.groupby("o").mean().sort_index().to_numpy().astype(np.float32)

        np.savez_compressed(dest, text=np.array(texts), pre=sent_mean(fp), rnd=sent_mean(fr), drop=drop)
        print(f"  saved {dest}", flush=True)


def evaluate(a):
    recs = []
    for f in sorted(CB.glob("*.npz")):
        sub, c, g = f.stem, np.load(f), np.load(GT / f"{f.stem}.npz")
        gk = {sentence_key(str(t)): i for i, t in enumerate(g["text"])}
        for j, t in enumerate(c["text"]):
            k = sentence_key(str(t))
            if k in gk:
                i = gk[k]
                recs.append((sub, k, str(t), c["pre"][j], c["rnd"][j], g["ours"][i], g["official"][i],
                             g["nsamp"][i], g["nwords"][i]))
    if len({r[0] for r in recs}) < 3:
        raise SystemExit("need at least 3 extracted subjects in data/processed/cbramod")
    subj = np.array([r[0] for r in recs])
    keys = np.array([r[1] for r in recs])
    P, R = np.stack([r[3] for r in recs]), np.stack([r[4] for r in recs])
    O, OFF = np.stack([r[5] for r in recs]), np.stack([r[6] for r in recs])
    L = np.column_stack([np.log([r[7] for r in recs]), np.log(np.clip([r[8] for r in recs], 1, None))])
    first = {}
    for r in recs:
        first.setdefault(r[1], r[2])
    ukeys = list(first)
    emb = dict(zip(ukeys, embed_texts([first[k] for k in ukeys], cache_path="data/processed/cbramod_sent_emb")))

    d = pd.DataFrame({"subject": subj, "key": keys})
    subs = sorted(d.subject.unique())
    n_hold = max(2, int(0.15 * d.key.nunique()))
    sent_split = split_items(d.key, n_hold, n_hold, a.seed)
    folds = subject_folds(subs, min(a.folds, len(subs)), a.seed)
    print(f"{len(d)} sentence rows, {len(subs)} subjects, {d.key.nunique()} unique sentences, "
          f"val/test sentences held out: {n_hold}/{n_hold}")
    feats = {
        "length_only": L,
        "official_bandpower": OFF,
        "our_bandpower": O,
        "our_bandpower+subjnorm": subject_standardize(O, subj),
        "cbramod_pretrained": P,
        "cbramod_pretrained+subjnorm": subject_standardize(P, subj),
        "cbramod_random_init": R,
        "cbramod_random_init+subjnorm": subject_standardize(R, subj),
    }
    rows = []
    for fi, fold in enumerate(folds):
        split = assign_split(d, fold, sent_split, "subject", "key")
        counts = check_no_leakage(d, split, "subject", "key")  # raises if subjects or sentences are shared
        sp, ids = split.to_numpy(), keys
        tr, va, te = (np.where(sp == nm)[0] for nm in ("train", "val", "test"))
        Ytr = np.stack([emb[x] for x in ids[tr]])
        cva, tva = candidates(ids[va], emb)
        cte, tte = candidates(ids[te], emb)
        base = dict(fold=fi, test_subjects=",".join(fold["test"]))
        print(f"fold {fi}: test subjects {fold['test']}, rows train/val/test "
              f"{counts['train']}/{counts['val']}/{counts['test']}, test candidates {len(cte)}", flush=True)
        for name, X in feats.items():
            record(rows, {**base, "method": name},
                   fit_eval(X[tr], Ytr, X[va], cva, tva, {"clean": (X[te], cte, tte)}))
        rnd = []
        for r in range(a.random_seeds):
            Z = np.random.default_rng(1000 + r).standard_normal((len(d), 400))
            rnd.append(fit_eval(Z[tr], Ytr, Z[va], cva, tva, {"clean": (Z[te], cte, tte)}))
        record(rows, {**base, "method": "random_input"}, average(rnd))

    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "results_cbramod.csv", index=False)
    g = res.groupby("method", sort=False)
    print("\nMEAN OVER FOLDS (unseen subjects AND unseen sentences; clean EEG)")
    print(g[["n_candidates", "top1", "chance_top1", "recall@10", "mrr", "chance_mrr", "sim_margin"]].mean().round(4).to_string())
    print("\nSTD OF sim_margin ACROSS FOLDS")
    print(g["sim_margin"].std().round(4).to_string())
    print(f"\nsaved {OUT / 'results_cbramod.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["extract", "eval", "all"])
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random-seeds", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="extract only the first N subjects (smoke test)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.mode in ("extract", "all"):
        files = sorted(glob.glob("data/raw/zuco1/task2 - NR/Matlab files/results*_NR.mat"))
        extract(files[:a.limit] if a.limit else files, a.force)
    if a.mode in ("eval", "all"):
        evaluate(a)


if __name__ == "__main__":
    main()