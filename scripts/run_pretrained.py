"""Frozen pretrained EEG encoder (CBraMod, braindecode/cbramod-pretrained) as a third representation.

Same protocol as run_robust_ssl.py: 4 subject folds, unseen subjects AND unseen sentences, same usable windows,
same ridge decoder (alpha chosen on validation subjects' clean data), cosine retrieval over the unique test sentences.

Input adapter (assumptions, documented in the report):
  chunks.npy (100 Hz, 2.4 s) -> noise added at 100 Hz (test conditions) -> upsampled x2 to 200 Hz -> first 400 samples
  (2 patches of 1 s) -> whole chunk scaled to global SD 0.5 (ZuCo units are unverified; CBraMod expects uV/100).
Features: encoder output averaged over patches, then mean and SD across channels (400 values); a window embedding is
the mean over its chunks. The encoder is frozen; nothing is trained except the ridge decoder.
Encoder outputs are cached in data/processed/cbramod_{frozen,untrained}_{level}.npy.

Outputs: experiments/results/metrics/results_pretrained.csv           (per fold)
         experiments/results/metrics/results_pretrained_subjects.csv  (per test subject, clean EEG)
Run  python scripts\run_pretrained.py --pilot   first (loads the model, prints shapes and timing, then stops).
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from eegdecode.data.splits import assign_split, check_no_leakage, split_items, subject_folds
from eegdecode.evaluation.metrics import cosine_matrix, retrieval_metrics
from run_robust_ssl import ALPHAS, KEEP, LEVELS, NOISE, OUT, P, candidates, load_all, mean_metrics, record

HUB = "braindecode/cbramod-pretrained"
N_CH, N_TIMES, TARGET_STD = 104, 400, 0.5


# ---------------------------------------------------------------- model
def load_model(untrained, seed=0):
    from braindecode.models import CBraMod
    attempts = [dict(n_chans=N_CH, n_times=N_TIMES, n_outputs=4, return_encoder_output=True),
                dict(return_encoder_output=True)]
    last = None
    for kw in attempts:
        try:
            if untrained:
                torch.manual_seed(seed)
                m = CBraMod(**kw)
            else:
                m = CBraMod.from_pretrained(HUB, **kw)
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
            n = sum(p.numel() for p in m.parameters())
            print(f"loaded CBraMod ({'random weights' if untrained else HUB}) with {kw}; parameters {n / 1e6:.2f}M", flush=True)
            return m
        except Exception as e:  # try the next argument set, report the last error if all fail
            last = e
            print(f"  load attempt with {kw} failed: {type(e).__name__}: {e}")
    raise SystemExit(f"could not load CBraMod: {last}")


def pool(out):
    if out.dim() == 4:  # (batch, channels, patches, features)
        out = out.mean(2)
    if out.dim() != 3:
        return out.reshape(out.shape[0], -1)
    return torch.cat([out.mean(1), out.std(1)], dim=1)  # (batch, 2 * features)


def prepare(x_np, level, gen):
    x = torch.from_numpy(np.nan_to_num(x_np.astype(np.float32)))
    if level > 0:
        x = x + level * x.std(-1, keepdim=True) * torch.randn(x.shape, generator=gen)
    x = torch.from_numpy(np.ascontiguousarray(resample_poly(x.numpy(), 2, 1, axis=-1)[..., :N_TIMES]).astype(np.float32))
    return x * (TARGET_STD / (x.std(dim=(1, 2), keepdim=True) + 1e-6))


def encode(model, CH, idx, level, seed, bs, label=""):
    gen = torch.Generator().manual_seed(seed)
    outs, t0 = [], time.time()
    with torch.no_grad():
        for b, i in enumerate(range(0, len(idx), bs)):
            x = prepare(CH[idx[i:i + bs]], level, gen)
            o = model(x)
            if b == 0:
                print(f"    {label}: input {tuple(x.shape)} -> raw encoder output {tuple(o.shape)} -> pooled {tuple(pool(o).shape)}", flush=True)
            outs.append(pool(o).cpu())
            if b % 20 == 0 and b > 0:
                done = i + bs
                print(f"    {label}: {done}/{len(idx)} chunks, {(time.time() - t0) / 60:.1f} min elapsed, "
                      f"about {(time.time() - t0) / done * (len(idx) - done) / 60:.1f} min left", flush=True)
    return torch.cat(outs).numpy().astype(np.float32)


def cached(tag, level, model_fn, CH, n_chunks, seed, bs):
    f = P / f"cbramod_{tag}_{level}.npy"
    if f.exists():
        print(f"  using cache {f}")
        return np.load(f)
    Z = encode(model_fn(), CH, np.arange(n_chunks), NOISE[level], seed, bs, label=f"{tag}/{level}")
    np.save(f, Z)
    return Z


def window_features(Zc, cw, n_windows):
    g = pd.DataFrame(Zc).assign(w=cw).groupby("w").mean()
    W = np.full((n_windows, Zc.shape[1]), np.nan, dtype=np.float32)
    W[g.index.to_numpy()] = g.to_numpy()
    return W


# ---------------------------------------------------------------- decoder
def fit_model(Xtr, Ytr, Xva, cva, tva, w=None):
    sc = StandardScaler().fit(Xtr)
    Ztr, Zva = sc.transform(Xtr), sc.transform(Xva)
    best, best_mrr = None, -np.inf
    for al in ALPHAS:
        m = Ridge(alpha=al).fit(Ztr, Ytr, sample_weight=w)
        mrr = retrieval_metrics(cosine_matrix(m.predict(Zva), cva), tva)["mrr"]
        if mrr > best_mrr:
            best, best_mrr = m, mrr
    return lambda X: best.predict(sc.transform(X))


def evaluate(pred, tests):
    return {name: retrieval_metrics(cosine_matrix(pred(X), c), t) for name, (X, c, t) in tests.items()}


def subject_rows(pred, X, subj, cte, tte, base, method):
    Pm, rows = pred(X), []
    for s in sorted(set(subj)):
        m = subj == s
        r = retrieval_metrics(cosine_matrix(Pm[m], cte), tte[m])
        rows.append({**base, "subject": s, "method": method, "n_windows": int(m.sum()),
                     **{k: r[k] for k in ("mrr", "chance_mrr", "top1", "chance_top1", "sim_margin")}})
    return rows


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--random-seeds", type=int, default=3)
    ap.add_argument("--pilot", action="store_true", help="load the model, encode 64 chunks, print shapes/timing, stop")
    a = ap.parse_args()

    w, emb, feats, art, CH, cw = load_all()
    n_chunks = len(cw)
    print(f"chunks: {n_chunks}, windows: {len(w)}")

    if a.pilot:
        t0 = time.time()
        m = load_model(False)
        print(f"model loaded in {time.time() - t0:.1f} s")
        idx = np.arange(64)
        for lv in ("clean", "strong"):
            t0 = time.time()
            Z = encode(m, CH, idx, NOISE[lv], 1, a.batch, label=lv)
            dt = (time.time() - t0) / len(idx)
            print(f"  {lv}: {dt * 1000:.0f} ms per chunk, features {Z.shape}, finite {bool(np.isfinite(Z).all())}, "
                  f"mean {Z.mean():.4f}, sd {Z.std():.4f}")
        print(f"estimated full encoding (4 noise levels + 1 untrained control): {dt * n_chunks * 5 / 60:.0f} min")
        m0 = load_model(True)
        Z0 = encode(m0, CH, idx, 0.0, 1, a.batch, label="untrained")
        print(f"  untrained: features {Z0.shape}, finite {bool(np.isfinite(Z0).all())}")
        return

    print("\nencoding all chunks with the frozen pretrained encoder (cached after the first run)")
    frozen = lambda: load_model(False)
    Wf = {lv: window_features(cached("frozen", lv, frozen, CH, n_chunks, 100 + i, a.batch), cw, len(w))
          for i, lv in enumerate(LEVELS)}
    Wu = window_features(cached("untrained", "clean", lambda: load_model(True), CH, n_chunks, 100, a.batch), cw, len(w))

    ok = (w.aligned.to_numpy() & w.valid.to_numpy() & np.isfinite(art)
          & np.all([np.isfinite(feats[lv]).all(axis=1) for lv in LEVELS], axis=0)
          & np.isin(np.arange(len(w)), cw)
          & np.isfinite(Wf["clean"]).all(axis=1))
    print(f"usable windows: {int(ok.sum())}/{len(w)}")

    rows, srows = [], []
    for ds in sorted(w.dataset.unique()):
        d = w[(w.dataset == ds) & ok]
        idx, keys, subj_all = d.index.to_numpy(), d.key.to_numpy(), d.subject.to_numpy()
        subs, n_sent = sorted(d.subject.unique()), d.key.nunique()
        n_hold = max(2, int(0.15 * n_sent))
        sent_split = split_items(d.key, n_val=n_hold, n_test=n_hold, seed=a.seed)
        folds = subject_folds(subs, a.folds, a.seed)
        print(f"\n{ds}: {len(d)} windows, {len(subs)} subjects, {n_sent} sentences")
        for fi, fold in enumerate(folds):
            split = assign_split(d, fold, sent_split, "subject", "key")
            check_no_leakage(d, split, "subject", "key")
            sp = split.to_numpy()
            tr, va, te = (np.where(sp == nm)[0] for nm in ("train", "val", "test"))
            Ytr = np.stack([emb[k] for k in keys[tr]])
            cva, tva = candidates(keys[va], emb)
            cte, tte = candidates(keys[te], emb)
            base = dict(dataset=ds, fold=fi, test_subjects=",".join(fold["test"]))
            print(f"  fold {fi}: test subjects {fold['test']}, test candidates {len(cte)}", flush=True)

            # quality weights: same label-free artifact score as run_robust_ssl.py
            z = (art[idx][tr] - art[idx][tr].mean()) / (art[idx][tr].std() + 1e-9)
            wq = 1.0 / (1.0 + np.clip(z, 0, None))

            # ---- frozen pretrained encoder
            Fc = Wf["clean"][idx]
            tests = {lv: (Wf[lv][idx][te], cte, tte) for lv in LEVELS}
            p_plain = fit_model(Fc[tr], Ytr, Fc[va], cva, tva)
            record(rows, {**base, "representation": "cbramod", "method": "frozen_pretrained"}, evaluate(p_plain, tests))
            p_q = fit_model(Fc[tr], Ytr, Fc[va], cva, tva, w=wq)
            record(rows, {**base, "representation": "cbramod", "method": "frozen_pretrained_quality_weighted"},
                   evaluate(p_q, tests))
            srows += subject_rows(p_plain, tests["clean"][0], subj_all[te], cte, tte, base, "cbramod_frozen")

            # ---- same architecture, random weights (clean only)
            Uc = Wu[idx]
            p_u = fit_model(Uc[tr], Ytr, Uc[va], cva, tva)
            record(rows, {**base, "representation": "cbramod", "method": "untrained_cbramod"},
                   evaluate(p_u, {"clean": (Uc[te], cte, tte)}))
            srows += subject_rows(p_u, Uc[te], subj_all[te], cte, tte, base, "cbramod_untrained")

            # ---- band-power baseline (per-subject numbers for the subject-level analysis)
            Bc = feats["clean"][idx]
            p_b = fit_model(Bc[tr], Ytr, Bc[va], cva, tva)
            srows += subject_rows(p_b, Bc[te], subj_all[te], cte, tte, base, "bandpower")

            # ---- random-input control with the same dimensionality as the CBraMod features
            rnd = []
            for r in range(a.random_seeds):
                R = np.random.default_rng(3000 + r).standard_normal((len(d), Fc.shape[1]))
                p_r = fit_model(R[tr], Ytr, R[va], cva, tva)
                rnd.append(evaluate(p_r, {"clean": (R[te], cte, tte)}))
                if r == 0:
                    srows += subject_rows(p_r, R[te], subj_all[te], cte, tte, base, "random_input")
            record(rows, {**base, "representation": "cbramod", "method": "random_input"}, mean_metrics(rnd))

    res, sres = pd.DataFrame(rows), pd.DataFrame(srows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "results_pretrained.csv", index=False)
    sres.to_csv(OUT / "results_pretrained_subjects.csv", index=False)
    print("\nMEAN OVER FOLDS (unseen subjects AND unseen sentences; noise added to TEST EEG only)")
    g = res.groupby(["dataset", "representation", "method", "noise"], sort=False)
    print(g[["n_candidates", "top1", "chance_top1", "mrr", "chance_mrr", "sim_margin"]].mean().round(4).to_string())
    print("\nPER-SUBJECT (clean EEG): mean MRR minus chance, by method")
    sres["mrr_minus_chance"] = sres.mrr - sres.chance_mrr
    print(sres.groupby(["dataset", "method"])["mrr_minus_chance"].agg(["mean", "std", "count"]).round(4).to_string())
    print(f"\nsaved {OUT / 'results_pretrained.csv'} and {OUT / 'results_pretrained_subjects.csv'}")


if __name__ == "__main__":
    main()