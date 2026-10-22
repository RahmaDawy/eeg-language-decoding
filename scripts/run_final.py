"""Final experiments on the ZuCo 1.0 'Matlab files' caches (true sentence labels; every subject in data/processed/gt).
Test = unseen subjects AND unseen sentences (leave-subjects-out folds). Candidates = unique test sentences.
Representations
  official_bandpower   authors' sentence-level band means (840)
  our_bandpower        our log band power from the whole sentence
  chunk_bandpower      log band power of 1.5 s chunks (mean over chunks); test EEG corrupted at 4 noise levels
  ssl_trained / untrained_encoder   small contrastive 1D-CNN on the same chunks; trained WITHOUT labels on TRAIN-subject,
                       TRAIN-sentence chunks only
Controls     length_only (reading-length shortcut), random_input
Quality-aware  ridge sample weights 1/(1+max(0,z)), z = standardised artifact score (training rows only)
Decoder (identical for all): standardise (train only) -> ridge (alpha chosen on validation subjects) -> cosine ranking.
Noise is added to TEST EEG only: x + level * std(channel) * N(0,1), level in {0, 0.5, 1, 2}.
"""
import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from eegdecode.data.splits import assign_split, check_no_leakage, sentence_key, split_items, subject_folds
from eegdecode.decoding.text_encoder import embed_texts
from eegdecode.evaluation.metrics import candidate_index, cosine_matrix, retrieval_metrics
from eegdecode.models.baseline import band_power_features

GT = Path("data/processed/gt")
OUT = Path("experiments/results/metrics")
FIG = Path("experiments/results/figures")
LEVELS = ["clean", "mild", "moderate", "strong"]
NOISE = {"clean": 0.0, "mild": 0.5, "moderate": 1.0, "strong": 2.0}
ALPHAS = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)
KEEP = ["top1", "chance_top1", "recall@5", "chance_recall@5", "recall@10", "mrr", "chance_mrr",
        "median_rank", "sim_margin", "n_candidates"]


# ------------------------------------------------------------------ data
def load_all():
    files = sorted(GT.glob("*.npz"))
    if len(files) < 3:
        raise SystemExit("need at least 3 cached subjects in data/processed/gt (run scripts/make_cache.py)")
    D = {k: [] for k in ("subj", "text", "off", "ours", "nsamp", "nwords", "art", "chunks", "cwin")}
    base = 0
    for f in files:
        z = np.load(f)
        n = len(z["text"])
        D["subj"] += [f.stem] * n
        D["text"] += z["text"].tolist()
        for src, dst in (("official", "off"), ("ours", "ours"), ("nsamp", "nsamp"), ("nwords", "nwords"),
                         ("artifact", "art")):
            D[dst].append(z[src])
        D["chunks"].append(z["chunks"])
        D["cwin"].append(z["chunk_sent"] + base)
        base += n
    D["subj"] = np.array(D["subj"])
    for k in ("off", "ours", "nsamp", "nwords", "art", "chunks", "cwin"):
        D[k] = np.concatenate(D[k])
    return D


def chunk_bandpower(chunks, level, seed):
    rng = np.random.default_rng(seed)
    out = []
    for c in chunks:
        x = np.asarray(c, dtype=np.float64).T  # (150, 105)
        if level > 0:
            x = x + level * x.std(axis=0, keepdims=True) * rng.standard_normal(x.shape)
        out.append(band_power_features(x, 100.0))
    return np.vstack(out)


def window_mean(Fm, cwin, n):
    out = np.full((n, Fm.shape[1]), np.nan)
    df = pd.DataFrame(Fm)
    df["w"] = cwin
    m = df.groupby("w").mean()
    out[m.index.to_numpy()] = m.to_numpy()
    return out


def candidates(keys, emb):
    uniq, true = candidate_index(list(keys))
    return np.stack([emb[u] for u in uniq]), true


# ------------------------------------------------------------------ decoder
def fit_eval(Xtr, Ytr, Xva, cva, tva, tests, w=None):
    sc = StandardScaler().fit(Xtr)
    Ztr, Zva = sc.transform(Xtr), sc.transform(Xva)
    best, best_mrr = None, -np.inf
    for al in ALPHAS:
        m = Ridge(alpha=al).fit(Ztr, Ytr, sample_weight=w)
        mrr = retrieval_metrics(cosine_matrix(m.predict(Zva), cva), tva)["mrr"]
        if mrr > best_mrr:
            best, best_mrr = m, mrr
    return {nm: retrieval_metrics(cosine_matrix(best.predict(sc.transform(X)), c), t)
            for nm, (X, c, t) in tests.items()}


def average(res_list):
    return {lv: {k: float(np.mean([r[lv][k] for r in res_list])) for k in KEEP} for lv in res_list[0]}


def record(rows, base, res):
    for lv, r in res.items():
        rows.append({**base, "noise": lv, **{k: r[k] for k in KEEP}})


# ------------------------------------------------------------------ self-supervised encoder
class Encoder(nn.Module):
    def __init__(self, c=105, d=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c, 64, 7, 2, 3), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, 5, 2, 2), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, d, 3, 2, 1), nn.BatchNorm1d(d), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten())

    def forward(self, x):
        return self.net(x)


def augment(x):
    x = x + 0.1 * x.std(-1, keepdim=True) * torch.randn_like(x)  # small Gaussian noise
    x = x * (0.9 + 0.2 * torch.rand(x.shape[0], 1, 1, device=x.device))  # mild amplitude scaling
    return torch.roll(x, int(torch.randint(-10, 11, (1,))), dims=-1)  # small shift (+-0.1 s)


def nt_xent(z1, z2, t=0.2):
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    n = z1.shape[0]
    sim = (z @ z.T / t).masked_fill(torch.eye(2 * n, dtype=torch.bool, device=z.device), -1e9)
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


def train_ssl(enc, head, CH, cidx, epochs, bs, device):
    X = torch.from_numpy(CH[cidx].astype(np.float32))
    mu = X.mean(dim=(0, 2), keepdim=True).to(device)  # statistics from TRAINING chunks only
    sd = (X.std(dim=(0, 2), keepdim=True) + 1e-6).to(device)
    bs = min(bs, len(X))
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=1e-4)
    enc.train(); head.train()
    for ep in range(epochs):
        perm, tot, nb = torch.randperm(len(X)), 0.0, 0
        for i in range(0, len(X) - bs + 1, bs):
            xb = X[perm[i:i + bs]].to(device)
            loss = nt_xent(head(enc((augment(xb) - mu) / sd)), head(enc((augment(xb) - mu) / sd)))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"      epoch {ep + 1}/{epochs}  contrastive loss {tot / max(nb, 1):.4f}", flush=True)
    enc.eval(); head.eval()
    return mu, sd


def embed_windows(enc, mu, sd, CH, cw, rows, level, seed, device):
    cidx = np.where(np.isin(cw, rows))[0]
    g = torch.Generator().manual_seed(seed)
    outs = []
    with torch.no_grad():
        for i in range(0, len(cidx), 512):
            x = torch.from_numpy(CH[cidx[i:i + 512]].astype(np.float32))
            if level > 0:
                x = x + level * x.std(-1, keepdim=True) * torch.randn(x.shape, generator=g)
            outs.append(enc((x.to(device) - mu) / sd).cpu())
    Z = pd.DataFrame(torch.cat(outs).numpy())
    Z["w"] = cw[cidx]
    return Z.groupby("w").mean().loc[rows].to_numpy()


# ------------------------------------------------------------------ figure
def make_figure(res, name):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing: run  pip install matplotlib  then rerun")
        return
    FIG.mkdir(parents=True, exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    clean = res[res.noise == "clean"].groupby("method", sort=False).sim_margin.agg(["mean", "std"])
    a1.bar(range(len(clean)), clean["mean"], yerr=clean["std"].fillna(0), capsize=3)
    a1.set_xticks(range(len(clean))); a1.set_xticklabels(clean.index, rotation=40, ha="right", fontsize=8)
    a1.axhline(0, color="k", ls=":", lw=1)
    a1.set_ylabel("sim_margin (0 = chance)"); a1.set_title("Clean EEG, unseen subjects + unseen sentences")
    for m in ["chunk_bandpower", "chunk_bandpower+quality", "ssl_trained", "ssl_trained+quality", "untrained_encoder"]:
        g = res[res.method == m]
        if g.empty:
            continue
        mm = g.groupby("noise").sim_margin.mean().reindex(LEVELS)
        ss = g.groupby("noise").sim_margin.std().reindex(LEVELS).fillna(0)
        a2.errorbar(range(len(LEVELS)), mm.values, yerr=ss.values, marker="o", capsize=3, label=m)
    rnd = res[res.method == "random_input"].sim_margin.mean()
    a2.axhline(rnd, color="gray", ls="--", lw=1, label="random input")
    a2.axhline(0, color="k", ls=":", lw=1)
    a2.set_xticks(range(len(LEVELS))); a2.set_xticklabels(LEVELS)
    a2.set_xlabel("noise added to test EEG"); a2.set_ylabel("sim_margin"); a2.legend(fontsize=7)
    a2.set_title("Robustness to noise")
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=150)
    print(f"saved {FIG / (name + '.png')}")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--max-chunks", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--random-seeds", type=int, default=5)
    ap.add_argument("--pilot", action="store_true", help="1 fold, 1 epoch, few chunks: smoke test only")
    a = ap.parse_args()
    if a.pilot:
        a.epochs, a.max_chunks = 1, 1500
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    D = load_all()
    n = len(D["subj"])
    keys = np.array([sentence_key(t) for t in D["text"]])
    first = {}
    for k, t in zip(keys, D["text"]):
        first.setdefault(k, t)
    ukeys = list(first)
    emb = dict(zip(ukeys, embed_texts([first[k] for k in ukeys], cache_path="data/processed/gt_sent_emb_final")))

    print("computing chunk band power at 4 noise levels (about a minute) ...", flush=True)
    CH, cwin = D["chunks"], D["cwin"]
    CB = {lv: window_mean(chunk_bandpower(CH, NOISE[lv], 7 + i), cwin, n) for i, lv in enumerate(LEVELS)}
    ok = np.isfinite(CB["clean"]).all(axis=1) & np.isfinite(CB["strong"]).all(axis=1)
    d = pd.DataFrame({"subject": D["subj"], "key": keys}).loc[ok]
    idx = d.index.to_numpy()
    ids = d.key.to_numpy()
    subs = sorted(d.subject.unique())
    k = min(a.folds, len(subs))
    n_hold = max(2, int(0.15 * d.key.nunique()))
    sent_split = split_items(d.key, n_hold, n_hold, a.seed)
    folds = subject_folds(subs, k, a.seed)
    folds = folds[:1] if a.pilot else folds
    print(f"{len(d)} windows, {len(subs)} subjects, {d.key.nunique()} unique sentences, "
          f"{len(folds)} fold(s), val/test sentences held out: {n_hold}/{n_hold}")

    static = {
        "official_bandpower": D["off"],
        "our_bandpower": D["ours"],
        "length_only": np.column_stack([np.log(D["nsamp"]), np.log(np.clip(D["nwords"], 1, None))]),
    }
    rows = []
    for fi, fold in enumerate(folds):
        split = assign_split(d, fold, sent_split, "subject", "key")
        counts = check_no_leakage(d, split, "subject", "key")  # raises if subjects or sentences are shared
        sp = split.to_numpy()
        tr, va, te = (np.where(sp == nm)[0] for nm in ("train", "val", "test"))
        Ytr = np.stack([emb[x] for x in ids[tr]])
        cva, tva = candidates(ids[va], emb)
        cte, tte = candidates(ids[te], emb)
        base = dict(fold=fi, test_subjects=",".join(fold["test"]))
        z = (D["art"][idx][tr] - D["art"][idx][tr].mean()) / (D["art"][idx][tr].std() + 1e-9)
        wq = 1.0 / (1.0 + np.clip(z, 0, None))
        print(f"\nfold {fi}: test subjects {fold['test']}, windows train/val/test "
              f"{counts['train']}/{counts['val']}/{counts['test']}, test candidates {len(cte)}", flush=True)

        for name, Xall in static.items():
            X = Xall[idx]
            record(rows, {**base, "method": name},
                   fit_eval(X[tr], Ytr, X[va], cva, tva, {"clean": (X[te], cte, tte)}))
        rnd = []
        for r in range(a.random_seeds):
            R = np.random.default_rng(1000 + r).standard_normal((len(d), 840))
            rnd.append(fit_eval(R[tr], Ytr, R[va], cva, tva, {"clean": (R[te], cte, tte)}))
        record(rows, {**base, "method": "random_input"}, average(rnd))

        Xc = CB["clean"][idx]
        tests = {lv: (CB[lv][idx][te], cte, tte) for lv in LEVELS}
        record(rows, {**base, "method": "chunk_bandpower"}, fit_eval(Xc[tr], Ytr, Xc[va], cva, tva, tests))
        record(rows, {**base, "method": "chunk_bandpower+quality"},
               fit_eval(Xc[tr], Ytr, Xc[va], cva, tva, tests, w=wq))

        torch.manual_seed(a.seed + fi)
        enc = Encoder().to(device)
        head = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 64)).to(device)
        enc0 = copy.deepcopy(enc).eval()
        cidx = np.where(np.isin(cwin, idx[tr]))[0]  # chunks of TRAIN subjects' TRAIN sentences only
        if len(cidx) > a.max_chunks:
            cidx = np.sort(np.random.default_rng(a.seed + fi).choice(cidx, a.max_chunks, replace=False))
        print(f"    training the encoder on {len(cidx)} chunks (no labels)", flush=True)
        mu, sd = train_ssl(enc, head, CH, cidx, a.epochs, a.batch, device)
        for method, model in (("ssl_trained", enc), ("untrained_encoder", enc0)):
            Ztr = embed_windows(model, mu, sd, CH, cwin, idx[tr], 0.0, 1, device)
            Zva = embed_windows(model, mu, sd, CH, cwin, idx[va], 0.0, 1, device)
            tz = {lv: (embed_windows(model, mu, sd, CH, cwin, idx[te], NOISE[lv], 100 + i, device), cte, tte)
                  for i, lv in enumerate(LEVELS)}
            record(rows, {**base, "method": method}, fit_eval(Ztr, Ytr, Zva, cva, tva, tz))
            if method == "ssl_trained":
                record(rows, {**base, "method": "ssl_trained+quality"},
                       fit_eval(Ztr, Ytr, Zva, cva, tva, tz, w=wq))

    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    name = "results_final_pilot" if a.pilot else "results_final"
    res.to_csv(OUT / f"{name}.csv", index=False)
    print("\nMEAN OVER FOLDS (test = unseen subjects AND unseen sentences; noise only on test EEG)")
    g = res.groupby(["method", "noise"], sort=False)
    print(g[["n_candidates", "top1", "chance_top1", "recall@10", "mrr", "chance_mrr", "sim_margin"]].mean().round(4).to_string())
    print("\nSTD ACROSS FOLDS (sim_margin)")
    print(g["sim_margin"].std().round(4).to_string())
    print(f"\nsaved {OUT / (name + '.csv')}")
    make_figure(res, "final_results_pilot" if a.pilot else "final_results")


if __name__ == "__main__":
    main()