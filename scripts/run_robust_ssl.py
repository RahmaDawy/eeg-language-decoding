"""Noise robustness, quality-aware weighting and a self-supervised encoder.
Evaluation: unseen subjects AND unseen sentences (4 subject folds), candidates = unique test sentences.

Representations
  bandpower      band-power features (clean for training; test signals corrupted at 4 noise levels)
  ssl            small 1D-CNN encoder trained WITHOUT labels on training-subject chunks only
                 (SimCLR-style contrastive loss; augmentations: Gaussian noise, amplitude scaling, small time shift);
                 window embedding = mean of its chunk embeddings
Methods: plain | quality_weighted (training rows weighted by 1/(1+max(0,z)) of artifact score z, fitted on training
         rows; no test labels or test performance involved) | untrained_encoder (same net, random weights) | random_input
Decoder: identical for all: standardise (train only) -> ridge (alpha chosen on validation subjects' clean data) -> cosine ranking.
Each fold's trained SSL encoder is saved to experiments/checkpoints/ssl_<dataset>_fold<k>.pt
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
from eegdecode.evaluation.metrics import candidate_index, cosine_matrix, retrieval_metrics

P = Path("data/processed")
OUT = Path("experiments/results/metrics")
FIG = Path("experiments/results/figures")
CKPT = Path("experiments/checkpoints")
LEVELS = ["clean", "mild", "moderate", "strong"]
NOISE = {"clean": 0.0, "mild": 0.5, "moderate": 1.0, "strong": 2.0}
ALPHAS = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)
KEEP = ["top1", "chance_top1", "recall@5", "chance_recall@5", "recall@10", "mrr", "chance_mrr",
        "median_rank", "sim_margin"]


# ---------------------------------------------------------------- data
def load_all():
    w = pd.read_csv(P / "windows_aligned.csv")
    s = pd.read_csv(P / "sentences.csv")
    E = np.load(P / "sent_emb.npy")
    s["key"] = s.text.map(sentence_key)
    emb = {}
    for k, i in zip(s.key, range(len(s))):
        emb.setdefault(k, E[i])
    w["key"] = None
    w.loc[w.aligned, "key"] = s.key.values[w.loc[w.aligned, "sent_row"]]
    feats = {lv: np.load(P / f"feat_{lv}.npy") for lv in LEVELS}
    art = np.load(P / "artifact_frac.npy")
    CH = np.load(P / "chunks.npy", mmap_mode="r")
    cw = np.load(P / "chunk_win.npy")
    return w, emb, feats, art, CH, cw


def candidates(keys, emb):
    uniq, true = candidate_index(list(keys))
    return np.stack([emb[u] for u in uniq]), true


def fit_eval(Xtr, Ytr, Xva, cand_va, true_va, tests, w=None):
    """Fit once on training rows, pick alpha on clean validation data, evaluate on every test condition."""
    sc = StandardScaler().fit(Xtr)
    Ztr, Zva = sc.transform(Xtr), sc.transform(Xva)
    best, best_mrr = None, -np.inf
    for al in ALPHAS:
        m = Ridge(alpha=al).fit(Ztr, Ytr, sample_weight=w)
        mrr = retrieval_metrics(cosine_matrix(m.predict(Zva), cand_va), true_va)["mrr"]
        if mrr > best_mrr:
            best, best_mrr = m, mrr
    return {name: retrieval_metrics(cosine_matrix(best.predict(sc.transform(X)), c), t)
            for name, (X, c, t) in tests.items()}


def mean_metrics(list_of_res):
    return {lv: {k: float(np.mean([r[lv][k] for r in list_of_res])) for k in KEEP + ["n_candidates"]}
            for lv in list_of_res[0]}


def record(rows, base, res):
    for lv, r in res.items():
        rows.append({**base, "noise": lv, "n_candidates": r["n_candidates"], **{k: r[k] for k in KEEP}})


# ---------------------------------------------------------------- self-supervised encoder
class Encoder(nn.Module):
    def __init__(self, c=104, d=128):
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
    return torch.roll(x, int(torch.randint(-10, 11, (1,))), dims=-1)  # small shift (+-0.1 s), one per batch


def nt_xent(z1, z2, t=0.2):
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    n = z1.shape[0]
    sim = (z @ z.T / t).masked_fill(torch.eye(2 * n, dtype=torch.bool, device=z.device), -1e9)
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


def train_ssl(enc, head, CH, cidx, epochs, bs, device):
    X = torch.from_numpy(CH[cidx].astype(np.float32))
    mu = X.mean(dim=(0, 2), keepdim=True).to(device)  # normalisation stats from TRAINING chunks only
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


# ---------------------------------------------------------------- figure
def make_figure(res, name):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed: run  pip install matplotlib  and rerun")
        return
    FIG.mkdir(parents=True, exist_ok=True)
    dss = sorted(res.dataset.unique())
    fig, axes = plt.subplots(1, len(dss), figsize=(6 * len(dss), 4), sharey=True, squeeze=False)
    for ax, ds in zip(axes[0], dss):
        r = res[(res.dataset == ds) & (res.method != "random_input")]
        for (rep, meth), g in r.groupby(["representation", "method"], sort=False):
            m = g.groupby("noise").sim_margin.mean().reindex(LEVELS)
            s = g.groupby("noise").sim_margin.std().reindex(LEVELS)
            ax.errorbar(range(len(LEVELS)), m.values, yerr=s.values, marker="o", capsize=3, label=f"{rep} / {meth}")
        ax.axhline(0, color="k", ls=":", lw=1, label="chance (random input = 0)")
        ax.set_xticks(range(len(LEVELS))); ax.set_xticklabels(LEVELS)
        ax.set_title(f"{ds}: unseen subjects + unseen sentences")
        ax.set_xlabel("noise added to test EEG"); ax.set_ylabel("sim_margin (true - other sentences)")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=150)
    print(f"saved {FIG / (name + '.png')}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--max-chunks", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--random-seeds", type=int, default=3)
    ap.add_argument("--pilot", action="store_true", help="1 fold, 1 epoch, few chunks: smoke test only")
    a = ap.parse_args()
    if a.pilot:
        a.epochs, a.max_chunks = 1, 1500
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    w, emb, feats, art, CH, cw = load_all()
    ok = (w.aligned.to_numpy() & w.valid.to_numpy() & np.isfinite(art)
          & np.all([np.isfinite(feats[lv]).all(axis=1) for lv in LEVELS], axis=0)
          & np.isin(np.arange(len(w)), cw))
    print(f"usable windows: {int(ok.sum())}/{len(w)}")

    rows = []
    for ds in sorted(w.dataset.unique()):
        d = w[(w.dataset == ds) & ok]
        idx, keys = d.index.to_numpy(), d.key.to_numpy()
        subs, n_sent = sorted(d.subject.unique()), d.key.nunique()
        n_hold = max(2, int(0.15 * n_sent))
        sent_split = split_items(d.key, n_val=n_hold, n_test=n_hold, seed=a.seed)
        folds = subject_folds(subs, a.folds, a.seed)
        if a.pilot:
            folds = folds[:1]
        print(f"\n{ds}: {len(d)} windows, {len(subs)} subjects, {n_sent} sentences, test/val sentences {n_hold}/{n_hold}")
        for fi, fold in enumerate(folds):
            split = assign_split(d, fold, sent_split, "subject", "key")
            counts = check_no_leakage(d, split, "subject", "key")
            sp = split.to_numpy()
            tr, va, te = (np.where(sp == nm)[0] for nm in ("train", "val", "test"))
            Ytr = np.stack([emb[k] for k in keys[tr]])
            cva, tva = candidates(keys[va], emb)
            cte, tte = candidates(keys[te], emb)
            base = dict(dataset=ds, fold=fi, test_subjects=",".join(fold["test"]))
            print(f"  fold {fi}: test subjects {fold['test']}, train/val/test windows "
                  f"{counts['train']}/{counts['val']}/{counts['test']}, test candidates {len(cte)}", flush=True)

            # ---- band power: plain, quality-weighted, random control; test signals corrupted
            Fc = feats["clean"][idx]
            tests = {lv: (feats[lv][idx][te], cte, tte) for lv in LEVELS}
            record(rows, {**base, "representation": "bandpower", "method": "plain"},
                   fit_eval(Fc[tr], Ytr, Fc[va], cva, tva, tests))
            z = (art[idx][tr] - art[idx][tr].mean()) / (art[idx][tr].std() + 1e-9)
            wq = 1.0 / (1.0 + np.clip(z, 0, None))
            record(rows, {**base, "representation": "bandpower", "method": "quality_weighted"},
                   fit_eval(Fc[tr], Ytr, Fc[va], cva, tva, tests, w=wq))
            rnd = []
            for r in range(a.random_seeds):
                R = np.random.default_rng(1000 + r).standard_normal(Fc.shape)
                rnd.append(fit_eval(R[tr], Ytr, R[va], cva, tva, {"clean": (R[te], cte, tte)}))
            record(rows, {**base, "representation": "bandpower", "method": "random_input"}, mean_metrics(rnd))

            # ---- self-supervised encoder, trained on TRAINING subjects' chunks only (no labels)
            torch.manual_seed(a.seed + fi)
            enc = Encoder().to(device)
            head = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 64)).to(device)
            enc0 = copy.deepcopy(enc).eval()
            cidx = np.where(np.isin(cw, idx[tr]))[0]
            if len(cidx) > a.max_chunks:
                cidx = np.sort(np.random.default_rng(a.seed + fi).choice(cidx, a.max_chunks, replace=False))
            print(f"    training SSL encoder on {len(cidx)} chunks from {len(set(d.subject.iloc[tr]))} subjects", flush=True)
            mu, sd = train_ssl(enc, head, CH, cidx, a.epochs, a.batch, device)

            CKPT.mkdir(parents=True, exist_ok=True)
            ck = CKPT / f"{'pilot_' if a.pilot else ''}ssl_{ds}_fold{fi}.pt"
            torch.save({"encoder": enc.state_dict(), "head": head.state_dict(), "mu": mu.cpu(), "sd": sd.cpu(),
                        "dataset": ds, "fold": fi, "train_subjects": sorted(set(d.subject.iloc[tr])),
                        "args": {k: (v if isinstance(v, (int, float, bool, str)) else str(v))
                                 for k, v in vars(a).items()}}, ck)
            print(f"    saved checkpoint {ck}", flush=True)

            for method, model in (("ssl_trained", enc), ("untrained_encoder", enc0)):
                Ztr = embed_windows(model, mu, sd, CH, cw, idx[tr], 0.0, 1, device)
                Zva = embed_windows(model, mu, sd, CH, cw, idx[va], 0.0, 1, device)
                tz = {lv: (embed_windows(model, mu, sd, CH, cw, idx[te], NOISE[lv], 100 + i, device), cte, tte)
                      for i, lv in enumerate(LEVELS)}
                record(rows, {**base, "representation": "ssl", "method": method},
                       fit_eval(Ztr, Ytr, Zva, cva, tva, tz))
            rnd = []
            for r in range(a.random_seeds):
                R = np.random.default_rng(2000 + r).standard_normal((len(d), 128))
                rnd.append(fit_eval(R[tr], Ytr, R[va], cva, tva, {"clean": (R[te], cte, tte)}))
            record(rows, {**base, "representation": "ssl", "method": "random_input"}, mean_metrics(rnd))

    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    name = "results_robust_ssl_pilot" if a.pilot else "results_robust_ssl"
    res.to_csv(OUT / f"{name}.csv", index=False)
    print("\nMEAN OVER FOLDS (test = unseen subjects AND unseen sentences; noise added to TEST EEG only)")
    g = res.groupby(["dataset", "representation", "method", "noise"], sort=False)
    print(g[["n_candidates", "top1", "chance_top1", "mrr", "chance_mrr", "sim_margin"]].mean().round(4).to_string())
    print(f"\nsaved {OUT / (name + '.csv')}")
    make_figure(res, "robustness_pilot" if a.pilot else "robustness")


if __name__ == "__main__":
    main()