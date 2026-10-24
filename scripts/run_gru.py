"""Word-level sequence model (GRU) vs controls, with SUBJECT-LEVEL uncertainty.

Input per sentence = sequence of per-word EEG features (the authors' 8 band means x 105 channels around the first
fixation = 840 values per fixated word). A small bidirectional GRU gives one sentence vector, trained with a contrastive
loss to match the sentence's MiniLM embedding (same-sentence rows are masked as negatives). No word/sentence text is given
to the EEG models. Test = unseen subjects AND unseen sentences (same leave-subjects-out folds as run_final.py);
candidates = the unique test sentences.
  gru_eeg_trainnorm        EEG, standardised with TRAINING-word statistics only (not transductive)
  gru_eeg_subjnorm         EEG, each subject z-scored with its own EEG (label-free but TRANSDUCTIVE)
  gru_behavior             control, NO EEG: [log first-fixation time, word length] per word
  gru_random               control: random numbers, same shape as the EEG sequences
  gru_eeg_*+behavior       EEG concatenated with the behaviour inputs: does EEG add anything BEYOND reading behaviour?
  ridge_*                  the earlier sentence-level ridge decoder on the same rows
Early stopping on VALIDATION subjects+sentences only. Metrics per test subject -> 95% bootstrap CIs over subjects.
One run; settings fixed in advance.
"""
import argparse
import re
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
from run_experiments import subject_standardize

GT = Path("data/processed/gt")
OUT = Path("experiments/results/metrics")
FIG = Path("experiments/results/figures")
ALPHAS = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)
MAXW = 60
PAIRS = [("gru_eeg_trainnorm", "gru_random"), ("gru_eeg_trainnorm", "gru_behavior"),
         ("gru_eeg_subjnorm", "gru_random"), ("gru_eeg_subjnorm", "gru_behavior"),
         ("gru_eeg_trainnorm+behavior", "gru_behavior"), ("gru_eeg_subjnorm+behavior", "gru_behavior"),
         ("ridge_official_raw", "ridge_random"), ("ridge_official_subjnorm", "ridge_random"),
         ("ridge_official_raw", "ridge_length_only"), ("ridge_official_subjnorm", "ridge_length_only")]
COL = {"gru_random": "#9e9e9e", "ridge_random": "#9e9e9e", "gru_behavior": "#424242", "ridge_length_only": "#424242"}


def clean(w):
    return re.sub(r"[^a-z0-9']", "", str(w).lower())


def load():
    R = []
    for f in sorted(GT.glob("*.npz")):
        z = np.load(f)
        texts, off, ns, nw = z["text"], z["official"], z["nsamp"], z["nwords"]
        ws, wf, wt, wff = z["w_sent"], z["w_feat"], z["w_text"], z["w_ffd"]  # read each array ONCE (npz is lazy)
        by = {}
        for j, s in enumerate(ws):
            by.setdefault(int(s), []).append(j)
        for i, t in enumerate(texts):
            js = by.get(i, [])[:MAXW]
            if len(js) < 2:
                continue
            ffd = np.log(np.clip(wff[js].astype(np.float64), 1.0, None))
            nch = np.array([len(clean(w)) for w in wt[js]], dtype=np.float64)
            R.append(dict(subject=f.stem, text=str(t), key=sentence_key(str(t)), eeg=wf[js].astype(np.float32),
                          beh=np.stack([ffd, nch], axis=1).astype(np.float32), official=off[i].astype(np.float32),
                          length=np.array([np.log(ns[i]), np.log(max(int(nw[i]), 1))])))
    return R


def subject_norm_seqs(seqs, subjects):
    out = list(seqs)
    for s in np.unique(subjects):
        idx = np.where(subjects == s)[0]
        allw = np.concatenate([seqs[i] for i in idx])
        mu, sd = allw.mean(0), allw.std(0) + 1e-6
        for i in idx:
            out[i] = ((seqs[i] - mu) / sd).astype(np.float32)
    return out


class Seqs:
    def __init__(self, arrays, needs_norm=False):
        self.a, self.needs_norm, self.norm = arrays, needs_norm, None
        self.dim = arrays[0].shape[1]

    def fit_norm(self, idx):  # statistics from TRAINING rows only
        allw = np.concatenate([self.a[i] for i in idx])
        self.norm = (allw.mean(0), allw.std(0) + 1e-6)

    def batch(self, idx):
        xs = [self.a[i] for i in idx]
        if self.norm is not None:
            xs = [(x - self.norm[0]) / self.norm[1] for x in xs]
        L = max(len(x) for x in xs)
        X = np.zeros((len(xs), L, self.dim), np.float32)
        for k, x in enumerate(xs):
            X[k, :len(x)] = x
        return torch.from_numpy(X), torch.tensor([len(x) for x in xs])


class SeqNet(nn.Module):
    def __init__(self, d_in, d_out, h=64, p=0.3):
        super().__init__()
        self.inp = nn.Sequential(nn.Dropout(p), nn.Linear(d_in, 128), nn.GELU(), nn.Dropout(p))
        self.gru = nn.GRU(128, h, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * h, d_out)

    def forward(self, x, lens):
        z = self.inp(x)
        pk = nn.utils.rnn.pack_padded_sequence(z, lens, batch_first=True, enforce_sorted=False)
        o, _ = self.gru(pk)
        o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=x.shape[1])
        mask = (torch.arange(x.shape[1], device=x.device)[None, :] < lens.to(x.device)[:, None]).float().unsqueeze(-1)
        return self.out((o * mask).sum(1) / mask.sum(1).clamp(min=1))


def predict(net, S, idx, device, bs=256):
    net.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            x, l = S.batch(idx[i:i + bs])
            outs.append(net(x.to(device), l).cpu())
    return torch.cat(outs).numpy()


def fit_gru(S, Y, keys, tr, va, cva, tva, device, seed, epochs, patience, bs=64):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    net = SeqNet(S.dim, Y.shape[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    Yt = F.normalize(torch.tensor(Y, dtype=torch.float32), dim=1).to(device)
    best_state, best_mrr, bad, ran = None, -1.0, 0, 0
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(tr)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            if len(b) < 16:
                continue
            x, l = S.batch(b)
            p = F.normalize(net(x.to(device), l), dim=1)
            logits = p @ Yt[b].T / 0.1
            kb = keys[b]
            same = torch.from_numpy(kb[:, None] == kb[None, :]).to(device)
            same &= ~torch.eye(len(b), dtype=torch.bool, device=device)  # same sentence = not a negative
            loss = F.cross_entropy(logits.masked_fill(same, -1e9), torch.arange(len(b), device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        ran = ep + 1
        mrr = retrieval_metrics(cosine_matrix(predict(net, S, va, device), cva), tva)["mrr"]  # VALIDATION only
        if mrr > best_mrr + 1e-4:
            best_mrr, bad = mrr, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return net, best_mrr, ran


def ridge_predict(Xtr, Ytr, Xva, cva, tva, Xte):
    sc = StandardScaler().fit(Xtr)  # statistics from training rows only
    Ztr, Zva, Zte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    best, best_m = None, -np.inf
    for al in ALPHAS:
        m = Ridge(alpha=al).fit(Ztr, Ytr)
        mrr = retrieval_metrics(cosine_matrix(m.predict(Zva), cva), tva)["mrr"]
        if mrr > best_m:
            best, best_m = m, mrr
    return best.predict(Zte)


def candidates(keys_, emb):
    uniq, true = candidate_index(list(keys_))
    return np.stack([emb[u] for u in uniq]), true


def subject_metrics(P, true_idx, cand, subj_te):
    out = []
    for s in sorted(set(subj_te)):
        m = subj_te == s
        out.append((s, retrieval_metrics(cosine_matrix(P[m], cand), true_idx[m])))
    return out


def boot(v, n, rng):
    v = np.asarray(v, float)
    means = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)])
    return v.mean(), np.quantile(means, 0.025), np.quantile(means, 0.975), means


def make_figure(per, summ, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    ms = list(summ.method)
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.5 * len(ms) + 2.4), sharey=True)
    for ax, (mean, lo, hi, metric, xl) in zip(axes, [("sim_margin", "sm_lo", "sm_hi", "sim_margin", "sim_margin  (0 = no information)"),
                                                     ("mrr_above_chance", "mrr_lo", "mrr_hi", "mrr_above_chance", "MRR minus chance  (0 = chance)")]):
        for i, r in enumerate(summ.itertuples()):
            y = len(ms) - 1 - i
            c = COL.get(r.method, "#1f77b4" if "eeg" in r.method or "official" in r.method else "#9e9e9e")
            mu, l, h = getattr(r, mean), getattr(r, lo), getattr(r, hi)
            ax.barh(y, mu, color=c, height=0.7, xerr=[[mu - l], [h - mu]], capsize=3)
            g = per[per.method == r.method]
            vals = g.sim_margin if metric == "sim_margin" else g.mrr - g.chance_mrr
            ax.scatter(vals, np.full(len(vals), y), s=12, color="black", zorder=3)
        ax.axvline(0, color="red", ls="--", lw=1)
        ax.set_xlabel(xl)
    axes[0].set_yticks(range(len(ms)))
    axes[0].set_yticklabels(ms[::-1])
    fig.suptitle("Word-level GRU vs controls: unseen subjects AND unseen sentences\n"
                 "bars = mean over test subjects with 95% bootstrap CI, dots = individual subjects", fontsize=11)
    fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", FIG / f"{name}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--pilot", action="store_true", help="one fold, 2 epochs: smoke test only")
    a = ap.parse_args()
    if a.pilot:
        a.epochs, a.patience = 2, 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    R = load()
    subj = np.array([r["subject"] for r in R])
    keys = np.array([r["key"] for r in R])
    first = {}
    for r in R:
        first.setdefault(r["key"], r["text"])
    ukeys = list(first)
    emb = dict(zip(ukeys, embed_texts([first[k] for k in ukeys], cache_path="data/processed/gt_gru_sent_emb")))
    Y = np.stack([emb[k] for k in keys])
    d = pd.DataFrame({"subject": subj, "key": keys})
    subs = sorted(d.subject.unique())
    n_hold = max(2, int(0.15 * d.key.nunique()))
    sent_split = split_items(d.key, n_hold, n_hold, a.seed)
    folds = subject_folds(subs, min(a.folds, len(subs)), a.seed)
    folds = folds[:1] if a.pilot else folds
    print(f"{len(d)} sentences with word-level EEG, {len(subs)} subjects, {d.key.nunique()} unique sentences, "
          f"val/test sentences held out: {n_hold}/{n_hold}, device {device}")

    eeg = [r["eeg"] for r in R]
    beh_raw = [r["beh"] for r in R]
    beh = subject_norm_seqs(beh_raw, subj)
    eeg_sn = subject_norm_seqs(eeg, subj)
    rnd = [np.random.default_rng(1000 + i).standard_normal(x.shape).astype(np.float32) for i, x in enumerate(eeg)]
    seq_sets = {
        "gru_random": Seqs(rnd),
        "gru_behavior": Seqs(beh),
        "gru_eeg_trainnorm": Seqs(eeg, needs_norm=True),
        "gru_eeg_subjnorm": Seqs(eeg_sn),
        "gru_eeg_trainnorm+behavior": Seqs([np.concatenate([e, b], axis=1) for e, b in zip(eeg, beh_raw)], needs_norm=True),
        "gru_eeg_subjnorm+behavior": Seqs([np.concatenate([e, b], axis=1) for e, b in zip(eeg_sn, beh)]),
    }
    OFF = np.stack([r["official"] for r in R])
    ridge_sets = {
        "ridge_random": np.random.default_rng(1000).standard_normal(OFF.shape),
        "ridge_length_only": np.stack([r["length"] for r in R]),
        "ridge_official_raw": OFF,
        "ridge_official_subjnorm": subject_standardize(OFF, subj),
    }

    rows = []
    for fi, fold in enumerate(folds):
        split = assign_split(d, fold, sent_split, "subject", "key")
        counts = check_no_leakage(d, split, "subject", "key")  # raises if subjects or sentences are shared
        sp = split.to_numpy()
        tr, va, te = (np.where(sp == nm)[0] for nm in ("train", "val", "test"))
        cva, tva = candidates(keys[va], emb)
        cte, tte = candidates(keys[te], emb)
        print(f"\nfold {fi}: test subjects {fold['test']}, sentences train/val/test "
              f"{counts['train']}/{counts['val']}/{counts['test']}, test candidates {len(cte)}", flush=True)
        preds = {n: ridge_predict(X[tr], Y[tr], X[va], cva, tva, X[te]) for n, X in ridge_sets.items()}
        for name, S in seq_sets.items():
            if S.needs_norm:
                S.fit_norm(tr)
            net, vm, ne = fit_gru(S, Y, keys, tr, va, cva, tva, device, a.seed + fi, a.epochs, a.patience)
            preds[name] = predict(net, S, te, device)
            print(f"    {name}: {ne} epochs, best validation MRR {vm:.4f}", flush=True)
        for name, P in preds.items():
            for s, r in subject_metrics(P, tte, cte, subj[te]):
                rows.append(dict(method=name, fold=fi, subject=s, n_queries=r["n_queries"], n_candidates=r["n_candidates"],
                                 top1=r["top1"], chance_top1=r["chance_top1"], mrr=r["mrr"], chance_mrr=r["chance_mrr"],
                                 sim_margin=r["sim_margin"]))

    per = pd.DataFrame(rows)
    tag = "_pilot" if a.pilot else ""
    OUT.mkdir(parents=True, exist_ok=True)
    per.to_csv(OUT / f"results_gru_subject{tag}.csv", index=False)
    rng = np.random.default_rng(0)
    S_rows = []
    for m, g in per.groupby("method", sort=False):
        mu, lo, hi, _ = boot(g.sim_margin, a.boot, rng)
        mm, mlo, mhi, _ = boot(g.mrr - g.chance_mrr, a.boot, rng)
        S_rows.append(dict(method=m, subjects=len(g), top1_pct=g.top1.mean() * 100, chance_top1_pct=g.chance_top1.mean() * 100,
                           mrr=g.mrr.mean(), chance_mrr=g.chance_mrr.mean(), sim_margin=mu, sm_lo=lo, sm_hi=hi,
                           mrr_above_chance=mm, mrr_lo=mlo, mrr_hi=mhi))
    summ = pd.DataFrame(S_rows)
    summ.to_csv(OUT / f"results_gru_summary{tag}.csv", index=False)
    P_rows = []
    for x, y in PAIRS:
        gx, gy = per[per.method == x].set_index("subject"), per[per.method == y].set_index("subject")
        for metric in ("sim_margin", "mrr"):
            diff = (gx[metric] - gy[metric].reindex(gx.index)).dropna().to_numpy()
            mu, lo, hi, means = boot(diff, a.boot, rng)
            P_rows.append(dict(a=x, b=y, metric=metric, mean_diff=mu, lo=lo, hi=hi, share_le_0=float((means <= 0).mean()),
                               subjects=len(diff)))
    pairs = pd.DataFrame(P_rows)
    pairs.to_csv(OUT / f"results_gru_pairs{tag}.csv", index=False)

    pd.set_option("display.width", 220)
    print("\nPER-SUBJECT METRICS, mean over test subjects, with 95% bootstrap CI over subjects")
    show = summ.copy()
    show["sim_margin [95% CI]"] = [f"{r.sim_margin:.4f} [{r.sm_lo:.4f}, {r.sm_hi:.4f}]" for r in summ.itertuples()]
    show["MRR above chance [95% CI]"] = [f"{r.mrr_above_chance:.4f} [{r.mrr_lo:.4f}, {r.mrr_hi:.4f}]" for r in summ.itertuples()]
    print(show[["method", "subjects", "top1_pct", "chance_top1_pct", "mrr", "chance_mrr", "sim_margin [95% CI]",
                "MRR above chance [95% CI]"]].round(3).to_string(index=False))
    print("\nPAIRED DIFFERENCES (a minus b), bootstrap over subjects; share_le_0 = share of resamples with difference <= 0")
    print(pairs.round(4).to_string(index=False))
    print("(exploratory: no correction for multiple comparisons; one seed)")
    make_figure(per, summ, "fig9_gru_subject_level" + tag)


if __name__ == "__main__":
    main()