"""Cross-dataset transfer for the pretrained (CBraMod) and self-supervised encoders, source-only.

For each direction (source -> target):
  1. features for source and target windows come from FROZEN encoders (CBraMod, cached by run_pretrained.py) or from the
     self-supervised checkpoints saved by run_robust_ssl.py (trained on the SOURCE dataset's training subjects only),
     plus band power and a random-input control;
  2. a ridge decoder is fitted on the source dataset only (alpha chosen on held-out source subjects AND sentences);
  3. target windows are ranked against target candidate sentences. No target labels, target EEG or target statistics
     are used for fitting or tuning (source-only transfer, no adaptation).
Target subsets: unseen_sentences (target sentences absent from the source dataset) and all_sentences.
SSL results are averaged over the 4 fold checkpoints of the source dataset (mrr_sd = SD across them).

Needs data/processed/cbramod_frozen_clean.npy and cbramod_untrained_clean.npy (made by run_pretrained.py) and
experiments/checkpoints/ssl_<dataset>_fold<k>.pt (made by run_robust_ssl.py).
Outputs: experiments/results/metrics/results_transfer_encoders.csv, experiments/results/transfer_encoders.md,
         experiments/results/figures/fig_e_transfer_encoders.png
"""
import numpy as np
import pandas as pd
import torch

from run_pretrained import evaluate, fit_model, window_features
from run_robust_ssl import CKPT, FIG, KEEP, LEVELS, OUT, P, Encoder, candidates, embed_windows, load_all

RESULTS_MD = OUT.parent / "transfer_encoders.md"
N_FOLDS, RANDOM_SEEDS = 4, 3


def source_split(keys, subj, seed=0):
    """Train = source rows outside the 2 validation subjects and outside the validation sentences."""
    subs = sorted(set(subj))
    val_subs = subs[-2:]
    uniq = sorted(set(keys))
    rng = np.random.default_rng(seed)
    val_sent = set(rng.choice(uniq, max(2, int(0.15 * len(uniq))), replace=False).tolist())
    in_val_sent = np.array([k in val_sent for k in keys])
    is_val_sub = np.isin(subj, val_subs)
    return np.where(~is_val_sub & ~in_val_sent)[0], np.where(is_val_sub & in_val_sent)[0]


def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def ssl_features(src, S, T, CH, cw, trained):
    pairs = []
    for fi in range(N_FOLDS):
        f = CKPT / f"ssl_{src}_fold{fi}.pt"
        if not f.exists():
            raise SystemExit(f"missing checkpoint {f}: run python scripts\\run_robust_ssl.py first")
        ck = torch.load(f, map_location="cpu", weights_only=False)
        enc = Encoder()
        if trained:
            enc.load_state_dict(ck["encoder"])
        else:
            torch.manual_seed(fi)  # same initialisation as the untrained copy in run_robust_ssl.py (seed 0 + fold)
            enc = Encoder()
        enc.eval()
        mu, sd = ck["mu"], ck["sd"]
        pairs.append((embed_windows(enc, mu, sd, CH, cw, S["idx"], 0.0, 1, "cpu"),
                      embed_windows(enc, mu, sd, CH, cw, T["idx"], 0.0, 1, "cpu")))
    return pairs


def main():
    for f in ("cbramod_frozen_clean.npy", "cbramod_untrained_clean.npy"):
        if not (P / f).exists():
            raise SystemExit(f"missing {P / f}: run python scripts\\run_pretrained.py first")
    w, emb, feats, art, CH, cw = load_all()
    Wf = window_features(np.load(P / "cbramod_frozen_clean.npy"), cw, len(w))
    Wu = window_features(np.load(P / "cbramod_untrained_clean.npy"), cw, len(w))
    ok = (w.aligned.to_numpy() & w.valid.to_numpy() & np.isfinite(art)
          & np.all([np.isfinite(feats[lv]).all(axis=1) for lv in LEVELS], axis=0)
          & np.isin(np.arange(len(w)), cw) & np.isfinite(Wf).all(axis=1))
    print(f"usable windows: {int(ok.sum())}/{len(w)}")
    data = {}
    for ds in ("zuco1", "zuco2"):
        d = w[(w.dataset == ds) & ok]
        data[ds] = dict(idx=d.index.to_numpy(), keys=d.key.to_numpy(), subj=d.subject.to_numpy())

    rows = []
    for src, tgt in (("zuco1", "zuco2"), ("zuco2", "zuco1")):
        S, T = data[src], data[tgt]
        tr, va = source_split(S["keys"], S["subj"])
        Ytr = np.stack([emb[k] for k in S["keys"][tr]])
        cva, tva = candidates(S["keys"][va], emb)
        src_keys = set(S["keys"])
        unseen = np.array([k not in src_keys for k in T["keys"]])
        subsets = {}
        for name, m in (("unseen_sentences", unseen), ("all_sentences", np.ones(len(T["keys"]), bool))):
            sel = np.where(m)[0]
            c, t = candidates(T["keys"][sel], emb)
            subsets[name] = (sel, c, t)
        print(f"\n{src} -> {tgt}: source train/val windows {len(tr)}/{len(va)}, target windows {len(T['idx'])}, "
              f"unseen-sentence queries {len(subsets['unseen_sentences'][0])}, "
              f"candidates {len(subsets['unseen_sentences'][1])} / {len(subsets['all_sentences'][1])}", flush=True)

        def run(rep, method, pairs):
            results = []
            for Xs, Xt in pairs:
                pred = fit_model(Xs[tr], Ytr, Xs[va], cva, tva)
                results.append(evaluate(pred, {n: (Xt[sel], c, t) for n, (sel, c, t) in subsets.items()}))
            for n, (sel, _, _) in subsets.items():
                vals = {k: float(np.mean([r[n][k] for r in results])) for k in KEEP + ["n_candidates"]}
                rows.append(dict(source=src, target=tgt, target_subset=n, representation=rep, method=method,
                                 n_target_queries=len(sel), n_runs=len(results),
                                 mrr_sd=float(np.std([r[n]["mrr"] for r in results])) if len(results) > 1 else np.nan,
                                 **vals))
            print(f"  done: {rep} / {method} ({len(results)} run(s))", flush=True)

        run("bandpower", "plain", [(feats["clean"][S["idx"]], feats["clean"][T["idx"]])])
        run("cbramod", "frozen_pretrained", [(Wf[S["idx"]], Wf[T["idx"]])])
        run("cbramod", "untrained_cbramod", [(Wu[S["idx"]], Wu[T["idx"]])])
        run("ssl", "ssl_trained", ssl_features(src, S, T, CH, cw, True))
        run("ssl", "untrained_encoder", ssl_features(src, S, T, CH, cw, False))
        rnd = [(np.random.default_rng(4000 + r).standard_normal((len(S["idx"]), 400)),
                np.random.default_rng(5000 + r).standard_normal((len(T["idx"]), 400))) for r in range(RANDOM_SEEDS)]
        run("random", "random_input", rnd)

    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "results_transfer_encoders.csv", index=False)
    show = ["source", "target", "target_subset", "representation", "method", "n_target_queries", "n_candidates",
            "top1", "chance_top1", "mrr", "chance_mrr", "mrr_sd", "sim_margin"]
    table = res[show]
    RESULTS_MD.write_text("**Cross-dataset transfer, source-only decoder; SSL rows are means over the 4 source "
                          "checkpoints (mrr_sd = SD across them)**\n\n" + md(table) + "\n", encoding="utf-8")
    print("\n" + table.round(4).to_string(index=False))
    print(f"\nsaved {OUT / 'results_transfer_encoders.csv'} and {RESULTS_MD}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    FIG.mkdir(parents=True, exist_ok=True)
    dirs = [("zuco1", "zuco2"), ("zuco2", "zuco1")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
    for ax, (s, t) in zip(axes, dirs):
        r = res[(res.source == s) & (res.target == t) & (res.target_subset == "unseen_sentences")].reset_index(drop=True)
        labels = (r.representation + "\n" + r.method).tolist()
        ax.bar(range(len(r)), r.mrr, yerr=r.mrr_sd.fillna(0), capsize=3, color="tab:blue")
        ax.axhline(r.chance_mrr.mean(), color="k", ls=":", label="chance")
        ax.set_xticks(range(len(r)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.set_title(f"{s} -> {t} (unseen target sentences, {int(r.n_candidates.iloc[0])} candidates)", fontsize=9)
        ax.set_ylabel("MRR (bars = SD over checkpoints/seeds where several)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_e_transfer_encoders.png", dpi=150)
    print(f"saved {FIG / 'fig_e_transfer_encoders.png'}")


if __name__ == "__main__":
    main()