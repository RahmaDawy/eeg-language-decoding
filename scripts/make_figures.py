"""Reads the results CSVs in experiments/results/metrics and
  1. saves three figures in experiments/results/figures:
       fig_a_eeg_vs_controls.png  EEG band power vs duration-only vs random-input control, per dataset
       fig_b_noise.png            performance vs noise added to the test EEG
       fig_c_transfer.png         ZuCo1 <-> ZuCo2 transfer
  2. writes experiments/results/results_summary.md (all tables, means over folds)
  3. fills the text between <!-- RESULTS:START --> and <!-- RESULTS:END --> in README.md (compact table)
     and reports/final_report.md (full tables)
Nothing is typed in by hand: every number comes from the CSVs.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = Path("experiments/results/metrics")
FIG = Path("experiments/results/figures")
SUMMARY = Path("experiments/results/results_summary.md")
NOISE = ["clean", "mild", "moderate", "strong"]
PREF = ["eeg_bandpower", "duration_only", "random_input"]
REGIMES = [("double", "unseen subjects + unseen sentences"), ("subject_only", "unseen subjects only")]
NORMS = [("raw", ""), ("subject-norm", "_subjnorm")]
COLORS = {"raw": "tab:blue", "subject-norm": "tab:orange"}


def load(name):
    p = M / name
    if not p.exists():
        print(f"missing (skipped): {p}")
        return None
    return pd.read_csv(p)


def order_conditions(values):
    present = list(dict.fromkeys(values))
    return [c for c in PREF if c in present] + [c for c in present if c not in PREF]


# ------------------------------------------------------------------ figure (a)
def fig_a():
    data = {}
    for reg, _ in REGIMES:
        for nm, suf in NORMS:
            df = load(f"results_{reg}{suf}.csv")
            if df is not None:
                data[(reg, nm)] = df
    if not data:
        return
    dss = sorted(set().union(*[set(d.dataset) for d in data.values()]))
    conds = order_conditions(pd.concat([d.condition for d in data.values()]))
    fig, axes = plt.subplots(len(REGIMES), len(dss), figsize=(5.5 * len(dss), 8), squeeze=False)
    for ri, (reg, title) in enumerate(REGIMES):
        for di, ds in enumerate(dss):
            ax = axes[ri][di]
            chance = []
            for vi, (nm, _) in enumerate(NORMS):
                df = data.get((reg, nm))
                if df is None:
                    continue
                sub = df[df.dataset == ds]
                chance.append(sub.chance_mrr.mean())
                for ci, c in enumerate(conds):
                    g = sub[sub.condition == c].mrr
                    if len(g) == 0:
                        continue
                    s = g.std()
                    ax.bar(ci + (vi - 0.5) * 0.38, g.mean(), 0.36, yerr=0 if np.isnan(s) else s, capsize=3,
                           color=COLORS[nm], label=nm if ci == 0 or not ax.get_legend_handles_labels()[1] else None)
            if chance:
                ax.axhline(np.mean(chance), color="k", ls=":", lw=1.2, label="chance")
            ax.set_xticks(range(len(conds)))
            ax.set_xticklabels(conds, rotation=15)
            ax.set_title(f"{ds}: {title}", fontsize=10)
            ax.set_ylabel("MRR (mean over folds, bars = SD)")
            h, l = ax.get_legend_handles_labels()
            uniq = dict(zip(l, h))
            ax.legend(uniq.values(), uniq.keys(), fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_a_eeg_vs_controls.png", dpi=150)
    plt.close(fig)
    print("saved", FIG / "fig_a_eeg_vs_controls.png")


# ------------------------------------------------------------------ figure (b)
def fig_b():
    df = load("results_robust_ssl.csv")
    if df is None:
        return
    dss = sorted(df.dataset.unique())
    fig, axes = plt.subplots(2, len(dss), figsize=(6 * len(dss), 8), squeeze=False)
    for di, ds in enumerate(dss):
        d = df[df.dataset == ds]
        for ri, (metric, ylabel) in enumerate([("mrr", "MRR"), ("sim_margin", "sim_margin (true - other sentences)")]):
            ax = axes[ri][di]
            for (rep, meth), g in d[d.method != "random_input"].groupby(["representation", "method"], sort=False):
                m = g.groupby("noise")[metric].mean().reindex(NOISE)
                s = g.groupby("noise")[metric].std().reindex(NOISE)
                ax.errorbar(range(len(NOISE)), m.values, yerr=s.values, marker="o", capsize=3, label=f"{rep} / {meth}")
            ax.axhline(d[d.method == "random_input"][metric].mean(), color="gray", ls="--", lw=1,
                       label="random-input control (mean)")
            if metric == "mrr":
                ax.axhline(d.chance_mrr.mean(), color="k", ls=":", lw=1, label="chance")
            else:
                ax.axhline(0, color="k", ls=":", lw=1, label="zero")
            ax.set_xticks(range(len(NOISE)))
            ax.set_xticklabels(NOISE)
            ax.set_xlabel("synthetic noise added to test EEG")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ds}: unseen subjects + unseen sentences (mean +- SD over folds)", fontsize=9)
            if ri == 0 and di == 0:
                ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_b_noise.png", dpi=150)
    plt.close(fig)
    print("saved", FIG / "fig_b_noise.png")


# ------------------------------------------------------------------ figure (c)
def fig_c():
    fig, axes = plt.subplots(1, len(NORMS), figsize=(6 * len(NORMS), 4.5), sharey=True, squeeze=False)
    drawn = False
    for ax, (nm, suf) in zip(axes[0], NORMS):
        df = load(f"results_transfer{suf}.csv")
        if df is None:
            continue
        drawn = True
        df = df.copy()
        df["group"] = df.source + "->" + df.target + "\n" + df.target_subset
        groups = list(dict.fromkeys(df.group))
        conds = order_conditions(df.condition)
        width = 0.8 / len(conds)
        x = np.arange(len(groups))
        for ci, c in enumerate(conds):
            vals = [df[(df.group == g) & (df.condition == c)].mrr.mean() for g in groups]
            ax.bar(x + (ci - (len(conds) - 1) / 2) * width, vals, width, label=c)
        ax.scatter(x, [df[df.group == g].chance_mrr.mean() for g in groups], marker="_", s=900, color="k",
                   zorder=3, label="chance")
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=8)
        ax.set_title(f"transfer, source-only decoder ({nm})", fontsize=10)
        ax.set_ylabel("MRR")
        ax.legend(fontsize=7)
    if drawn:
        fig.tight_layout()
        fig.savefig(FIG / "fig_c_transfer.png", dpi=150)
        print("saved", FIG / "fig_c_transfer.png")
    plt.close(fig)


# ------------------------------------------------------------------ tables
def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def mean_table(df, keys, cols):
    cols = [c for c in cols if c in df.columns]
    return df.groupby(keys, sort=False)[cols].mean().reset_index()


def build_summary():
    parts = ["_Means over subject folds. Chance = expected score of a random ranking. "
             "No significance tests were run._", ""]
    for reg, title in REGIMES:
        for nm, suf in NORMS:
            df = load(f"results_{reg}{suf}.csv")
            if df is not None:
                t = mean_table(df, ["dataset", "condition"],
                               ["n_candidates", "top1", "chance_top1", "recall@5", "chance_recall@5", "mrr",
                                "chance_mrr", "sim_margin"])
                parts += [f"**{title} ({nm})**", "", md(t), ""]
    for nm, suf in NORMS:
        df = load(f"results_transfer{suf}.csv")
        if df is not None:
            t = mean_table(df, ["source", "target", "target_subset", "condition"],
                           ["n_target_queries", "n_candidates", "top1", "chance_top1", "mrr", "chance_mrr",
                            "sim_margin"])
            parts += [f"**transfer, source-only decoder ({nm})**", "", md(t), ""]
    df = load("results_robust_ssl.csv")
    if df is not None:
        t = mean_table(df, ["dataset", "representation", "method", "noise"],
                       ["n_candidates", "top1", "chance_top1", "mrr", "chance_mrr", "sim_margin"])
        parts += ["**robustness and self-supervised encoder (unseen subjects + unseen sentences; "
                  "noise added to test EEG only)**", "", md(t), ""]
    return "\n".join(parts)


def build_compact():
    frames = []
    cols = ["top1", "chance_top1", "mrr", "chance_mrr"]
    for reg, _ in REGIMES:
        for nm, suf in NORMS:
            df = load(f"results_{reg}{suf}.csv")
            if df is not None:
                t = mean_table(df, ["dataset", "condition"], cols)
                t.insert(0, "experiment", f"{reg}, {nm}")
                frames.append(t)
    for nm, suf in NORMS:
        df = load(f"results_transfer{suf}.csv")
        if df is not None:
            t = mean_table(df, ["source", "target", "target_subset", "condition"], cols)
            t["dataset"] = t.source + "->" + t.target
            t["experiment"] = "transfer (" + nm + ", " + t.target_subset + ")"
            frames.append(t[["experiment", "dataset", "condition"] + cols])
    if not frames:
        return "(no results found)"
    return ("_Means over folds; chance is the expected score of a random ranking._\n\n"
            + md(pd.concat(frames, ignore_index=True))
            + "\n\nAll tables, including the noise/SSL runs: `experiments/results/results_summary.md`.")


def inject(path, text):
    p = Path(path)
    if not p.exists():
        print(f"not found, skipped: {p}")
        return
    s = p.read_text(encoding="utf-8")
    pat = re.compile(r"(<!-- RESULTS:START -->).*?(<!-- RESULTS:END -->)", re.S)
    if not pat.search(s):
        print(f"no RESULTS markers in {p}")
        return
    p.write_text(pat.sub(lambda m: m.group(1) + "\n" + text + "\n" + m.group(2), s), encoding="utf-8")
    print(f"updated tables in {p}")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    fig_a()
    fig_b()
    fig_c()
    summary = build_summary()
    SUMMARY.write_text(summary, encoding="utf-8")
    print("saved", SUMMARY)
    inject("README.md", build_compact())
    inject("reports/final_report.md", summary)
    print("\n" + summary)


if __name__ == "__main__":
    main()