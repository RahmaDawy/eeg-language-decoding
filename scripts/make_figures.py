"""Reads the results CSVs in experiments/results/metrics and
  1. saves four figures in experiments/results/figures:
       fig_a_eeg_vs_controls.png  EEG band power vs duration-only vs random-input control, per dataset
       fig_b_noise.png            performance vs noise added to the test EEG (band power, SSL, pretrained CBraMod)
       fig_c_transfer.png         ZuCo1 <-> ZuCo2 transfer
       fig_d_subjects.png         per-test-subject MRR minus chance (clean EEG), with subject-level bootstrap CI
  2. writes experiments/results/results_summary.md (all tables, means over folds)
  3. fills the text between <!-- RESULTS:START --> and <!-- RESULTS:END --> in README.md (compact tables)
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


def load_robust():
    frames = [f for f in (load("results_robust_ssl.csv"), load("results_pretrained.csv")) if f is not None]
    return pd.concat(frames, ignore_index=True) if frames else None


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
                           color=COLORS[nm], label=nm)
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
    df = load_robust()
    if df is None:
        return
    dss = sorted(df.dataset.unique())
    fig, axes = plt.subplots(2, len(dss), figsize=(6.5 * len(dss), 8.5), squeeze=False)
    for di, ds in enumerate(dss):
        d = df[df.dataset == ds]
        for ri, (metric, ylabel) in enumerate([("mrr", "MRR"), ("sim_margin", "sim_margin (true - other sentences)")]):
            ax = axes[ri][di]
            for (rep, meth), g in d[d.method != "random_input"].groupby(["representation", "method"], sort=False):
                m = g.groupby("noise")[metric].mean().reindex(NOISE)
                s = g.groupby("noise")[metric].std().reindex(NOISE)
                ax.errorbar(range(len(NOISE)), m.values, yerr=s.values, marker="o", capsize=3, label=f"{rep} / {meth}")
            ax.axhline(d[d.method == "random_input"][metric].mean(), color="gray", ls="--", lw=1,
                       label="random-input controls (mean)")
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
                ax.legend(fontsize=6)
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


# ------------------------------------------------------------------ subject-level analysis + figure (d)
def subject_analysis(n_boot=5000):
    df = load("results_pretrained_subjects.csv")
    if df is None:
        return None, None
    df = df.copy()
    df["mrr_minus_chance"] = df.mrr - df.chance_mrr
    rng = np.random.default_rng(0)
    rows = []
    for (ds, meth), g in df.groupby(["dataset", "method"], sort=False):
        s = g.groupby("subject")[["mrr_minus_chance", "sim_margin"]].mean()  # one value per subject
        n = len(s)
        row = {"dataset": ds, "method": meth, "n_subjects": n}
        for col in ("mrr_minus_chance", "sim_margin"):
            v = s[col].to_numpy()
            boots = [v[rng.integers(0, n, n)].mean() for _ in range(n_boot)]  # resample SUBJECTS, not windows
            lo, hi = np.percentile(boots, [2.5, 97.5])
            row[f"{col}_mean"], row[f"{col}_ci_lo"], row[f"{col}_ci_hi"] = v.mean(), lo, hi
        rows.append(row)
    return pd.DataFrame(rows), df


def fig_d(df):
    if df is None:
        return
    dss = sorted(df.dataset.unique())
    methods = list(dict.fromkeys(df.method))
    fig, axes = plt.subplots(1, len(dss), figsize=(6.5 * len(dss), 4.5), sharey=True, squeeze=False)
    rng = np.random.default_rng(1)
    for ax, ds in zip(axes[0], dss):
        d = df[df.dataset == ds]
        for mi, m in enumerate(methods):
            v = d[d.method == m].groupby("subject").mrr_minus_chance.mean()
            ax.scatter(mi + rng.uniform(-0.12, 0.12, len(v)), v.values, alpha=0.8)
            ax.hlines(v.mean(), mi - 0.25, mi + 0.25, color="k", lw=2)
        ax.axhline(0, color="k", ls=":", lw=1)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=20, fontsize=8)
        ax.set_title(f"{ds}: each dot = one unseen test subject (clean EEG)", fontsize=9)
        ax.set_ylabel("MRR minus chance")
    fig.tight_layout()
    fig.savefig(FIG / "fig_d_subjects.png", dpi=150)
    plt.close(fig)
    print("saved", FIG / "fig_d_subjects.png")


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


def build_summary(subj_tab):
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
    df = load_robust()
    if df is not None:
        t = mean_table(df, ["dataset", "representation", "method", "noise"],
                       ["n_candidates", "top1", "chance_top1", "mrr", "chance_mrr", "sim_margin"])
        parts += ["**representations, robustness and quality weighting (unseen subjects + unseen sentences; "
                  "noise added to test EEG only)**", "", md(t), ""]
    if subj_tab is not None:
        parts += ["**subject-level analysis (clean EEG, unseen subjects + unseen sentences): mean over test subjects, "
                  "95% bootstrap CI from resampling subjects (5000 resamples)**", "", md(subj_tab), ""]
    return "\n".join(parts)


def build_compact(subj_tab):
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
    rob = load_robust()
    if rob is not None:
        t = mean_table(rob[rob.noise == "clean"], ["dataset", "representation", "method"], cols)
        t["condition"] = t.representation + " / " + t.method
        t["experiment"] = "representations, clean EEG, double"
        frames.append(t[["experiment", "dataset", "condition"] + cols])
    if not frames:
        return "(no results found)"
    out = ("_Means over folds; chance is the expected score of a random ranking._\n\n"
           + md(pd.concat(frames, ignore_index=True)))
    if subj_tab is not None:
        out += ("\n\n_Subject-level (clean EEG, double regime): mean over test subjects of MRR minus chance, "
                "95% bootstrap CI over subjects._\n\n"
                + md(subj_tab[["dataset", "method", "n_subjects", "mrr_minus_chance_mean",
                               "mrr_minus_chance_ci_lo", "mrr_minus_chance_ci_hi"]]))
    return out + "\n\nAll tables: `experiments/results/results_summary.md`."


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
    subj_tab, subj_df = subject_analysis()
    fig_d(subj_df)
    summary = build_summary(subj_tab)
    SUMMARY.write_text(summary, encoding="utf-8")
    print("saved", SUMMARY)
    inject("README.md", build_compact(subj_tab))
    inject("reports/final_report.md", summary)
    print("\n" + summary)


if __name__ == "__main__":
    main()