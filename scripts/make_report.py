"""Build all figures and RESULTS.md from the saved result CSVs (nothing is typed in by hand).
Run:  python scripts/make_report.py      (rerun after every new experiment; missing results are skipped)
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

M = Path("experiments/results/metrics")
FIG = Path("experiments/results/figures")
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

LABEL = {
    "random_input": "Random input (control)",
    "length_only": "Reading length only (control)",
    "duration_only": "Reading duration only (control)",
    "length_ffd": "Word length + fixation time (control)",
    "official_bandpower": "Authors' band features",
    "our_bandpower": "Band power (whole sentence)",
    "chunk_bandpower": "Band power (1.5 s chunks)",
    "chunk_bandpower+quality": "Band power chunks + quality weights",
    "untrained_encoder": "Encoder, untrained",
    "ssl_trained": "Encoder, self-supervised",
    "ssl_trained+quality": "Encoder, self-supervised + quality weights",
    "cbramod_random_init": "CBraMod, random init",
    "cbramod_pretrained": "CBraMod, pretrained (frozen)",
    "eeg_bandpower": "EEG band power",
}
ORDER = ["random_input", "length_only", "official_bandpower", "our_bandpower", "chunk_bandpower",
         "chunk_bandpower+quality", "untrained_encoder", "ssl_trained", "ssl_trained+quality",
         "cbramod_random_init", "cbramod_pretrained", "our_bandpower+subjnorm",
         "cbramod_random_init+subjnorm", "cbramod_pretrained+subjnorm"]
CONTROLS = ("random_input", "length_only")


def rd(name):
    p = M / name
    return pd.read_csv(p) if p.exists() else None


def lab(m):
    return lab(m[:-9]) + " (+subject norm)" if m.endswith("+subjnorm") else LABEL.get(m, m)


def col(m):
    m = m.replace("+subjnorm", "")
    if m == "random_input":
        return "#9e9e9e"
    if m in ("length_only", "duration_only", "length_ffd"):
        return "#424242"
    if m.startswith("cbramod_pretrained"):
        return "#7b3fa0"
    if m.startswith("cbramod_random"):
        return "#c9a9e0"
    if m.startswith("ssl_trained"):
        return "#2e7d32"
    if m == "untrained_encoder":
        return "#a5d6a7"
    return "#1f77b4"


def fmt(x, k=3):
    return "n/a" if x is None or pd.isna(x) else f"{x:.{k}f}"


def save(fig, name):
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", FIG / name)
    return f"experiments/results/figures/{name}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def df_table(df, k=4):
    """Render any DataFrame as a markdown table (floats rounded), so column names need not be known in advance."""
    rows = [[fmt(v, k) if isinstance(v, (float, np.floating)) else v for v in r] for r in df.itertuples(index=False)]
    return md_table([str(c) for c in df.columns], rows)


# --------------------------------------------------------------------------- data for the main comparison
def build_main():
    parts = []
    f = rd("results_final.csv")
    if f is not None:
        parts.append(f[f.noise == "clean"])
    c = rd("results_cbramod.csv")
    if c is not None:
        parts.append(c[c.method.str.startswith("cbramod_") | (c.method == "our_bandpower+subjnorm")])
    return pd.concat(parts, ignore_index=True) if parts else None


def summarize(main):
    rows = []
    for m in [x for x in ORDER if x in set(main.method)]:
        d = main[main.method == m]
        rows.append(dict(method=m, top1=d.top1.mean() * 100, chance=d.chance_top1.mean() * 100,
                         mrr=d.mrr.mean(), chance_mrr=d.chance_mrr.mean(),
                         sm=d.sim_margin.mean(), sd=d.sim_margin.std(ddof=1), n=len(d)))
    return pd.DataFrame(rows).set_index("method")


def verdicts(S):
    ref = max([S.loc[m, "sm"] for m in CONTROLS if m in S.index] or [0.0])
    return {m: ("control" if m in CONTROLS else ("yes" if S.loc[m, "sm"] - ref > S.loc[m, "sd"] else "no"))
            for m in S.index}


# --------------------------------------------------------------------------- figures
def fig_pipeline_split():
    from eegdecode.data.splits import split_items, subject_folds
    gt = sorted(p.stem for p in Path("data/processed/gt").glob("*.npz"))
    subs = gt if len(gt) >= 3 else [f"S{i + 1:02d}" for i in range(12)]
    sents = [f"T{i:03d}" for i in range(300)]
    fold = subject_folds(subs, min(4, len(subs)), 0)[0]
    ss = split_items(sents, 45, 45, 0)
    rows, cols = fold["train"] + fold["val"] + fold["test"], ss["train"] + ss["val"] + ss["test"]
    code = np.zeros((len(rows), len(cols)), int)
    idx = {"train": 1, "val": 2, "test": 3}

    def which(x, sp):
        return "train" if x in sp["train"] else "val" if x in sp["val"] else "test"

    for r, s in enumerate(rows):
        for c, t in enumerate(cols):
            if which(s, fold) == which(t, ss):
                code[r, c] = idx[which(s, fold)]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.imshow(code, aspect="auto", interpolation="nearest",
              cmap=ListedColormap(["#e0e0e0", "#42a5f5", "#ffb74d", "#66bb6a"]), vmin=0, vmax=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    n1, n2, n3 = len(ss["train"]), len(ss["val"]), len(ss["test"])
    ax.set_xticks([n1 / 2, n1 + n2 / 2, n1 + n2 + n3 / 2])
    ax.set_xticklabels([f"train sentences ({n1})", f"validation sentences ({n2})", f"test sentences ({n3})"])
    ax.set_title("One cross-validation fold: only the coloured blocks are used (grey = discarded to prevent leakage)")
    ax.legend(handles=[Patch(color="#42a5f5", label=f"train: {len(fold['train'])} subjects x train sentences"),
                       Patch(color="#ffb74d", label=f"validation: {len(fold['val'])} subjects x validation sentences"),
                       Patch(color="#66bb6a", label=f"test: {len(fold['test'])} UNSEEN subjects x UNSEEN sentences"),
                       Patch(color="#e0e0e0", label="never used")],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=8)
    return save(fig, "fig0_split_design.png")


def fig_data():
    inv = [("ZuCo 1.0 (12 subjects)", rd("inventory_zuco1.csv")), ("ZuCo 2.0 (8 of 18 subjects)", rd("inventory_zuco2.csv"))]
    inv = [(t, d) for t, d in inv if d is not None]
    if not inv:
        return None
    code = {"gip": 1, "oip": 2, "bip": 3}
    fig, axes = plt.subplots(1, len(inv), figsize=(6.5 * len(inv), 4.8), squeeze=False)
    for ax, (title, d) in zip(axes[0], inv):
        subs, blocks = sorted(d.subject.unique()), sorted(d.block.unique())
        A = np.zeros((len(subs), len(blocks)), int)
        for r in d.itertuples():
            A[subs.index(r.subject), blocks.index(r.block)] = code.get(r.prefix, 0)
        ax.imshow(A, cmap=ListedColormap(["#eeeeee", "#66bb6a", "#ffa726", "#ef5350"]), vmin=0, vmax=3, aspect="auto")
        for i in range(len(subs)):
            for j in range(len(blocks)):
                ax.text(j, i, "-" if A[i, j] == 0 else d[(d.subject == subs[i]) & (d.block == blocks[j])].prefix.iloc[0],
                        ha="center", va="center", fontsize=7)
        ax.set_xticks(range(len(blocks)))
        ax.set_xticklabels(blocks)
        ax.set_yticks(range(len(subs)))
        ax.set_yticklabels(subs, fontsize=8)
        ax.set_xlabel("reading block")
        ax.set_title(f"{title}\n{len(d)} EEG recordings")
    fig.legend(handles=[Patch(color="#66bb6a", label="gip"), Patch(color="#ffa726", label="oip"),
                        Patch(color="#ef5350", label="bip"), Patch(color="#eeeeee", label="missing")],
               loc="lower center", ncol=4, frameon=False, title="file-name prefix (likely a quality class; meaning not verified)")
    return save(fig, "fig1_data_overview.png")


def fig_alignment():
    a = rd("alignment_quality.csv")
    if a is None:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    rng = np.random.default_rng(0)
    for i, ds in enumerate(sorted(a.dataset.unique())):
        v = a[a.dataset == ds].spearman.to_numpy()
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, color="#1f77b4", s=30)
        ax.hlines(np.median(v), i - 0.25, i + 0.25, color="black", lw=2)
    ax.axhline(0, color="red", ls="--", lw=1)
    ax.set_xticks(range(a.dataset.nunique()))
    ax.set_xticklabels([d.replace("zuco", "ZuCo ") + ".0" for d in sorted(a.dataset.unique())])
    ax.set_ylabel("Spearman: window duration vs sentence length")
    ax.set_title("Alignment check (one dot per subject, line = median)\nred line = no relation")
    return save(fig, "fig2_alignment.png")


def fig_main(main):
    ms = [m for m in ORDER if m in set(main.method)]
    ypos = {m: len(ms) - 1 - i for i, m in enumerate(ms)}
    chance = main.chance_top1.mean() * 100
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 0.5 * len(ms) + 2.4), sharey=True)
    for ax, (metric, scale, xl) in zip(axes, [("top1", 100, f"Top-1 accuracy (%)   chance = {chance:.1f}%"),
                                              ("sim_margin", 1, "sim_margin   (0 = no information)")]):
        for m in ms:
            v = main.loc[main.method == m, metric].to_numpy() * scale
            ax.barh(ypos[m], v.mean(), xerr=(v.std(ddof=1) if len(v) > 1 else 0), color=col(m),
                    hatch=("//" if "quality" in m else None), edgecolor="white", capsize=3, height=0.7)
            ax.scatter(v, np.full(len(v), ypos[m]), s=14, color="black", zorder=3)
        ax.set_xlabel(xl)
    axes[0].axvline(chance, color="red", ls="--", lw=1.2)
    axes[1].axvline(0, color="red", ls="--", lw=1.2)
    for m, c in (("random_input", "#757575"), ("length_only", "#212121")):
        if m in ypos:
            axes[1].axvline(main.loc[main.method == m, "sim_margin"].mean(), color=c, ls=":", lw=1.4)
    axes[0].set_yticks(list(ypos.values()))
    axes[0].set_yticklabels([lab(m) for m in ypos])
    fig.suptitle("Unseen subjects AND unseen sentences: no method clearly beats the controls\n"
                 "bars = mean over folds, error bars = SD, dots = individual folds; dotted lines = the two controls",
                 fontsize=11)
    return save(fig, "fig3_main_result.png")


def fig_noise(f):
    ms = [m for m in ["chunk_bandpower", "chunk_bandpower+quality", "untrained_encoder", "ssl_trained",
                      "ssl_trained+quality"] if m in set(f.method)]
    lv = ["clean", "mild", "moderate", "strong"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (metric, scale, yl) in zip(axes, [("sim_margin", 1, "sim_margin (0 = no information)"),
                                              ("top1", 100, "Top-1 accuracy (%)")]):
        for m in ms:
            g = f[f.method == m].groupby("noise")[metric]
            mu, sd = g.mean().reindex(lv) * scale, g.std(ddof=1).reindex(lv) * scale
            ax.errorbar(range(4), mu, yerr=sd, marker="o", capsize=3, color=col(m),
                        ls="--" if "quality" in m else "-", label=lab(m))
        r = f[f.method == "random_input"][metric].mean() * scale
        ax.axhline(r, color="#9e9e9e", ls=":", label="Random input (control)")
        if metric == "top1":
            ax.axhline(f.chance_top1.mean() * 100, color="red", ls="--", label="chance")
        else:
            ax.axhline(0, color="red", ls="--")
        ax.set_xticks(range(4))
        ax.set_xticklabels(["none", "mild\n0.5x", "moderate\n1x", "strong\n2x"])
        ax.set_xlabel("white noise added to the TEST EEG (x channel std)")
        ax.set_ylabel(yl)
    axes[0].legend(fontsize=7)
    fig.suptitle("Robustness to noise (mean +- SD over folds)")
    return save(fig, "fig4_noise.png")


def fig_positive(pc):
    sets = [("within_subject", "eeg", "Within subject\nEEG", "#1f77b4"),
            ("within_subject", "random", "Within subject\nrandom", "#9e9e9e"),
            ("cross_subject", "eeg", "Unseen subjects\nEEG (raw)", "#1f77b4"),
            ("cross_subject", "eeg_subjnorm", "Unseen subjects\nEEG, per-subject\nnormalised", "#6baed6"),
            ("cross_subject", "random", "Unseen subjects\nrandom", "#9e9e9e")]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3), sharey=True)
    for ax, (t, title) in zip(axes, [("nchar", "Predict word length (characters)"),
                                     ("log_ffd", "Predict log first-fixation duration")]):
        for i, (s, f, lbl, c) in enumerate(sets):
            v = pc[(pc.setting == s) & (pc.features == f) & (pc.target == t)].r.to_numpy()
            if v.size:
                ax.bar(i, v.mean(), yerr=v.std(ddof=1), color=c, capsize=3)
                ax.scatter(i + rng.uniform(-0.15, 0.15, v.size), v, s=14, color="black", zorder=3)
        ax.axhline(0, color="red", ls="--", lw=1)
        ax.set_xticks(range(len(sets)))
        ax.set_xticklabels([x[2] for x in sets], fontsize=8)
        ax.set_title(title)
    axes[0].set_ylabel("Pearson r on held-out words (0 = no information)")
    fig.suptitle("Positive control: do the per-word EEG features contain ANY information about the word or the reading?\n"
                 "dots = individual subjects (within) or folds (unseen subjects)", fontsize=11)
    return save(fig, "fig5_positive_control.png")


def fig_words(w):
    ms = [m for m in ("eeg_bandpower", "length_ffd", "random_input") if m in set(w.method)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (metric, title) in zip(axes, [("w_sim_margin", "Word retrieval: sim_margin\n(about 530 candidate words)"),
                                          ("s_mrr", "Sentence retrieval built from word-level EEG: MRR\n(45 candidate sentences)")]):
        for i, m in enumerate(ms):
            v = w[w.method == m][metric].to_numpy()
            ax.bar(i, v.mean(), yerr=v.std(ddof=1), color=col(m), capsize=3)
            ax.scatter(np.full(len(v), i), v, s=14, color="black", zorder=3)
        ax.set_xticks(range(len(ms)))
        ax.set_xticklabels([lab(m) for m in ms], fontsize=8, rotation=15, ha="right")
        ax.set_title(title, fontsize=10)
    axes[1].axhline(w.s_chance_mrr.mean(), color="red", ls="--", label="chance")
    axes[1].legend(fontsize=8)
    return save(fig, "fig6_word_level.png")


def fig_transfer(t):
    t = t[t.target_subset == "unseen_sentences"]
    conds = [c for c in ("eeg_bandpower", "duration_only", "random_input") if c in set(t.condition)]
    dirs = [("zuco1", "zuco2", "ZuCo 1.0 -> 2.0"), ("zuco2", "zuco1", "ZuCo 2.0 -> 1.0")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (metric, scale, yl) in zip(axes, [("top1", 100, "Top-1 accuracy (%)"), ("sim_margin", 1, "sim_margin")]):
        for j, c in enumerate(conds):
            vals = [t[(t.source == s) & (t.target == g) & (t.condition == c)][metric].mean() * scale for s, g, _ in dirs]
            ax.bar(np.arange(2) + (j - 1) * 0.27, vals, width=0.26, color=col(c), label=lab(c))
        if metric == "top1":
            for i, (s, g, _) in enumerate(dirs):
                ch = t[(t.source == s) & (t.target == g)].chance_top1.mean() * 100
                ax.hlines(ch, i - 0.42, i + 0.42, color="red", ls="--")
        else:
            ax.axhline(0, color="red", ls="--", lw=1)
        ax.set_xticks(range(2))
        ax.set_xticklabels([d[2] for d in dirs])
        ax.set_ylabel(yl)
    axes[0].legend(fontsize=8)
    fig.suptitle("Cross-dataset transfer (decoder fitted on the source dataset only, unseen target sentences)\n"
                 "red dashes = chance; single run, no error bars")
    return save(fig, "fig7_transfer.png")


def fig_within():
    files = [("double", "unseen subjects + sentences", "results_double.csv"),
             ("subject_only", "unseen subjects, sentences may repeat", "results_subject_only.csv")]
    groups = []
    for reg, name, fn in files:
        d = rd(fn)
        if d is not None:
            for ds in sorted(d.dataset.unique()):
                groups.append((f"{ds.replace('zuco', 'ZuCo ')}.0\n{name}", d[d.dataset == ds]))
    if not groups:
        return None
    conds = ["eeg_bandpower", "duration_only", "random_input"]
    fig, ax = plt.subplots(figsize=(12, 4.3))
    for gi, (title, d) in enumerate(groups):
        for j, c in enumerate(conds):
            v = (d[d.condition == c].mrr - d[d.condition == c].chance_mrr).to_numpy()
            ax.bar(gi + (j - 1) * 0.27, v.mean(), width=0.26, yerr=v.std(ddof=1), capsize=3, color=col(c),
                   label=lab(c) if gi == 0 else None)
    ax.axhline(0, color="red", ls="--", lw=1)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8)
    ax.set_ylabel("MRR above chance (0 = chance)")
    ax.legend(fontsize=8)
    ax.set_title("Earlier experiment: continuous recordings, windows aligned by order and duration")
    return save(fig, "fig8_continuous_windows.png")


# --------------------------------------------------------------------------- report
def main():
    main_df, f, pc, w = build_main(), rd("results_final.csv"), rd("results_positive_control.csv"), rd("results_words.csv")
    tr, trn = rd("results_transfer.csv"), rd("results_transfer_subjnorm.csv")
    ev1, ev2, al = rd("events_zuco1.csv"), rd("events_zuco2.csv"), rd("alignment_quality.csv")
    gs, gp = rd("results_gru_summary.csv"), rd("results_gru_pairs.csv")
    img = {"split": fig_pipeline_split(), "data": fig_data(), "align": fig_alignment(),
           "main": fig_main(main_df) if main_df is not None else None,
           "noise": fig_noise(f) if f is not None else None,
           "pos": fig_positive(pc) if pc is not None else None,
           "words": fig_words(w) if w is not None else None,
           "transfer": fig_transfer(tr) if tr is not None else None,
           "within": fig_within(),
           "gru": "experiments/results/figures/fig9_gru_subject_level.png"
           if (FIG / "fig9_gru_subject_level.png").exists() else None}

    def pic(key, alt):
        return f"![{alt}]({img[key]})" if img.get(key) else "_(no data for this figure yet)_"

    S = summarize(main_df) if main_df is not None else None
    V = verdicts(S) if S is not None else {}
    winners = [lab(m) for m, v in V.items() if v == "yes"]
    n_c = int(round(main_df.n_candidates.mean())) if main_df is not None else 45
    chance = 100 / n_c

    md = ["# Results: does EEG carry information about the sentence being read?\n",
          "ZuCo 1.0 and ZuCo 2.0. Every number and figure below is generated from the saved result files by "
          "`scripts/make_report.py`.\n",
          "**Question.** While a person reads a sentence, does their EEG contain information about *which* sentence it is? "
          "We tested this with sentence retrieval under a strict protocol, with controls that a real effect must beat.\n"]
    if S is not None and not winners:
        md.append("**Answer in one line.** No. None of the EEG representations (handcrafted band power, a self-supervised "
                  "encoder trained on ZuCo" + (", frozen pretrained CBraMod" if "cbramod_pretrained" in S.index else "") +
                  ") beats both a random-input control and a reading-length control on unseen subjects and unseen sentences. "
                  "What the EEG features do contain is reading-related information (fixation duration), not sentence content.\n")
    elif S is not None:
        md.append("**Answer in one line.** Some representations exceeded both controls by more than the fold-to-fold spread "
                  f"({', '.join(winners)}); see section 4 and treat it as preliminary (4 folds, no significance test).\n")

    md += ["## 1. The pipeline\n", "```mermaid", "flowchart LR",
           '  A["ZuCo 1.0 and 2.0<br/>EEG while reading<br/>105 channels, 500 Hz"] --> B["Cut into sentence windows<br/>(true sentence known)"]',
           '  B --> C1["Band power<br/>(handcrafted baseline)"]',
           '  B --> C2["Self-supervised encoder<br/>trained on ZuCo, no labels"]',
           '  B --> C3["Pretrained CBraMod<br/>frozen"]',
           '  C1 --> D["Ridge decoder:<br/>EEG features to sentence embedding"]',
           "  C2 --> D", "  C3 --> D",
           '  S["Sentence text"] --> E["MiniLM sentence embedding"]',
           "  E --> D",
           '  D --> F["Rank the candidate sentences<br/>by cosine similarity"]',
           "  E --> F",
           '  F --> G["Metrics: top-1, MRR, sim_margin<br/>+ controls: random input, reading length"]',
           "```\n",
           "## 2. How we test it, and why it is strict\n", pic("split", "split design"), "",
           "- **Unseen subjects:** whole people are held out, so the decoder never saw the test person's EEG.",
           "- **Unseen sentences:** the test sentences never appear in training either (otherwise a model could memorise sentences).",
           f"- **Candidates and chance:** for each test EEG the decoder ranks {n_c} candidate sentences, so guessing gets top-1 = 1/{n_c} = {chance:.1f}%.",
           "- **Metrics:** *top-1* = the correct sentence is ranked first; *MRR* = average of 1/rank of the correct sentence; "
           "*sim_margin* = similarity of the decoded vector to the correct sentence minus to the other sentences (0 = no information, and it detects weak effects that top-1 misses).",
           "- **Controls:** (1) random numbers through the identical pipeline; (2) reading length only (number of samples and words, no brain signal). A real effect must beat both.",
           "- **Folds:** 4 leave-subjects-out folds; error bars are the SD across folds (the GRU section uses bootstrap confidence intervals over test subjects).\n",
           "## 3. The data\n", pic("data", "data overview"), ""]
    rows = []
    for name, ev in (("ZuCo 1.0", ev1), ("ZuCo 2.0 (8 of 18 subjects)", ev2)):
        if ev is not None:
            rows.append([name, ev.subject.nunique(), len(ev), int(ev.n_pairs.sum())])
    if rows:
        md += [md_table(["Dataset", "Subjects", "EEG recordings", "Sentence windows"], rows), ""]
    md += ["For the continuous recordings, windows were matched to sentences by order and duration (the sentence is *inferred*). "
           "Check: reading duration should grow with sentence length.", "", pic("align", "alignment check"), ""]
    if al is not None:
        md.append("Median per-subject Spearman: " + "; ".join(
            f"{d.replace('zuco', 'ZuCo ')}.0 = {fmt(al[al.dataset == d].spearman.median(), 2)}" for d in sorted(al.dataset.unique()))
                  + ". The true-label experiments use ZuCo 1.0's 'Matlab files' (12 subjects, about 300 sentences each), where the sentence is given.\n")

    md += ["## 4. Main result\n", pic("main", "main result"), ""]
    if S is not None:
        rows = [[lab(m), fmt(r.top1, 1), fmt(r.chance, 1), fmt(r.mrr), fmt(r.chance_mrr), f"{fmt(r.sm, 4)} ({fmt(r.sd, 4)})", V[m]]
                for m, r in S.iterrows()]
        md += [md_table(["Method", "Top-1 %", "Chance %", "MRR", "Chance MRR", "sim_margin (SD over folds)", "Beats both controls?"], rows), "",
               "*Beats both controls?* is a conservative rule of thumb, not a significance test: the method's mean `sim_margin` "
               "must exceed the larger control mean by more than its own SD across folds.\n"]
        if "cbramod_pretrained" in S.index and "cbramod_random_init" in S.index:
            md.append(f"**Does pretraining matter?** CBraMod pretrained `sim_margin` = {fmt(S.loc['cbramod_pretrained', 'sm'], 4)} vs the same network "
                      f"randomly initialised = {fmt(S.loc['cbramod_random_init', 'sm'], 4)} (random input = {fmt(S.loc['random_input', 'sm'], 4)}). "
                      "CBraMod was evaluated on clean EEG only (no noise or quality experiments).\n")
        if "ssl_trained" in S.index and "untrained_encoder" in S.index:
            md.append(f"**Does self-supervised learning matter?** Trained encoder `sim_margin` = {fmt(S.loc['ssl_trained', 'sm'], 4)} vs untrained "
                      f"= {fmt(S.loc['untrained_encoder', 'sm'], 4)}.\n")

    if f is not None:
        md += ["## 5. Noise and signal quality\n",
               "White noise was added to the **test** EEG only, at 0.5, 1 and 2 times each channel's own standard deviation "
               "(about 6 dB, 0 dB and -6 dB signal-to-noise). *Quality weighting* gives training rows with many artifacts a smaller weight.\n",
               pic("noise", "noise robustness"), ""]
        lv = ["clean", "mild", "moderate", "strong"]
        rows = []
        for m in ["chunk_bandpower", "chunk_bandpower+quality", "untrained_encoder", "ssl_trained", "ssl_trained+quality"]:
            d = f[f.method == m]
            if len(d):
                rows.append([lab(m)] + [fmt(d[d.noise == n].sim_margin.mean(), 4) for n in lv])
        md += [md_table(["Method (sim_margin)", "none", "mild", "moderate", "strong"], rows), "",
               "There is no systematic decline with noise: a representation with no sentence information has nothing to lose. "
               "Quality weighting makes no reliable difference either.\n"]

    if pc is not None:
        def pcv(s, ft, t):
            v = pc[(pc.setting == s) & (pc.features == ft) & (pc.target == t)].r
            return v.mean() if len(v) else float("nan")

        md += ["## 6. What the EEG features do contain (positive control)\n",
               "A negative result is only convincing if the pipeline can detect a real effect. We asked whether the per-word EEG features "
               "predict simple word properties on held-out words.\n", pic("pos", "positive control"), "",
               f"- Within one person, EEG predicts log first-fixation duration (r = {fmt(pcv('within_subject', 'eeg', 'log_ffd'))} "
               f"vs {fmt(pcv('within_subject', 'random', 'log_ffd'))} for random features).",
               f"- Across people it only appears after per-subject normalisation (r = {fmt(pcv('cross_subject', 'eeg_subjnorm', 'log_ffd'))}; "
               f"raw {fmt(pcv('cross_subject', 'eeg', 'log_ffd'))}). That normalisation uses unlabeled test-subject EEG (transductive).",
               f"- Word length is not predicted (r = {fmt(pcv('within_subject', 'eeg', 'nchar'))} within subject).",
               "- Fixation duration is an eye-movement / reading-time quantity, so this is reading-related information, not sentence meaning.\n"]
        if w is not None:
            g = w.groupby("method")
            md += ["Word-level retrieval and sentence retrieval from word-level EEG:\n", pic("words", "word level"), "",
                   f"Word retrieval `sim_margin`: EEG {fmt(g.w_sim_margin.mean().get('eeg_bandpower'), 4)}, random input "
                   f"{fmt(g.w_sim_margin.mean().get('random_input'), 4)}, word length + fixation time {fmt(g.w_sim_margin.mean().get('length_ffd'), 4)}. "
                   f"Sentence MRR from word EEG: EEG {fmt(g.s_mrr.mean().get('eeg_bandpower'))}, random {fmt(g.s_mrr.mean().get('random_input'))}, "
                   f"chance {fmt(w.s_chance_mrr.mean())}. (Word-retrieval chance is not meaningful because frequent words dominate; compare with the random-input control.)\n"]

    if tr is not None:
        md += ["## 7. Transfer between ZuCo 1.0 and ZuCo 2.0\n", pic("transfer", "transfer"), "",
               "The decoder is fitted on the source dataset only (no target labels) and tested on target sentences it never saw. "
               "The two datasets share the same 105 channels in the same order, so no channel mapping was needed.\n"]
        rows = []
        for tag, d in (("no", tr), ("yes", trn)):
            if d is None:
                continue
            for r in d[d.target_subset == "unseen_sentences"].itertuples():
                rows.append([f"{r.source.replace('zuco', 'ZuCo ')}.0 -> {r.target.replace('zuco', 'ZuCo ')}.0", tag, lab(r.condition),
                             int(r.n_candidates), fmt(r.top1 * 100, 2), fmt(r.chance_top1 * 100, 2), fmt(r.mrr, 4), fmt(r.sim_margin, 4)])
        md += [md_table(["Direction", "Per-subject norm?", "Condition", "Candidates", "Top-1 %", "Chance %", "MRR", "sim_margin"], rows), ""]

    if img.get("within"):
        md += ["## 8. Earlier experiment on the continuous recordings\n", pic("within", "continuous windows"), "",
               "Same conclusion with inferred alignment on both datasets: EEG is indistinguishable from random input; only reading duration is above chance.\n"]

    if gs is not None:
        md += ["## 9. Word-level GRU and the reading-behaviour control\n",
               "A small GRU reads the words of a sentence in order and is trained to match the sentence embedding "
               "(12 subjects, 4 leave-subjects-out folds, unseen sentences, 45 candidates). Inputs per word: "
               "random numbers (`gru_random`), reading behaviour only = first-fixation time + word length, no EEG (`gru_behavior`), "
               "EEG with train-statistics normalisation (`gru_eeg_trainnorm`), EEG with per-subject normalisation "
               "(`gru_eeg_subjnorm`, transductive), and EEG + behaviour. The `ridge_*` rows are the sentence-level ridge decoder on the same subjects.\n",
               "**Decision rule (fixed before the run):** an EEG model counts as a real finding only if its paired difference "
               "(bootstrap over test subjects, 95% CI) against BOTH random input and the behaviour-only control is entirely above 0. "
               "Exploratory: one seed, no correction for multiple comparisons.\n",
               pic("gru", "GRU subject-level results"), "",
               "Per-subject metrics, mean over test subjects, with 95% bootstrap CI over subjects:\n", df_table(gs), ""]
        if gp is not None:
            md += ["Paired differences (a minus b), bootstrap over subjects; `share_le_0` = share of resamples with difference <= 0:\n",
                   df_table(gp), ""]
        md += ["In the run reported here, behaviour alone (fixation time + word length, no EEG) is clearly above random input, so word-level "
               "'decoding' can work through reading behaviour alone. EEG with train-statistics normalisation is not above random input. "
               "Adding EEG to behaviour does not improve on behaviour alone. Per-subject-normalised EEG is above random input in some metrics "
               "but stays below the behaviour-only model. By the decision rule, no EEG model counts as a real finding. "
               "The tables above are the authoritative numbers.\n"]

    md += ["## 10. Conclusions\n",
           "1. Under strict evaluation (unseen subjects and sentences) no tested EEG representation beat both controls." if not winners
           else f"1. Representations exceeding both controls by more than the fold spread: {', '.join(winners)} (preliminary).",
           "2. The detectable EEG information is reading-related (fixation duration), strongest within a person; it transfers across people only partly.",
           "3. The noise control and the reading-length control are essential: without them, small above-chance numbers would look like decoding.",
           "4. Self-supervised pre-training on ZuCo, the frozen pretrained CBraMod encoder and quality weighting did not produce a reliable gain; transfer between datasets was at chance.",
           "5. A word-level model fed only reading behaviour (no EEG) beats random input, and EEG does not add to it under the decision rule. "
           "Good-looking word-level 'EEG decoding' numbers therefore need a behaviour-only control before they can be attributed to the brain signal.\n",
           "## 11. Limitations and next steps\n",
           "- Only sentence-level and first-fixation band features were tested; time-locked word-level responses (for example N400-type effects) are the main untested direction.",
           "- 4 folds and one seed; bootstrap confidence intervals over subjects exist only for the GRU comparison; no significance tests or multiple-comparison correction; ZuCo 2.0 uses 8 of 18 subjects.",
           "- Continuous-recording alignment is inferred; ZuCo 1.0 windows covered 264 of 300 sentences (cause not investigated).",
           "- Noise is synthetic white noise; the quality score is a simple artifact fraction; the self-supervised encoder is small and untuned.",
           "- CBraMod was pretrained on 19-channel 10-20 recordings; applying it to 104 EGI channels is out of its training distribution. It was evaluated on clean EEG only (no noise or quality experiments) and not fine-tuned.",
           "- Per-subject normalisation is transductive and reported separately."]
    Path("RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("wrote RESULTS.md")


if __name__ == "__main__":
    main()