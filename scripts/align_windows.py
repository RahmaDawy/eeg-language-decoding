"""Align EEG windows to sentences by ORDER + DURATION and report how good the alignment is.

Windows (trigger 10->11) are fewer than the sentences in every block, so they cannot simply be zipped.
All subjects read the same sentence sequence, so the median window duration across subjects is a clean
per-window signal. We find the order-preserving matching (sentences may be skipped) that best fits
log(duration) ~ a + b*log(n_words). This is an INFERRED alignment: its quality is printed, not assumed.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

P = "data/processed"
GAP_WINDOW = 5.0  # cost of leaving a window unmatched (large: we want every window matched)


def align(x, y, iters=6):
    n, m = len(x), len(y)
    j0 = np.round(np.linspace(0, m - 1, n)).astype(int)
    b, a = np.polyfit(y[j0], x, 1)
    for _ in range(iters):
        C = (x[:, None] - a - b * y[None, :]) ** 2
        D = np.zeros((n + 1, m + 1))
        B = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0 match, 1 skip sentence, 2 skip window
        D[1:, 0] = GAP_WINDOW * np.arange(1, n + 1)
        B[1:, 0] = 2
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                opts = (D[i - 1, j - 1] + C[i - 1, j - 1], D[i, j - 1], D[i - 1, j] + GAP_WINDOW)
                k = int(np.argmin(opts))
                D[i, j], B[i, j] = opts[k], k
        i, j, mi, mj = n, m, [], []
        while i > 0 and j > 0:
            k = B[i, j]
            if k == 0:
                mi.append(i - 1); mj.append(j - 1); i -= 1; j -= 1
            elif k == 1:
                j -= 1
            else:
                i -= 1
        mi, mj = np.array(mi[::-1]), np.array(mj[::-1])
        if len(mi) >= 5:
            b, a = np.polyfit(y[mj], x[mi], 1)
    return mi, mj, a, b


def main():
    w = pd.read_csv(f"{P}/windows.csv")
    s = pd.read_csv(f"{P}/sentences.csv").reset_index(drop=True)
    if "n_words" not in s or s.n_words.isna().any():
        s["n_words"] = s.text.str.split().str.len()
    maps, slopes = [], {}
    for ds in sorted(w.dataset.unique()):
        wd, sd = w[w.dataset == ds], s[s.dataset == ds]
        blocks = sorted(wd.block.unique())
        per_block = bool(sd.block.notna().all()) and set(blocks) <= set(sd.block.unique())
        units = [(wd[wd.block == b], sd[sd.block == b]) for b in blocks] if per_block else [(wd, sd)]
        for wu, su in units:
            med = wu.groupby(["block", "win_idx"]).duration_s.median().sort_index()
            x = np.log(med.values)
            y = np.log(su.n_words.clip(lower=1).values.astype(float))
            mi, mj, a, b = align(x, y)
            idx = med.index[mi]
            maps.append(pd.DataFrame({"dataset": ds, "block": idx.get_level_values(0),
                                      "win_idx": idx.get_level_values(1), "sent_row": su.index.values[mj]}))
            slopes.setdefault(ds, []).append(b)
        print(f"{ds}: aligned per block = {per_block}; slope log(dur)/log(words): "
              f"{np.round(slopes[ds], 2).tolist()}")
    m = pd.concat(maps)
    w = w.drop(columns=["sent_row", "aligned"], errors="ignore").merge(m, on=["dataset", "block", "win_idx"], how="left")
    w["aligned"] = w.sent_row.notna()
    w["sent_row"] = w.sent_row.fillna(-1).astype(int)
    ok = w.aligned.to_numpy()
    rows_ = w.sent_row.to_numpy()
    sid = np.full(len(w), None, dtype=object)
    sid[ok] = s.sent_id.to_numpy()[rows_[ok]]
    nw = np.full(len(w), np.nan)
    nw[ok] = s.n_words.to_numpy()[rows_[ok]]
    w["sent_id"] = sid
    w["n_words"] = nw

    rows, rng = [], np.random.default_rng(0)
    print("\nALIGNMENT QUALITY (Spearman between window duration and sentence word count)")
    for ds in sorted(w.dataset.unique()):
        a = w[(w.dataset == ds) & w.aligned]
        rhos = []
        for sj, g in a.groupby("subject"):
            rho = spearmanr(np.log(g.duration_s), np.log(g.n_words.clip(lower=1)))[0]
            rhos.append(rho)
            rows.append(dict(dataset=ds, subject=sj, n_windows=len(g), spearman=rho))
        med = a.groupby(["block", "win_idx"]).agg(d=("duration_s", "median"), n=("n_words", "first"))
        null = spearmanr(np.log(med.d), rng.permutation(np.log(med.n.clip(lower=1))))[0]
        print(f"{ds}: aligned {int(len(a))}/{int((w.dataset == ds).sum())} windows | "
              f"per-subject Spearman median {np.median(rhos):.2f} (min {np.min(rhos):.2f}, max {np.max(rhos):.2f}) | "
              f"shuffled-sentences null {null:.2f}")
    w.to_csv(f"{P}/windows_aligned.csv", index=False)
    pd.DataFrame(rows).to_csv("experiments/results/metrics/alignment_quality.csv", index=False)
    print("\nsaved data/processed/windows_aligned.csv and experiments/results/metrics/alignment_quality.csv")


if __name__ == "__main__":
    main()