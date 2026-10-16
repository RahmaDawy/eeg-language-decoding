"""One pass over the raw EEG. Saves, row-aligned with data/processed/windows_aligned.csv:
  feat_{clean,mild,moderate,strong}.npy  band-power features (n_windows, 104*5). Noisy versions add Gaussian
                                         noise to the WINDOW before feature extraction:
                                         noise sd = level x per-channel std of that window (seeded, reproducible)
  artifact_frac.npy                      label-free quality indicator: share of samples beyond 4 robust SDs
                                         (lower = cleaner)
  chunks.npy / chunk_win.npy             up to 4 non-overlapping 2.4 s chunks per window at 100 Hz (float16) for
                                         the self-supervised encoder; chunk_win = row index into windows_aligned.csv
  skipped_windows.csv                    windows that were NOT processed (too short, or a ValueError) and why
The flat reference channel Cz is dropped (104 channels).
Windows shorter than MIN_SAMPLES (1 s at 500 Hz) are skipped: they cannot be real sentence-reading windows.
Skipped windows keep NaN features and are excluded downstream.
"""
import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import decimate

from eegdecode.data.load_data import channel_keep_idx, list_recordings, load_recording
from eegdecode.models.baseline import band_power_features

LEVELS = {"clean": 0.0, "mild": 0.5, "moderate": 1.0, "strong": 2.0}
CHUNK, MAX_CHUNKS, DECIM = 240, 4, 5  # 240 samples at 100 Hz = 2.4 s
MIN_SAMPLES = 500  # 1 s at 500 Hz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--limit", type=int, default=0, help="process only N recordings (smoke test)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    w = pd.read_csv("data/processed/windows_aligned.csv")
    n = len(w)
    recs = {}
    for ds in ("zuco1", "zuco2"):
        for r in list_recordings(f"data/raw/{ds}"):
            recs[(ds, r["subject"], r["block"])] = r["path"]

    feats = {lv: None for lv in LEVELS}
    art = np.full(n, np.nan)
    chunks, chunk_win = [], []
    skipped = []
    groups = list(w.groupby(["dataset", "subject", "block"]))
    if a.limit:
        groups = groups[:a.limit]

    for gi, ((ds, sub, blk), g) in enumerate(groups, 1):
        g = g.sort_values("win_idx")
        rec = load_recording(recs[(ds, sub, blk)], ds)
        if len(rec.windows) != len(g) or (g.win_idx.to_numpy() != np.arange(len(g))).any():
            raise SystemExit(f"{ds} {sub} {blk}: {len(rec.windows)} windows in EEG vs {len(g)} in windows_aligned.csv")
        keep = channel_keep_idx(rec.channels)
        n_skip_block = 0
        for wi, (x, row) in enumerate(zip(rec.windows, g.index.to_numpy())):
            x = x[:, keep].astype(np.float64)
            if x.shape[0] < MIN_SAMPLES:
                skipped.append((ds, sub, blk, wi, row, x.shape[0], "too short"))
                n_skip_block += 1
                continue
            try:
                robust_sd = np.median(np.abs(x)) / 0.6745
                a_val = float(np.mean(np.abs(x) > 4 * robust_sd))
                sd_c = x.std(axis=0, keepdims=True)
                f_all = {}
                for lv, k in LEVELS.items():
                    if k == 0:
                        xn = x
                    else:
                        rng = np.random.default_rng(zlib.crc32(f"{ds}{sub}{blk}{wi}{lv}".encode()))
                        xn = x + k * sd_c * rng.standard_normal(x.shape)
                    f_all[lv] = band_power_features(xn, rec.srate)
                xd = decimate(x, DECIM, axis=0, zero_phase=True)
                new_chunks = [xd[c * CHUNK:(c + 1) * CHUNK].T.astype(np.float16)
                              for c in range(min(MAX_CHUNKS, len(xd) // CHUNK))]
            except ValueError as e:
                skipped.append((ds, sub, blk, wi, row, x.shape[0], f"ValueError: {e}"))
                n_skip_block += 1
                continue
            # commit only after everything for this window succeeded
            art[row] = a_val
            for lv, f in f_all.items():
                if feats[lv] is None:
                    feats[lv] = np.full((n, f.size), np.nan, dtype=np.float32)
                feats[lv][row] = f
            for c in new_chunks:
                chunks.append(c)
                chunk_win.append(row)
        print(f"[{gi}/{len(groups)}] {ds} {sub} block {blk}: {len(g)} windows, skipped {n_skip_block}", flush=True)

    for lv in LEVELS:
        np.save(out / f"feat_{lv}.npy", feats[lv])
    np.save(out / "artifact_frac.npy", art)
    np.save(out / "chunks.npy", np.stack(chunks))
    np.save(out / "chunk_win.npy", np.array(chunk_win))
    sk = pd.DataFrame(skipped, columns=["dataset", "subject", "block", "win_idx", "row", "n_samples", "reason"])
    sk.to_csv(out / "skipped_windows.csv", index=False)
    print(f"\nsaved to {out}: windows with features {int(np.isfinite(feats['clean']).all(axis=1).sum())}/{n}, "
          f"chunks {len(chunks)}, artifact_frac median {np.nanmedian(art):.4f} (max {np.nanmax(art):.4f})")
    print(f"skipped windows: {len(sk)}")
    if len(sk):
        print(sk.groupby(["dataset", "subject", "reason"]).size().to_string())
        print(sk.n_samples.describe().round(1).to_string())


if __name__ == "__main__":
    main()