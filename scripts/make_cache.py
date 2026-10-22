"""Convert each ZuCo 1.0 'Matlab files' subject (results<SUBJ>_NR.mat, ~600 MB) into a small cache file
data/processed/gt/<SUBJ>.npz, so every later experiment loads in seconds. True sentence labels come with the file.
Per sentence : text, authors' sentence-level band means (8 bands x 105 ch = 840), our log band power from rawData,
               number of samples / words, omission rate, artifact fraction (label-free quality indicator), and up to
               4 raw chunks (1.5 s at 100 Hz, 105 ch) for the self-supervised encoder.
Per word     : text, first-fixation band means (840), first-fixation duration (only words that have them).
"""
import argparse
import glob
from pathlib import Path

import numpy as np
from scipy import io as sio
from scipy.signal import decimate

from eegdecode.models.baseline import band_power_features

BANDS = ["t1", "t2", "a1", "a2", "b1", "b2", "g1", "g2"]
CHUNK, MAX_CHUNKS, DECIM = 150, 4, 5  # 150 samples at 100 Hz = 1.5 s
OUT = Path("data/processed/gt")


def band_vec(obj, prefix):
    """Concatenate <prefix>_<band> (105 values each) over the 8 bands; None if anything is missing/invalid."""
    parts = []
    for b in BANDS:
        v = np.asarray(getattr(obj, f"{prefix}_{b}"), dtype=np.float64).ravel()
        if v.size != 105:
            return None
        parts.append(v)
    out = np.concatenate(parts)
    return out if np.isfinite(out).all() else None


def process(path):
    sd = sio.loadmat(path, struct_as_record=False, squeeze_me=True, variable_names=["sentenceData"])["sentenceData"]
    S = {k: [] for k in ("idx", "text", "official", "ours", "nsamp", "nwords", "omission", "artifact")}
    W = {k: [] for k in ("sent", "text", "feat", "ffd")}
    chunks, chunk_sent = [], []
    for i, s in enumerate(np.atleast_1d(sd)):
        try:
            raw = np.asarray(s.rawData, dtype=np.float64)
            off = band_vec(s, "mean")
        except AttributeError:
            continue
        if raw.ndim != 2 or raw.shape[0] != 105 or raw.shape[1] < 500 or not np.isfinite(raw).all() or off is None:
            continue
        x = raw.T  # (samples, 105)
        row = len(S["idx"])
        robust_sd = np.median(np.abs(x)) / 0.6745
        S["idx"].append(i)
        S["text"].append(str(s.content))
        S["official"].append(off)
        S["ours"].append(band_power_features(x, 500.0))
        S["nsamp"].append(x.shape[0])
        S["nwords"].append(int(np.size(s.word)))
        S["omission"].append(float(s.omissionRate))
        S["artifact"].append(float(np.mean(np.abs(x) > 4 * robust_sd)))
        xd = decimate(x, DECIM, axis=0, zero_phase=True)
        for c in range(min(MAX_CHUNKS, len(xd) // CHUNK)):
            chunks.append(xd[c * CHUNK:(c + 1) * CHUNK].T.astype(np.float16))
            chunk_sent.append(row)
        for w in np.atleast_1d(s.word):
            try:
                f = band_vec(w, "FFD")
                ffd = float(np.asarray(w.FFD).ravel()[0])
            except (AttributeError, IndexError, TypeError, ValueError):
                continue
            if f is None or not np.isfinite(ffd):
                continue
            W["sent"].append(row)
            W["text"].append(str(w.content))
            W["feat"].append(f)
            W["ffd"].append(ffd)
    return S, W, chunks, chunk_sent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+",
                    default=sorted(glob.glob("data/raw/zuco1/task2 - NR/Matlab files/results*_NR.mat")))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for p in a.files:
        subj = Path(p).stem.replace("results", "").replace("_NR", "")
        dest = OUT / f"{subj}.npz"
        if dest.exists() and not a.force:
            print(f"{subj}: cache exists, skipping")
            continue
        print(f"{subj}: loading (about a minute) ...", flush=True)
        S, W, chunks, chunk_sent = process(p)
        np.savez_compressed(
            dest,
            idx=np.array(S["idx"]), text=np.array(S["text"]), official=np.array(S["official"], dtype=np.float32),
            ours=np.array(S["ours"], dtype=np.float32), nsamp=np.array(S["nsamp"]), nwords=np.array(S["nwords"]),
            omission=np.array(S["omission"]), artifact=np.array(S["artifact"]),
            chunks=np.stack(chunks), chunk_sent=np.array(chunk_sent),
            w_sent=np.array(W["sent"]), w_text=np.array(W["text"]),
            w_feat=np.array(W["feat"], dtype=np.float32), w_ffd=np.array(W["ffd"]),
        )
        print(f"  {subj}: {len(S['idx'])} sentences, {len(W['sent'])} words, {len(chunks)} chunks "
              f"-> {dest} ({dest.stat().st_size / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()