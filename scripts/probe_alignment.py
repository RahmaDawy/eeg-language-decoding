"""Read-only alignment probe: windows per block, word counts, ET messages, CSV heads."""
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio

from eegdecode.data.load_data import list_recordings
from eegdecode.data.probe_eeg import read_events, sentence_windows


def lengths(path):
    with h5py.File(path, "r") as f:
        ev = read_events(f)
    wins, problems = sentence_windows(ev)
    return np.array([b - a for a, b in wins], dtype=float), problems


def word_counts(path):
    wb = sio.loadmat(str(path))["wordbounds"][0]
    return [int(np.atleast_2d(x).shape[0]) for x in wb]


def first(root, pattern):
    hits = sorted(Path(root).rglob(pattern))
    return hits[0] if hits else None


def windows_table(root, label, wc):
    print(f"\n=== {label}: windows per EEG file (wordbounds has {len(wc)} sentences) ===")
    for it in list_recordings(root):
        L, problems = lengths(it["path"])
        line = f'{it["subject"]} block {it["block"]} [{it["prefix"]}]: {len(L)} windows, {problems} problems'
        if it["block"] == 1 and len(L) <= len(wc):
            r = np.corrcoef(L, wc[: len(L)])[0, 1]
            line += f" | corr(duration, first {len(L)} word counts) = {r:.2f}"
        print(line)


def et_peek(path, label):
    print(f"\n=== {label}: eye-tracking file {path.name} ===")
    d = sio.loadmat(str(path), squeeze_me=True)
    print("data columns:", list(d["colheader"]))
    msgs = [str(m).strip() for m in np.atleast_1d(d["messages"])]
    kinds = Counter(m.split()[2] if len(m.split()) > 2 else "?" for m in msgs)
    print("message kinds:", kinds.most_common(12))
    other = [m for m in msgs if len(m.split()) > 2 and not m.split()[2].startswith("!")]
    print(f"{len(other)} messages not starting with '!' -> first 15:")
    for m in other[:15]:
        print("  ", m[:100])
    print("event (first 12 rows):")
    print(d["event"][:12])


def csv_peek():
    print("\n=== CSV files under data/ ===")
    for p in sorted(Path("data").rglob("*.csv")):
        df = pd.read_csv(p)
        print(f"\n{p}: {len(df)} rows, columns = {list(df.columns)}")
        print(df.head(3).to_string(max_colwidth=60))


if __name__ == "__main__":
    wb1 = first("data/raw/zuco1", "wordbounds_NR*.mat")
    wb2 = first("data/raw/zuco2", "wordbounds_NR1.mat")
    wc1, wc2 = word_counts(wb1), word_counts(wb2)
    print("ZuCo1 word counts, first 45 sentences:", wc1[:45])
    print("ZuCo2 word counts, block 1:", wc2)
    windows_table("data/raw/zuco1", "ZuCo 1", wc1)
    windows_table("data/raw/zuco2", "ZuCo 2", wc2)
    et_peek(first("data/raw/zuco1", "*NR1_corrected_ET.mat"), "ZuCo 1")
    et_peek(first("data/raw/zuco2", "*NR1_corrected_ET.mat"), "ZuCo 2")
    csv_peek()