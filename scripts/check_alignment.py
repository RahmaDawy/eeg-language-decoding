"""Test: sentence windows = 10->11 plus 12->13, in time order. Read-only."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio

from eegdecode.data.load_data import list_recordings
from eegdecode.data.probe_eeg import read_events, sentence_windows

SRATE = 500.0


def all_sentence_windows(path):
    with h5py.File(path, "r") as f:
        ev = read_events(f)
    normal, p1 = sentence_windows(ev, "10", "11")
    quest, p2 = sentence_windows(ev, "12", "13")
    rows = sorted([(a, b, "10-11") for a, b in normal] + [(a, b, "12-13") for a, b in quest])
    return rows, p1 + p2


def word_counts(path):
    wb = sio.loadmat(str(path))["wordbounds"][0]
    return [int(np.atleast_2d(x).shape[0]) for x in wb]


def first(root, pattern):
    hits = sorted(Path(root).rglob(pattern))
    return hits[0] if hits else None


def check(label, root, wb_pattern, expected_total):
    wc = word_counts(first(root, wb_pattern))
    totals, problems_total = {}, 0
    print(f"\n=== {label} (wordbounds: {len(wc)} sentences) ===")
    for it in list_recordings(root):
        rows, problems = all_sentence_windows(it["path"])
        problems_total += problems
        totals[it["subject"]] = totals.get(it["subject"], 0) + len(rows)
        if it["block"] == 1 and len(rows) <= len(wc):
            dur = np.array([(b - a) / SRATE for a, b, _ in rows])
            r = np.corrcoef(dur, wc[: len(rows)])[0, 1]
            spw = np.median(dur / np.array(wc[: len(rows)]))
            print(f'{it["subject"]} block 1: {len(rows)} sentences '
                  f'({sum(k == "12-13" for _, _, k in rows)} via 12-13) | '
                  f'corr(duration, word count) = {r:.2f} | median {spw:.2f} s/word')
    print(f"unpaired events: {problems_total}")
    print(f"sentences per subject, all blocks (expected about {expected_total}):")
    print("  ", totals)


def csv_peek():
    print("\n=== CSV files under data/ (excluding data/raw) ===")
    found = False
    for p in sorted(Path("data").rglob("*.csv")):
        if "raw" in p.parts:
            continue
        found = True
        df = pd.read_csv(p, encoding="latin-1")
        print(f"\n{p}: {len(df)} rows, columns = {list(df.columns)}")
        print(df.head(3).to_string(max_colwidth=50))
    if not found:
        print("none found")


if __name__ == "__main__":
    check("ZuCo 1", "data/raw/zuco1", "wordbounds_NR*.mat", 300)
    check("ZuCo 2", "data/raw/zuco2", "wordbounds_NR1.mat", 349)
    csv_peek()