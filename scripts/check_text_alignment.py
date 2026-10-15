"""Check that sentence CSVs line up with EEG sentence windows. Read-only."""
from pathlib import Path

import numpy as np
import pandas as pd

from check_alignment import all_sentence_windows, first, word_counts
from eegdecode.data.load_data import list_recordings

Z1 = "data/raw/zuco1"
Z2 = "data/raw/zuco2"


def eeg_counts(root, subject):
    out = {}
    for it in list_recordings(root):
        if it["subject"] == subject:
            out[it["block"]] = len(all_sentence_windows(it["path"])[0])
    return dict(sorted(out.items()))


print("=== ZuCo 1: text CSV vs wordbounds (300 sentences) ===")
df = pd.read_csv(r"data\raw\zuco1\task_materials\relations_labels_task2.csv", encoding="latin-1")
wc_text = np.array([len(str(s).split()) for s in df["sentence"]])
wc_wb = np.array(word_counts(first(Z1, "wordbounds_NR*.mat")))
d = wc_text - wc_wb
print(f"rows: {len(df)} | exact match: {(d == 0).mean():.2f} | within 1 word: {(abs(d) <= 1).mean():.2f} "
      f"| corr: {np.corrcoef(wc_text, wc_wb)[0, 1]:.3f}")
bad = np.where(abs(d) > 1)[0]
print("rows differing by more than 1 word (index, csv, wordbounds):", [(int(i), int(wc_text[i]), int(wc_wb[i])) for i in bad[:15]])
print("ZuCo 1 sentences per block (subject ZAB):", eeg_counts(Z1, "ZAB"))

print("\n=== ZuCo 2: nr_k.csv rows vs EEG sentences (subject YHS) ===")
eeg = eeg_counts(Z2, "YHS")
for k in range(1, 8):
    p = Path(rf"data\raw\zuco2\task_materials\nr_{k}.csv")
    t = pd.read_csv(p, sep=";", header=None, encoding="latin-1")
    t = t.dropna(subset=[2])
    print(f"block {k}: csv rows = {len(t)} | EEG sentences = {eeg.get(k)}")

t1 = pd.read_csv(r"data\raw\zuco2\task_materials\nr_1.csv", sep=";", header=None, encoding="latin-1").dropna(subset=[2])
print("\nblock 1 csv word counts :", [len(str(s).split()) for s in t1[2]])
print("block 1 wordbounds      :", word_counts(first(Z2, "wordbounds_NR1.mat")))