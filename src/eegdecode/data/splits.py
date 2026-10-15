"""Subject-wise (and optionally sentence-wise) splits with explicit leakage checks.

Two regimes:
  * subject_only: test subjects unseen, test SENTENCES MAY ALSO APPEAR IN TRAINING (weaker claim).
  * double hold-out: test subjects AND test sentences unseen (stricter; recommended primary result).
Rows whose subject and sentence land in different partitions are labelled 'drop' and never used.
"""
import hashlib
import itertools
import re

import numpy as np
import pandas as pd

NAMES = ("train", "val", "test")


def sentence_key(text: str) -> str:
    """Group key for a sentence: identical after lowercasing and removing punctuation/spaces."""
    norm = re.sub(r"[^a-z0-9]", "", text.lower())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]


def split_items(items, n_val, n_test, seed):
    """Deterministically split unique items into train/val/test by counts."""
    uniq = sorted(set(items))
    if n_val < 1 or n_test < 1 or n_val + n_test >= len(uniq):
        raise ValueError(f"cannot split {len(uniq)} items with n_val={n_val}, n_test={n_test}")
    rng = np.random.RandomState(seed)
    shuffled = [uniq[i] for i in rng.permutation(len(uniq))]
    return {
        "test": shuffled[:n_test],
        "val": shuffled[n_test:n_test + n_val],
        "train": shuffled[n_test + n_val:],
    }


def subject_folds(subjects, k, seed):
    """Leave-subjects-out folds: fold i tests on group i, validates on group i+1, trains on the rest."""
    if k < 3:
        raise ValueError("k must be >= 3")
    uniq = sorted(set(subjects))
    rng = np.random.RandomState(seed)
    order = [uniq[i] for i in rng.permutation(len(uniq))]
    groups = [order[i::k] for i in range(k)]
    folds = []
    for i in range(k):
        j = (i + 1) % k
        folds.append({
            "test": groups[i],
            "val": groups[j],
            "train": [s for g, grp in enumerate(groups) if g not in (i, j) for s in grp],
        })
    return folds


def assign_split(df, subject_split, sentence_split=None, subject_col="subject", sentence_col="sentence_id"):
    out = pd.Series("drop", index=df.index, dtype=object)
    for name in NAMES:
        mask = df[subject_col].isin(subject_split[name])
        if sentence_split is not None:
            mask &= df[sentence_col].isin(sentence_split[name])
        out.loc[mask] = name
    return out


def check_no_leakage(df, split, subject_col="subject", sentence_col="sentence_id", sentence_disjoint=True):
    """Raise AssertionError if partitions share subjects (or sentences when sentence_disjoint)."""
    problems = []
    subs = {n: set(df.loc[split == n, subject_col]) for n in NAMES}
    sents = {n: set(df.loc[split == n, sentence_col]) for n in NAMES}
    for a, b in itertools.combinations(NAMES, 2):
        if subs[a] & subs[b]:
            problems.append(f"subjects shared between {a} and {b}: {sorted(subs[a] & subs[b])[:5]}")
        if sentence_disjoint and sents[a] & sents[b]:
            problems.append(f"{len(sents[a] & sents[b])} sentences shared between {a} and {b}")
    if problems:
        raise AssertionError("; ".join(problems))
    return {n: int((split == n).sum()) for n in (*NAMES, "drop")}


def check_no_duplicate_examples(df, cols=("subject", "block", "win_idx")):
    dup = df.duplicated(list(cols), keep=False)
    if dup.any():
        raise AssertionError(f"{int(dup.sum())} duplicated examples on {cols}")