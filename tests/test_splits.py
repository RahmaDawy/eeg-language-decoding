import pandas as pd
import pytest

from eegdecode.data.splits import assign_split, check_no_leakage, sentence_key, split_items, subject_folds


def make_df(n_sub=6, n_sent=30):
    return pd.DataFrame([dict(subject=f"S{s}", sentence_id=f"T{t}") for s in range(n_sub) for t in range(n_sent)])


def test_split_items_deterministic_and_disjoint():
    a = split_items(list("abcdefghij"), n_val=2, n_test=3, seed=7)
    assert a == split_items(list("abcdefghij"), n_val=2, n_test=3, seed=7)
    assert len(a["test"]) == 3 and len(a["val"]) == 2 and len(a["train"]) == 5
    assert set(a["train"]) | set(a["val"]) | set(a["test"]) == set("abcdefghij")
    assert not set(a["train"]) & set(a["test"])


def test_double_holdout_has_no_leakage():
    df = make_df()
    subj = split_items(df.subject, 1, 2, seed=0)
    sent = split_items(df.sentence_id, 5, 5, seed=0)
    counts = check_no_leakage(df, assign_split(df, subj, sent))
    assert counts["test"] == 2 * 5 and counts["val"] == 1 * 5
    assert counts["train"] > 0 and counts["drop"] > 0


def test_subject_only_mode_allows_shared_sentences_but_not_subjects():
    df = make_df()
    split = assign_split(df, split_items(df.subject, 1, 2, seed=0), None)
    check_no_leakage(df, split, sentence_disjoint=False)
    with pytest.raises(AssertionError):
        check_no_leakage(df, split, sentence_disjoint=True)


def test_leakage_detector_catches_shared_subject():
    df = make_df()
    subj = split_items(df.subject, 1, 2, seed=0)
    bad = assign_split(df, subj, None).copy()
    idx = df.index[df.subject == subj["train"][0]][0]
    bad.loc[idx] = "test"
    with pytest.raises(AssertionError):
        check_no_leakage(df, bad, sentence_disjoint=False)


def test_subject_folds_cover_each_subject_once_as_test():
    subs = [f"S{i}" for i in range(6)]
    folds = subject_folds(subs, k=3, seed=0)
    assert sorted(s for f in folds for s in f["test"]) == sorted(subs)
    for f in folds:
        parts = [set(f["train"]), set(f["val"]), set(f["test"])]
        assert not (parts[0] & parts[1]) and not (parts[0] & parts[2]) and not (parts[1] & parts[2])
        assert parts[0] | parts[1] | parts[2] == set(subs)


def test_sentence_key_ignores_case_spacing_punctuation():
    assert sentence_key("Henry Ford, Motor.") == sentence_key("henry  ford motor")
    assert sentence_key("Henry Ford") != sentence_key("Henry Ford Jr")