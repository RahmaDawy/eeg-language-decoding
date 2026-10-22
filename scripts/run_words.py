"""Word-level test on the ZuCo 1.0 caches (true word and sentence labels).

For every word the authors give 8 band means x 105 channels around the first fixation (840 values).
Test = unseen subjects AND unseen sentences (leave-subjects-out folds; leakage check per fold).
1) Word retrieval: ridge maps a word's EEG features to its word embedding; candidates = unique words of the test
   sentences; chance = 1 / n_candidates.
2) Sentence retrieval from word evidence: average the predicted word embeddings over a sentence's words (per subject),
   rank the test sentences by cosine to each sentence's mean word embedding; chance = 1 / n_test_sentences.
3) Positive control: can the same EEG features predict word length / first-fixation duration (Pearson r on test)?
Controls: length_ffd (word length + log fixation duration only, no EEG) and random_input (same decoder).
Alpha is chosen on validation subjects/sentences only. One run, settings fixed in advance.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from eegdecode.data.splits import assign_split, check_no_leakage, sentence_key, split_items, subject_folds
from eegdecode.decoding.text_encoder import embed_texts
from eegdecode.evaluation.metrics import candidate_index, cosine_matrix, retrieval_metrics

GT = Path("data/processed/gt")
OUT = Path("experiments/results/metrics")
ALPHAS = (100.0, 1e3, 1e4, 1e5)
KEEP = ["n_candidates", "top1", "chance_top1", "recall@10", "mrr", "chance_mrr", "median_rank", "sim_margin"]


def clean(w):
    return re.sub(r"[^a-z0-9']", "", str(w).lower())


def load():
    parts, feats = [], []
    for f in sorted(GT.glob("*.npz")):
        z = np.load(f)
        parts.append(pd.DataFrame({
            "subject": f.stem,
            "sent_text": z["text"][z["w_sent"]],
            "word": [clean(w) for w in z["w_text"]],
            "ffd": z["w_ffd"].astype(float),
        }))
        feats.append(z["w_feat"])
    d, X = pd.concat(parts, ignore_index=True), np.vstack(feats)
    keep = (d.word.str.len() > 0).to_numpy() & np.isfinite(d.ffd.to_numpy()) & (d.ffd.to_numpy() > 0)
    d, X = d[keep].reset_index(drop=True), X[keep]
    d["key"] = d.sent_text.map(sentence_key)
    d["nchar"] = d.word.str.len()
    return d, X


def cands(words, wemb):
    uniq, true = candidate_index(list(words))
    return np.stack([wemb[u] for u in uniq]), true


def safe_r(a, b):
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if np.isfinite(r) else -1.0


def pick(Ztr, Ytr, score):
    best, best_s = None, -np.inf
    for al in ALPHAS:
        m = Ridge(alpha=al).fit(Ztr, Ytr)
        s = score(m)
        if s > best_s:
            best, best_s = m, s
    return best


def evaluate(X, d, tr, va, te, wemb, sent_emb, tn, tf, pos):
    sc = StandardScaler().fit(X[tr])  # statistics from training rows only
    Ztr, Zva, Zte = (sc.transform(X[i]) for i in (tr, va, te))
    words = d.word.to_numpy()
    Ytr = np.stack([wemb[w] for w in words[tr]])
    cva, tva = cands(words[va], wemb)
    cte, tte = cands(words[te], wemb)
    model = pick(Ztr, Ytr, lambda m: retrieval_metrics(cosine_matrix(m.predict(Zva), cva), tva)["mrr"])
    out = {f"w_{k}": v for k, v in retrieval_metrics(cosine_matrix(model.predict(Zte), cte), tte).items() if k in KEEP}

    P = model.predict(Zte)
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    g = pd.DataFrame(P)
    g["s"], g["k"] = d.subject.to_numpy()[te], d.key.to_numpy()[te]
    m = g.groupby(["s", "k"]).mean()
    uniq, true = candidate_index([k for _, k in m.index])
    cs = np.stack([sent_emb[u] for u in uniq])
    out.update({f"s_{k}": v for k, v in retrieval_metrics(cosine_matrix(m.to_numpy(), cs), true).items() if k in KEEP})

    if pos:
        for name, tgt in (("r_nchar", tn), ("r_ffd", tf)):
            mm = pick(Ztr, tgt[tr], lambda q: safe_r(q.predict(Zva), tgt[va]))
            out[name] = safe_r(mm.predict(Zte), tgt[te])
    return out


def avg(ds):
    return {k: float(np.nanmean([x[k] for x in ds])) for k in ds[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random-seeds", type=int, default=2)
    ap.add_argument("--pilot", action="store_true", help="one fold only (smoke test)")
    a = ap.parse_args()

    d, X = load()
    first = d.drop_duplicates("key").set_index("key").sent_text.to_dict()
    sent_words = {k: [w for w in (clean(t) for t in text.split()) if w] for k, text in first.items()}
    vocab = sorted(set(d.word) | {w for ws in sent_words.values() for w in ws})
    wemb = dict(zip(vocab, embed_texts(vocab, cache_path="data/processed/gt_word_emb")))
    sent_emb = {}
    for k, ws in sent_words.items():
        v = np.mean([wemb[w] for w in ws], axis=0)
        sent_emb[k] = v / (np.linalg.norm(v) + 1e-12)

    tn = d.nchar.to_numpy(float)
    tf = np.log(d.ffd.to_numpy())
    L = np.column_stack([tn, tf]).astype(np.float32)
    subs = sorted(d.subject.unique())
    n_hold = max(2, int(0.15 * d.key.nunique()))
    sent_split = split_items(d.key, n_hold, n_hold, a.seed)
    folds = subject_folds(subs, min(a.folds, len(subs)), a.seed)
    folds = folds[:1] if a.pilot else folds
    print(f"{len(d)} word rows, {len(subs)} subjects, {d.key.nunique()} sentences, {len(vocab)} distinct words, "
          f"{len(folds)} fold(s), val/test sentences held out: {n_hold}/{n_hold}")

    rows = []
    for fi, fold in enumerate(folds):
        split = assign_split(d, fold, sent_split, "subject", "key")
        counts = check_no_leakage(d, split, "subject", "key")  # raises if subjects or sentences are shared
        sp = split.to_numpy()
        tr, va, te = (np.where(sp == nm)[0] for nm in ("train", "val", "test"))
        print(f"\nfold {fi}: test subjects {fold['test']}, word rows train/val/test "
              f"{counts['train']}/{counts['val']}/{counts['test']}", flush=True)
        rows.append({"fold": fi, "method": "eeg_bandpower", **evaluate(X, d, tr, va, te, wemb, sent_emb, tn, tf, True)})
        rows.append({"fold": fi, "method": "length_ffd", **evaluate(L, d, tr, va, te, wemb, sent_emb, tn, tf, False)})
        rnd = [evaluate(np.random.default_rng(1000 + r).standard_normal(X.shape, dtype=np.float32),
                        d, tr, va, te, wemb, sent_emb, tn, tf, True) for r in range(a.random_seeds)]
        rows.append({"fold": fi, "method": "random_input", **avg(rnd)})

    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    name = "results_words_pilot" if a.pilot else "results_words"
    res.to_csv(OUT / f"{name}.csv", index=False)
    cols = ["w_n_candidates", "w_top1", "w_chance_top1", "w_mrr", "w_chance_mrr", "w_sim_margin",
            "s_n_candidates", "s_top1", "s_chance_top1", "s_mrr", "s_chance_mrr", "s_sim_margin", "r_nchar", "r_ffd"]
    g = res.groupby("method", sort=False)
    print("\nMEAN OVER FOLDS (w_ = word retrieval, s_ = sentence retrieval from word evidence, r_ = positive control)")
    print(g[cols].mean().round(4).T.to_string())
    print("\nSTD ACROSS FOLDS")
    print(g[["w_sim_margin", "s_sim_margin", "s_mrr", "r_nchar", "r_ffd"]].std().round(4).to_string())
    print(f"\nsaved {OUT / (name + '.csv')}")


if __name__ == "__main__":
    main()