"""EEG-feature -> sentence-embedding alignment (ridge) and the single entry point for retrieval experiments.

Model selection (alpha) uses the VALIDATION split only; the test split is touched once.
Normalisation statistics (StandardScaler) are fitted on training rows only.
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from eegdecode.evaluation.metrics import candidate_index, cosine_matrix, retrieval_metrics

DEFAULT_ALPHAS = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)


class RidgeAligner:
    def __init__(self, alpha=10.0):
        self.alpha = alpha

    def fit(self, X, Y):
        self.scaler_ = StandardScaler().fit(X)
        self.model_ = Ridge(alpha=self.alpha).fit(self.scaler_.transform(X), Y)
        return self

    def predict(self, X):
        return self.model_.predict(self.scaler_.transform(X))


def evaluate(aligner, X, cand_emb, true_idx, ks=(1, 5, 10)):
    sim = cosine_matrix(aligner.predict(X), cand_emb)
    return retrieval_metrics(sim, true_idx, ks)


def select_alpha(Xtr, Ytr, Xva, cand_emb_va, true_idx_va, alphas=DEFAULT_ALPHAS, metric="mrr"):
    best_alpha, best_score, table = None, -np.inf, []
    for a in alphas:
        r = evaluate(RidgeAligner(a).fit(Xtr, Ytr), Xva, cand_emb_va, true_idx_va)
        table.append({"alpha": a, "mrr": r["mrr"], "top1": r["top1"]})
        if r[metric] > best_score:
            best_alpha, best_score = a, r[metric]
    return best_alpha, table


def run_experiment(X, sentence_ids, emb_by_sentence, split, alphas=DEFAULT_ALPHAS, ks=(1, 5, 10)):
    """X (n_rows, n_features); sentence_ids (n_rows,); emb_by_sentence {sentence_id: embedding};
    split (n_rows,) in {'train','val','test','drop'}. Candidates = unique sentences of the evaluated split."""
    X = np.asarray(X, dtype=np.float64)
    sentence_ids = np.asarray(sentence_ids)
    split = np.asarray(split)
    if not (len(X) == len(sentence_ids) == len(split)):
        raise ValueError("X, sentence_ids and split must have the same length")
    parts = {n: np.where(split == n)[0] for n in ("train", "val", "test")}
    for n, idx in parts.items():
        if len(idx) == 0:
            raise ValueError(f"empty {n} split")

    def candidates(idx):
        uniq, true_idx = candidate_index(sentence_ids[idx].tolist())
        return np.stack([emb_by_sentence[s] for s in uniq]), true_idx

    Ytr = np.stack([emb_by_sentence[s] for s in sentence_ids[parts["train"]]])
    cand_va, true_va = candidates(parts["val"])
    best_alpha, table = select_alpha(X[parts["train"]], Ytr, X[parts["val"]], cand_va, true_va, alphas)
    model = RidgeAligner(best_alpha).fit(X[parts["train"]], Ytr)
    cand_te, true_te = candidates(parts["test"])
    return {
        "alpha": best_alpha,
        "val_table": table,
        "test": evaluate(model, X[parts["test"]], cand_te, true_te, ks),
        "n_rows": {n: int(len(i)) for n, i in parts.items()},
    }