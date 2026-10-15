"""Sentence embeddings with a Sentence Transformers model (cached, L2-normalised)."""
import hashlib
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def embed_texts(texts, model_name=DEFAULT_MODEL, cache_path=None, batch_size=64, device=None):
    texts = list(texts)
    key = hashlib.sha1((model_name + "\n" + "\n".join(texts)).encode("utf-8")).hexdigest()
    if cache_path:
        cache_path = Path(cache_path).with_suffix(".npz")
        if cache_path.exists():
            z = np.load(cache_path, allow_pickle=False)
            if str(z["key"]) == key:
                return z["emb"]
    from sentence_transformers import SentenceTransformer  # imported lazily: heavy

    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                       show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, emb=emb, key=np.array(key))
    return emb