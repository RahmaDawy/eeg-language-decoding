"""Handcrafted baseline: log band power per channel (Welch), plus a duration-only control feature.

Bands stop at 45 Hz because the data are filtered to roughly 0.5-50 Hz with 50 Hz line noise.
"""
import numpy as np
from scipy.signal import welch

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
BAND_NAMES = list(BANDS)


def band_power_features(window, srate=500.0, bands=BANDS, nperseg=1024):
    """window (T, C) -> (C * n_bands,) log10 band power, channel-major (all bands of ch0, then ch1, ...)."""
    x = np.asarray(window, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected (T, C), got {x.shape}")
    freqs, psd = welch(x, fs=srate, nperseg=min(nperseg, x.shape[0]), axis=0)  # psd (F, C)
    df = freqs[1] - freqs[0]
    feats = []
    for lo, hi in bands.values():
        m = (freqs >= lo) & (freqs < hi)
        feats.append(np.log10(psd[m].sum(axis=0) * df + 1e-12))
    return np.stack(feats, axis=1).reshape(-1)


def extract_features(windows, srate=500.0, keep_idx=None):
    rows = []
    for w in windows:
        if keep_idx is not None:
            w = w[:, keep_idx]
        rows.append(band_power_features(w, srate))
    return np.vstack(rows)


def duration_features(windows, srate=500.0):
    """Shortcut control: log window duration in seconds, shape (n, 1)."""
    return np.log(np.array([[len(w) / srate] for w in windows]))