"""Controlled input perturbations. Everything is seeded so runs are reproducible."""
import numpy as np


def noise_windows_like(windows, seed=0, scale=None):
    """Gaussian-noise windows with the same shapes (so the same durations) as `windows`.

    Pass `scale` computed from TRAINING windows to avoid using test statistics; the default
    (mean per-window std of the given windows) is only for quick checks.
    """
    rng = np.random.default_rng(seed)
    if scale is None:
        scale = float(np.mean([w.std() for w in windows]))
    return [(rng.standard_normal(w.shape) * scale).astype(np.float32) for w in windows]