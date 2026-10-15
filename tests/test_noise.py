import numpy as np

from eegdecode.evaluation.robustness import noise_windows_like


def _wins():
    rng = np.random.default_rng(0)
    return [rng.standard_normal((100, 3)).astype(np.float32) * 2, rng.standard_normal((150, 3)).astype(np.float32) * 2]


def test_noise_windows_are_deterministic_and_shape_preserving():
    a, b, c = noise_windows_like(_wins(), seed=1), noise_windows_like(_wins(), seed=1), noise_windows_like(_wins(), seed=2)
    assert [x.shape for x in a] == [(100, 3), (150, 3)]
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[0], c[0])


def test_noise_scale_matches_input_scale():
    out = noise_windows_like(_wins(), seed=3)
    assert abs(float(np.mean([w.std() for w in out])) - 2.0) < 0.3