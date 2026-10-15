import numpy as np
import pytest

from eegdecode.data.load_data import channel_keep_idx, extract_windows
from eegdecode.models.baseline import BAND_NAMES, band_power_features, duration_features, extract_features


def test_extract_windows_slices_samples():
    data = np.arange(100, dtype=np.float64).reshape(50, 2)
    w = extract_windows(data, [(10.0, 20.0), (30.0, 35.0)])
    assert [x.shape for x in w] == [(10, 2), (5, 2)]
    assert w[0][0, 0] == data[10, 0] and w[0].dtype == np.float32


def test_extract_windows_clips_to_recording_end():
    assert extract_windows(np.zeros((50, 2)), [(40.0, 80.0)])[0].shape == (10, 2)


def test_extract_windows_rejects_empty():
    with pytest.raises(ValueError):
        extract_windows(np.zeros((50, 2)), [(20.0, 20.0)])


def test_channel_keep_idx_drops_cz():
    assert channel_keep_idx(["E2", "E3", "Cz"]) == [0, 1]


def test_alpha_sinusoid_peaks_in_alpha_band():
    t = np.arange(2000) / 500.0
    x = np.zeros((2000, 3))
    x[:, 0] = np.sin(2 * np.pi * 10 * t)
    feats = band_power_features(x, 500.0)
    assert feats.shape == (3 * len(BAND_NAMES),)
    assert BAND_NAMES[int(np.argmax(feats[:len(BAND_NAMES)]))] == "alpha"


def test_extract_features_handles_different_window_lengths():
    rng = np.random.default_rng(0)
    F = extract_features([rng.standard_normal((1500, 4)), rng.standard_normal((3000, 4))], 500.0)
    assert F.shape == (2, 4 * len(BAND_NAMES)) and np.isfinite(F).all()


def test_duration_feature_is_log_seconds():
    f = duration_features([np.zeros((500, 2)), np.zeros((1000, 2))], 500.0)
    assert f.shape == (2, 1) and np.isclose(f[1, 0] - f[0, 0], np.log(2))