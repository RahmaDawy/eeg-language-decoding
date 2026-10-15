"""Load ZuCo EEG recordings and cut them into sentence windows (event 10 -> event 11).

Assumptions (to be confirmed in reports/data_audit.md):
  * EEG/data is (samples x channels), as seen in both inspected files.
  * Event latencies follow EEGLAB's 1-based sample convention (effect of being wrong: 1 sample = 2 ms).
  * Trigger 10 = window start, 11 = window end. This is a HYPOTHESIS supported by alternation and
    by identical window counts across subjects; the duration-vs-sentence-length check comes after alignment.
"""
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from eegdecode.data.inspect_data import EEG_RE
from eegdecode.data.probe_eeg import get, read_events, sentence_windows

EEGLAB_LATENCY_OFFSET = 1.0
DEFAULT_DROP_CHANNELS = ("Cz",)  # all-zero reference channel in both inspected files


@dataclass
class Recording:
    dataset: str
    subject: str
    block: int
    prefix: str
    srate: float
    channels: list
    windows: list  # list of float32 arrays (T, C)
    bounds: list  # list of (start_sample, end_sample), 0-based


def list_recordings(root):
    items = []
    for p in sorted(Path(root).rglob("*_EEG.mat")):
        m = EEG_RE.match(p.name)
        if m:
            items.append(dict(path=p, subject=m["subject"], block=int(m["block"]), prefix=m["prefix"]))
    return items


def extract_windows(data, bounds):
    """Slice (n_samples, n_channels) data into windows; ends are clipped to the recording."""
    out = []
    for a, b in bounds:
        s, e = max(int(round(a)), 0), min(int(round(b)), data.shape[0])
        if e <= s:
            raise ValueError(f"empty window {a} -> {b}")
        out.append(np.asarray(data[s:e], dtype=np.float32))
    return out


def channel_keep_idx(channels, drop=DEFAULT_DROP_CHANNELS):
    return [i for i, c in enumerate(channels) if c not in drop]


def load_recording(path, dataset, start="10", end="11"):
    path = Path(path)
    m = EEG_RE.match(path.name)
    if not m:
        raise ValueError(f"unexpected file name: {path.name}")
    with h5py.File(path, "r") as f:
        srate = float(get(f, "EEG/srate"))
        channels = list(get(f, "EEG/chanlocs/labels"))
        events = read_events(f)
        data = f["EEG/data"][()]
    if data.shape[1] != len(channels):
        raise ValueError(f"{path.name}: data {data.shape} vs {len(channels)} channel labels")
    wins, problems = sentence_windows(events, start, end)
    if problems:
        raise ValueError(f"{path.name}: {problems} unpaired {start}/{end} events")
    bounds = [(a - EEGLAB_LATENCY_OFFSET, b - EEGLAB_LATENCY_OFFSET) for a, b in wins]
    return Recording(dataset, m["subject"], int(m["block"]), m["prefix"], srate, channels,
                     extract_windows(data, bounds), bounds)


def qc_table(rec, min_duration_s=1.0):
    rows = []
    for i, (w, (a, b)) in enumerate(zip(rec.windows, rec.bounds)):
        dur = len(w) / rec.srate
        finite = bool(np.isfinite(w).all())
        rows.append(dict(dataset=rec.dataset, subject=rec.subject, block=rec.block, prefix=rec.prefix,
                         win_idx=i, start=a, end=b, duration_s=dur, finite=finite,
                         valid=finite and dur >= min_duration_s))
    return pd.DataFrame(rows)