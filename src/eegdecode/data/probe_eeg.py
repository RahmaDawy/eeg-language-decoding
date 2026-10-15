"""Probe ZuCo files, reading only fields seen in the inspected trees.

Subcommands:
  eeg <file> [<file2>]        shapes, srate, channels, events, quality fields of 1-2 EEG files
  events <root> [--out CSV] [--zuco2-dir DIR]
                              scan every *_EEG.mat under root (events + quality fields; signal not loaded)
  sentences <mat>             count / duplicates in sentencesNR.mat
  overlap --zuco1-mat M --zuco2-dir D [--out CSV]
                              sentence overlap between ZuCo 1.0 and ZuCo 2.0
"""
import argparse
import re
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import io as sio

from eegdecode.data.inspect_data import EEG_RE

MISSING = "(missing)"


def is_missing(v) -> bool:
    return isinstance(v, str) and v == MISSING


# ------------------------------------------------------------ HDF5 helpers
def _decode(arr):
    """Decode a value read from a MATLAB v7.3 file."""
    if not isinstance(arr, np.ndarray):
        return arr
    if arr.dtype == np.uint16:  # MATLAB char array stored as UTF-16 code units
        return "".join(chr(int(c)) for c in arr.ravel())
    if arr.dtype == np.uint64 and arr.shape == (2,):  # MATLAB empty-array marker
        return None
    if arr.size == 1:
        return arr.item()
    return arr


def deref(f, ref):
    return _decode(f[ref][()])


def get(f, path):
    """Read a dataset by path; follows object references (MATLAB cells / struct arrays)."""
    if path not in f:
        return MISSING
    obj = f[path]
    if isinstance(obj, h5py.Group):
        return f"(group: {list(obj.keys())})"
    arr = obj[()]
    if arr.dtype == object:
        return [deref(f, r) for r in arr.ravel()]
    return _decode(arr)


def flat_str(v):
    """Compact, CSV-friendly text for a value read with get()."""
    if v is None or is_missing(v):
        return None
    if isinstance(v, np.ndarray):
        return ",".join(
            str(int(x)) if float(x).is_integer() else f"{x:.4g}" for x in v.ravel()[:60]
        )
    return v


def _short(x, n=10):
    if isinstance(x, list):
        return f"list[{len(x)}] first {n}: {x[:n]}"
    if isinstance(x, np.ndarray):
        if x.size <= 20:
            return f"array{x.shape} {x.tolist()}"
        return f"array{x.shape} dtype={x.dtype} min={x.min():.4g} max={x.max():.4g}"
    return repr(x)


AUTOMAGIC_PATHS = [
    # newer layout (seen in the second file)
    "automagic/SamplingFrequency", "automagic/RecordingDuration", "automagic/EEGChannelCount",
    "automagic/qualityScores/CHV", "automagic/qualityScores/MAV", "automagic/qualityScores/OHA",
    "automagic/qualityScores/RBC", "automagic/qualityScores/THV",
    "automagic/qualityThresholds/overallThresh", "automagic/selectedQualityScore/OHA",
    "automagic/finalBadChans", "automagic/interpolation/channels", "automagic/isInterpolated",
    "automagic/isManuallyRated",
    "automagic/filtering/highpass/freq", "automagic/filtering/lowpass/freq",
    # older layout (seen in the first file)
    "auto_badchans", "man_badchans", "is_interpolated", "tobe_interpolated", "ica_rejected",
    "params/filter_params/high_freq", "params/filter_params/low_freq", "params/filter_params/notch_freq",
    "EEG/etc/clean_channel_mask",
]


# ------------------------------------------------------------ events
def read_events(f):
    """Return [(type_str_stripped, latency_in_samples, value_str)] for EEG/event."""
    types = get(f, "EEG/event/type")
    lat = get(f, "EEG/event/latency")
    val = get(f, "EEG/event/value")
    if not (isinstance(types, list) and isinstance(lat, list)):
        raise ValueError(f"cannot read events: type={types!r}")
    if not isinstance(val, list):
        val = [""] * len(types)
    return [(str(t).strip(), float(l), str(v).strip()) for t, l, v in zip(types, lat, val)]


def sentence_windows(events, start="10", end="11"):
    """Pair each `start` event with the next `end` event. Returns (windows, n_problems).

    A problem is a start with no matching end, or an end with no preceding start.
    The meaning of the trigger codes is a HYPOTHESIS until verified.
    """
    ev = sorted((float(l), str(t).strip()) for t, l, _ in events if str(t).strip() in (start, end))
    wins, open_, problems = [], None, 0
    for lat, t in ev:
        if t == start:
            if open_ is not None:
                problems += 1
            open_ = lat
        elif open_ is None:
            problems += 1
        else:
            wins.append((open_, lat))
            open_ = None
    if open_ is not None:
        problems += 1
    return wins, problems


# ------------------------------------------------------------ eeg probe
def probe_eeg(path, n_events=12):
    path = Path(path)
    print(f"\n=== {path} ({path.stat().st_size / 1e6:.1f} MB)")
    with h5py.File(path, "r") as f:
        data = f["EEG/data"]
        print(f"EEG/data: shape={data.shape} dtype={data.dtype}")
        for key in ("srate", "nbchan", "pnts", "trials", "xmin", "xmax"):
            print(f"EEG/{key}: {_short(get(f, 'EEG/' + key))}")
        print(f"EEG/ref: {_short(get(f, 'EEG/ref'))}")

        times = f["EEG/times"][()].ravel()
        print(f"EEG/times: first={times[0]:.3f} second={times[1]:.3f} "
              f"last={times[-1]:.3f} step={times[1] - times[0]:.4f}")

        labels = get(f, "EEG/chanlocs/labels")
        print(f"channels ({len(labels)}): {labels}")

        arr = data[()]
        nb = int(get(f, "EEG/nbchan"))
        if arr.shape[1] == nb:
            ch_axis, layout = 1, "samples x channels"
        elif arr.shape[0] == nb:
            ch_axis, layout = 0, "channels x samples"
        else:
            ch_axis, layout = None, "UNKNOWN (no axis equals nbchan)"
        print(f"layout: {layout}")
        print(f"NaN count: {int(np.isnan(arr).sum())}, Inf count: {int(np.isinf(arr).sum())}")
        print(f"fraction of exactly-zero values: {float((arr == 0).mean()):.5f}")
        flat = arr.ravel()[::97]
        p = np.percentile(flat, [0.1, 50, 99.9])
        print(f"values (subsampled): min={flat.min():.4g} max={flat.max():.4g} "
              f"p0.1={p[0]:.4g} median={p[1]:.4g} p99.9={p[2]:.4g}")
        if ch_axis is not None:
            ch_std = arr.std(axis=1 - ch_axis)
            flat_idx = np.where(ch_std == 0)[0]
            print(f"per-channel std: min={ch_std.min():.4g} median={np.median(ch_std):.4g} "
                  f"max={ch_std.max():.4g}")
            print(f"flat channels (std==0): {[labels[i] for i in flat_idx]}")
        del arr

        print(f"eventdescription: {_short(get(f, 'EEG/eventdescription'))}")
        try:
            ev = read_events(f)
        except ValueError as e:
            print(e)
            ev = []
        if ev:
            srate = float(get(f, "EEG/srate"))
            print(f"events: {len(ev)}; type counts: {dict(Counter(t for t, _, _ in ev))}")
            print("idx | type | latency(sample) | latency(s) | value")
            shown = list(range(min(n_events, len(ev)))) + ["..."] + list(range(max(n_events, len(ev) - 4), len(ev)))
            for i in shown:
                if i == "...":
                    print("...")
                    continue
                t, l, v = ev[i]
                print(f"{i:3d} | {t!r} | {l} | {l / srate:.3f} | {v!r}")
            wins, problems = sentence_windows(ev)
            d = np.array([(b - a) / srate for a, b in wins])
            if d.size:
                print(f"10->11 windows: {len(wins)} (problems={problems}); duration s: "
                      f"min={d.min():.2f} median={np.median(d):.2f} max={d.max():.2f}")

        print("\nquality / preprocessing fields (missing ones are skipped):")
        for pth in AUTOMAGIC_PATHS:
            v = get(f, pth)
            if not is_missing(v):
                print(f"  {pth}: {_short(v)}")
    return labels


def compare_labels(a, b):
    print("\n=== channel label comparison (file 1 vs file 2)")
    print(f"same length: {len(a) == len(b)}; identical order: {a == b}")
    sa, sb = set(a), set(b)
    print(f"only in file 1: {sorted(sa - sb)}")
    print(f"only in file 2: {sorted(sb - sa)}")


# ------------------------------------------------------------ events scan over all files
def events_summary(root, start="10", end="11"):
    rows = []
    for p in sorted(Path(root).rglob("*_EEG.mat")):
        m = EEG_RE.match(p.name)
        if not m:
            continue
        row = dict(subject=m["subject"], block=int(m["block"]), prefix=m["prefix"], file=p.name)
        try:
            with h5py.File(p, "r") as f:
                shape = f["EEG/data"].shape  # not loaded
                srate = float(get(f, "EEG/srate"))
                row.update(srate=srate, n_samples=shape[0], n_cols=shape[1],
                           minutes=round(shape[0] / srate / 60, 2))
                ev = read_events(f)
                row.update({f"t_{k}": v for k, v in Counter(t for t, _, _ in ev).items()})
                wins, problems = sentence_windows(ev, start, end)
                d = np.array([(b - a) / srate for a, b in wins])
                row.update(
                    n_events=len(ev), n_pairs=len(wins), pair_problems=problems,
                    dur_min=d.min() if d.size else np.nan,
                    dur_med=float(np.median(d)) if d.size else np.nan,
                    dur_max=d.max() if d.size else np.nan,
                )
                for k in ("CHV", "MAV", "OHA", "RBC", "THV"):
                    row[f"q_{k}"] = flat_str(get(f, f"automagic/selectedQualityScore/{k}"))
                row["final_bad"] = flat_str(get(f, "automagic/finalBadChans"))
                row["old_auto_bad"] = flat_str(get(f, "auto_badchans"))
                row["old_man_bad"] = flat_str(get(f, "man_badchans"))
                interp = get(f, "automagic/isInterpolated")
                if is_missing(interp):
                    interp = get(f, "is_interpolated")
                row["interpolated"] = flat_str(interp)
        except Exception as e:  # record and keep scanning
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
    return pd.DataFrame(rows)


def cmd_events(root, out=None, zuco2_dir=None):
    df = events_summary(Path(root))
    if df.empty:
        print("No EEG files found.")
        return
    if "error" in df and df["error"].notna().any():
        print("\n[errors]")
        print(df.loc[df["error"].notna(), ["subject", "block", "error"]].to_string(index=False))
    ok = df[df.get("error").isna()] if "error" in df else df
    tcols = sorted(c for c in ok.columns if c.startswith("t_"))
    print(f"\nfiles: {len(df)} (ok: {len(ok)}); sampling rates: {sorted(ok.srate.dropna().unique())}; "
          f"data columns: {sorted(ok.n_cols.dropna().unique())}")
    print(f"event-type columns present: {tcols}")
    print(f"total 10->11 pair problems: {int(ok.pair_problems.sum())}")
    g = ok.groupby("block").agg(
        files=("file", "size"), pairs_min=("n_pairs", "min"), pairs_med=("n_pairs", "median"),
        pairs_max=("n_pairs", "max"), problems=("pair_problems", "sum"), dur_med_s=("dur_med", "median"),
        minutes_med=("minutes", "median"),
    )
    print("\nper block (windows = 10->11 pairs found in EEG events):")
    print(g.to_string())
    if zuco2_dir:
        s = pd.DataFrame(load_zuco2_sentences(zuco2_dir))
        c = s.groupby("block").agg(csv_rows=("text", "size"), csv_noncontrol=("control", lambda x: int((~x).sum())))
        print("\nZuCo 2.0 csv rows per block vs windows found in EEG events:")
        print(c.join(g[["pairs_min", "pairs_med", "pairs_max"]]).to_string())
    print("\nprefix counts:", ok.prefix.value_counts().to_dict())
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"saved: {out}")


# ------------------------------------------------------------ sentences
def norm_text(s: str) -> str:
    """Lowercase and drop everything except letters/digits, so tiny spacing/punctuation differences don't hide a match."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def load_zuco1_sentences(mat_path):
    arr = sio.loadmat(mat_path, squeeze_me=True)["sentences"]
    return [str(s) for s in np.asarray(arr).ravel()]


def parse_nr_line(line: str) -> dict:
    parts = line.rstrip("\r\n").split(";")
    if len(parts) < 4:
        raise ValueError(f"unexpected line (<4 fields): {line[:60]!r}")
    return dict(par=int(parts[0]), sent=int(parts[1]),
                text=";".join(parts[2:-1]).strip(), control=parts[-1].strip() == "CONTROL")


def load_zuco2_sentences(task_dir):
    rows = []
    for p in sorted(Path(task_dir).glob("nr_*.csv")):
        m = re.fullmatch(r"nr_(\d+)\.csv", p.name)
        if not m:  # skips nr_*_control_questions.csv
            continue
        for line in p.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            if line.strip():
                r = parse_nr_line(line)
                r["block"] = int(m[1])
                rows.append(r)
    return rows


def cmd_sentences(mat_path):
    s = load_zuco1_sentences(mat_path)
    n = [norm_text(x) for x in s]
    print(f"sentences in file: {len(s)}; unique after normalization: {len(set(n))}")
    print(f"duplicates inside the file: {len(s) - len(set(n))}")
    print(f"shortest/longest: {min(map(len, s))}/{max(map(len, s))} chars")


def cmd_overlap(mat_path, zuco2_dir, out=None):
    z1 = {norm_text(x) for x in load_zuco1_sentences(mat_path)}
    df = pd.DataFrame(load_zuco2_sentences(zuco2_dir))
    df["norm"] = df.text.map(norm_text)
    df["in_zuco1"] = df.norm.isin(z1)
    print(f"ZuCo 2.0 NR rows: {len(df)}, unique texts: {df.norm.nunique()}")
    print(f"ZuCo 1.0 unique sentences: {len(z1)}")
    print(df.groupby("block").agg(rows=("norm", "size"), in_zuco1=("in_zuco1", "sum")).to_string())
    print(f"\ntotal ZuCo 2.0 rows found in ZuCo 1.0: {int(df.in_zuco1.sum())} "
          f"({df.loc[df.in_zuco1, 'norm'].nunique()} unique)")
    print(f"rows in ZuCo 2.0 whose text repeats elsewhere in ZuCo 2.0: {int(df.norm.duplicated(keep=False).sum())}")
    if out:  # no sentence text is saved
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        df[["block", "par", "sent", "control", "in_zuco1"]].to_csv(out, index=False)
        print(f"saved (no text): {out}")


# ------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("eeg"); a.add_argument("files", nargs="+")
    e = sub.add_parser("events"); e.add_argument("root"); e.add_argument("--out"); e.add_argument("--zuco2-dir")
    b = sub.add_parser("sentences"); b.add_argument("mat")
    c = sub.add_parser("overlap")
    c.add_argument("--zuco1-mat", required=True); c.add_argument("--zuco2-dir", required=True)
    c.add_argument("--out")
    args = ap.parse_args(argv)

    if args.cmd == "eeg":
        labels = [probe_eeg(p) for p in args.files[:2]]
        if len(labels) == 2:
            compare_labels(*labels)
    elif args.cmd == "events":
        cmd_events(args.root, args.out, args.zuco2_dir)
    elif args.cmd == "sentences":
        cmd_sentences(args.mat)
    else:
        cmd_overlap(args.zuco1_mat, args.zuco2_dir, args.out)


if __name__ == "__main__":
    main()