"""Inspect ZuCo files without assuming any field names.

Subcommands:
  inventory <root>                        subject x block table built from EEG file names
  mat <file> [--depth N]                  print the nested structure of a .mat file (v7 or v7.3)
  csv <file> [--sep ;] [--no-header]      print columns and first rows of a CSV
"""
import argparse
import re
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import io as sio

# Naming pattern seen in the OSF listings: <prefix>_<SUBJECT>_NR<block>_EEG.mat
# The meaning of the 3-letter prefix (gip/oip/bip) is NOT verified yet.
EEG_RE = re.compile(
    r"^(?P<prefix>[a-z]{3})_(?P<subject>[A-Z]{3})_NR(?P<block>\d+)_EEG\.mat$"
)


# ---------------------------------------------------------------- inventory
def inventory(root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(root).rglob("*_EEG.mat")):
        m = EEG_RE.match(p.name)
        if not m:
            print(f"[warn] name does not match pattern: {p.name}")
            continue
        rows.append(
            dict(
                subject=m["subject"],
                block=int(m["block"]),
                prefix=m["prefix"],
                size_mb=round(p.stat().st_size / 1e6, 1),
                folder=p.parent.name,
                path=str(p),
            )
        )
    return pd.DataFrame(rows)


def print_inventory(df: pd.DataFrame) -> None:
    if df.empty:
        print("No EEG files found. Check the path.")
        return
    table = df.pivot(index="subject", columns="block", values="prefix").fillna("-")
    print(f"\nEEG files: {len(df)} | subjects: {df.subject.nunique()} "
          f"| blocks seen: {sorted(df.block.unique())}")
    print("\nSubject x block (cell = filename prefix, '-' = missing):")
    print(table.to_string())
    print("\nPrefix counts:")
    print(df.prefix.value_counts().to_string())
    mism = df[df.subject != df.folder]
    if len(mism):
        print(f"\n[warn] {len(mism)} files sit in a folder that differs from their subject code:")
        print(mism[["subject", "block", "folder"]].to_string(index=False))
    print(f"\nTotal size: {df.size_mb.sum() / 1e3:.2f} GB")


# ---------------------------------------------------------------- .mat files
def mat_format(path: Path) -> str:
    with open(path, "rb") as f:
        header = f.read(128)
    text = header.decode("latin-1", errors="replace")
    print(f"header: {text[:116].strip()}")
    return "hdf5" if "MATLAB 7.3" in text else "v5"


def _scalar_repr(x) -> str:
    s = repr(x)
    return s if len(s) <= 80 else s[:77] + "..."


def _walk_v5(obj, name, depth, max_depth, indent=0):
    pad = "  " * indent
    if hasattr(obj, "_fieldnames"):  # MATLAB struct
        fields = list(obj._fieldnames)
        print(f"{pad}{name}: struct, {len(fields)} fields: {fields[:30]}")
        if depth < max_depth:
            for fld in fields:
                _walk_v5(getattr(obj, fld), fld, depth + 1, max_depth, indent + 1)
    elif isinstance(obj, np.ndarray) and obj.dtype == object:  # cell array / struct array
        print(f"{pad}{name}: object array (cell or struct array), shape={obj.shape}")
        if depth < max_depth and obj.size > 0:
            _walk_v5(obj.flat[0], f"{name}[0]", depth + 1, max_depth, indent + 1)
    elif isinstance(obj, np.ndarray):
        extra = ""
        if obj.size and obj.size <= 8 and obj.dtype.kind in "iuf":
            extra = f" values={obj.tolist()}"
        print(f"{pad}{name}: ndarray shape={obj.shape} dtype={obj.dtype}{extra}")
    elif isinstance(obj, str):
        print(f"{pad}{name}: str {_scalar_repr(obj)}")
    else:
        print(f"{pad}{name}: {type(obj).__name__} {_scalar_repr(obj)}")


def _walk_h5(path: Path, max_depth: int):
    with h5py.File(path, "r") as f:
        def visit(name, item):
            if name.count("/") >= max_depth:
                return
            pad = "  " * name.count("/")
            leaf = name.split("/")[-1]
            if isinstance(item, h5py.Dataset):
                kind = "HDF5 object references (MATLAB cell)" if item.dtype == object else str(item.dtype)
                print(f"{pad}{leaf}: dataset shape={item.shape} dtype={kind}")
            else:
                print(f"{pad}{leaf}: group, keys={list(item.keys())[:30]}")
        print(f"top-level keys: {list(f.keys())}")
        f.visititems(visit)


def inspect_mat(path: Path, max_depth: int = 3) -> None:
    path = Path(path)
    print(f"\nFile: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    fmt = mat_format(path)
    print(f"detected format: {fmt}\n")
    if fmt == "hdf5":
        _walk_h5(path, max_depth)
        return
    print("variables (cheap listing):")
    for var, shape, cls in sio.whosmat(path):
        print(f"  {var}: shape={shape} class={cls}")
    print("\nnested structure (loads the file; may take a few seconds):")
    data = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    for key, val in data.items():
        if key.startswith("__"):
            continue
        _walk_v5(val, key, 0, max_depth)


# ---------------------------------------------------------------- csv
def inspect_csv(path: Path, n: int = 5, sep: str = ",", header: bool = True) -> None:
    path = Path(path)
    print(f"\nFile: {path}")
    if header:
        df = pd.read_csv(path, sep=sep, encoding_errors="replace")
    else:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        counts = Counter(line.count(sep) for line in lines)
        print(f"lines={len(lines)}; separators per line -> number of lines: {dict(counts)}")
        if len(counts) > 1:
            print("[warn] lines have different numbers of separators; the text may contain "
                  f"'{sep}' itself, so check parsing before trusting columns")
        df = pd.read_csv(path, sep=sep, header=None, engine="python",
                         on_bad_lines="warn", encoding_errors="replace")
    print(f"shape={df.shape}\ncolumns={list(df.columns)}\n")
    print(df.head(n).to_string())


# ---------------------------------------------------------------- CLI
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("inventory"); a.add_argument("root"); a.add_argument("--out")
    b = sub.add_parser("mat"); b.add_argument("file"); b.add_argument("--depth", type=int, default=3)
    c = sub.add_parser("csv"); c.add_argument("file"); c.add_argument("--rows", type=int, default=5)
    c.add_argument("--sep", default=","); c.add_argument("--no-header", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "inventory":
        df = inventory(Path(args.root))
        print_inventory(df)
        if args.out and not df.empty:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            df.drop(columns="path").to_csv(args.out, index=False)
            print(f"\nSaved: {args.out}")
    elif args.cmd == "mat":
        inspect_mat(Path(args.file), args.depth)
    else:
        inspect_csv(Path(args.file), args.rows, args.sep, not args.no_header)


if __name__ == "__main__":
    main()