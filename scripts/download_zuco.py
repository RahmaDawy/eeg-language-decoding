"""Download selected ZuCo files from OSF into data/raw (never committed; see .gitignore).

Examples:
  python scripts/download_zuco.py --project 2urht --task "task1 - NR" --out data/raw/zuco2 --areas Preprocessed task_materials --subjects YAC YAG --dry-run
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

from osfclient import OSF

BLOCK_RE = re.compile(r"_(?:NR|TSR|SR)(\d+)")


def wanted(rel, task, subjects, no_et, blocks, areas):
    parts = rel.split("/")
    if parts[0] == "task_materials":
        return "task_materials" in areas
    if len(parts) < 3 or parts[0] != task or parts[1] not in areas:
        return False
    name = parts[-1]
    if no_et and name.endswith("_ET.mat"):
        return False
    if subjects:
        if parts[1] == "Preprocessed" and len(parts) > 3 and parts[2] not in subjects:
            return False
        if parts[1] == "Matlab files" and not any(s in name for s in subjects):
            return False
    if blocks:
        m = BLOCK_RE.search(name)
        if m and int(m.group(1)) not in blocks:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--areas", nargs="*", default=["Preprocessed", "Matlab files", "task_materials"])
    ap.add_argument("--subjects", nargs="*", default=[])
    ap.add_argument("--blocks", nargs="*", type=int, default=[])
    ap.add_argument("--no-et", action="store_true", help="skip eye-tracking files")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    storage = OSF().project(args.project).storage("osfstorage")

    by_kind = defaultdict(lambda: [0, 0])
    n_files, total, n_skipped = 0, 0, 0
    for f in storage.files:
        rel = f.path.lstrip("/")
        if not wanted(rel, args.task, args.subjects, args.no_et, args.blocks, args.areas):
            continue
        size = getattr(f, "size", None) or 0
        parts = rel.split("/")
        if parts[0] == "task_materials":
            kind = "task_materials"
        else:
            kind = parts[1] + (" / ET" if parts[-1].endswith("_ET.mat") else
                               " / EEG" if parts[-1].endswith("_EEG.mat") else " / other")
        by_kind[kind][0] += 1
        by_kind[kind][1] += size
        n_files += 1
        total += size
        if args.dry_run:
            continue
        dest = out / rel
        if dest.exists() and size and dest.stat().st_size == size:
            n_skipped += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        print(f"[{n_files}] downloading {rel} ({size / 1e6:.0f} MB)")
        with open(tmp, "wb") as fp:
            f.write_to(fp)
        tmp.replace(dest)

    print("\nBreakdown:")
    for k, (n, b) in sorted(by_kind.items()):
        print(f"  {k:28s} {n:4d} files  {b / 1e9:7.2f} GB")
    print(f"Matched files: {n_files}, total size: {total / 1e9:.2f} GB")
    if not args.dry_run:
        print(f"Already present (skipped): {n_skipped}")
    if n_files == 0:
        print("Nothing matched. Check --project and --task spelling.")


if __name__ == "__main__":
    main()