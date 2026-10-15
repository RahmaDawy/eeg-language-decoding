"""Print the raw first lines, shape, columns and head of the sentence CSVs. Read-only."""
from pathlib import Path

import pandas as pd

FILES = [
    r"data\raw\zuco1\task_materials\relations_labels_task2.csv",
    r"data\raw\zuco1\task2 - NR\Preprocessed\relations_normal_reading.csv",
    r"data\raw\zuco1\task_materials\duplicate_sentences.csv",
    r"data\raw\zuco2\task_materials\nr_1.csv",
    r"data\raw\zuco2\task_materials\nr_2.csv",
    r"data\raw\zuco2\task_materials\nr_1_control_questions.csv",
]

for p in FILES:
    print("\n" + "=" * 100)
    print(p)
    raw = Path(p).read_bytes().decode("latin-1").splitlines()
    print("RAW first 3 lines:")
    for line in raw[:3]:
        print("   ", line[:200])
    for sep in (",", ";", "\t"):
        try:
            df = pd.read_csv(p, encoding="latin-1", sep=sep)
            if df.shape[1] > 1:
                break
        except Exception as e:
            df = None
    if df is None:
        print("could not parse")
        continue
    print(f"parsed with sep={sep!r}: {len(df)} rows, columns = {list(df.columns)}")
    print(df.head(4).to_string(max_colwidth=70))