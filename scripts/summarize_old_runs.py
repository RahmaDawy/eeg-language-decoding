"""One-time summary of the earlier CBraMod / SSL runs (continuous windows, inferred alignment).
Run:  python scripts\summarize_old_runs.py | Out-File -Encoding utf8 old_runs.txt
"""
import pandas as pd

M = "experiments/results/metrics/"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)

for name in ("results_pretrained.csv", "results_robust_ssl.csv"):
    d = pd.read_csv(M + name)
    print("\n=====", name, "rows:", len(d), "| datasets:", sorted(d.dataset.unique()), "| noise:", list(d.noise.unique()))
    g = d.groupby(["dataset", "representation", "method", "noise"])
    out = g.agg(folds=("fold", "nunique"), top1=("top1", "mean"), chance_top1=("chance_top1", "mean"),
                mrr=("mrr", "mean"), chance_mrr=("chance_mrr", "mean"),
                sim_margin=("sim_margin", "mean"), sim_sd=("sim_margin", "std")).round(4)
    print(out.to_string())

t = pd.read_csv(M + "results_transfer_encoders.csv")
print("\n===== results_transfer_encoders.csv rows:", len(t))
cols = ["source", "target", "target_subset", "representation", "method", "n_runs", "n_candidates",
        "top1", "chance_top1", "mrr", "chance_mrr", "sim_margin"]
print(t[cols].round(4).to_string(index=False))