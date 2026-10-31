"""Compute classification metrics for ANY model from its score files, with the SAME code.

Java (Smile) writes  results/tables/scores_smile_val.csv  and  scores_smile_test.csv  (columns: label,score).
This script reads them, checks the labels equal the Python split, picks thresholds on VALIDATION
only, applies them to TEST, and writes results/tables/metrics_<name>.json.

Usage:  python s08_evaluate_scores.py --name smile          (or smile_w90 for the class-weight sensitivity run)
Reports the ACTUAL test FPR at each validation-selected threshold, next to the validation FPR.
"""
import argparse
import json
import numpy as np
import pandas as pd
from common import (load_config, resolve, load_processed, evaluate,
                    threshold_at_fpr, threshold_best_f1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    tdir = resolve(cfg, cfg["paths"]["results_dir"]) / "tables"
    npz, _ = load_processed(cfg)
    val = pd.read_csv(tdir / f"scores_{a.name}_val.csv")
    test = pd.read_csv(tdir / f"scores_{a.name}_test.csv")
    if not (np.array_equal(val.label.to_numpy(), npz["y_val"]) and np.array_equal(test.label.to_numpy(), npz["y_test"])):
        raise SystemExit("Label mismatch between score files and the Python split: the two pipelines saw different data.")
    yva, pva, yte, pte = val.label.to_numpy(), val.score.to_numpy(), test.label.to_numpy(), test.score.to_numpy()
    thr_fpr, thr_f1 = threshold_at_fpr(yva, pva, 0.05), threshold_best_f1(yva, pva)
    ev_fpr = evaluate(yte, pte, thr_fpr); ev_fpr["validation_fpr_at_threshold"] = evaluate(yva, pva, thr_fpr)["fpr"]
    ev_f1 = evaluate(yte, pte, thr_f1); ev_f1["validation_fpr_at_threshold"] = evaluate(yva, pva, thr_f1)["fpr"]
    out = {"model": a.name,
           "test_at_val_fpr5_threshold": ev_fpr,
           "test_at_val_bestf1_threshold": ev_f1,
           "test_at_default_0.5": evaluate(yte, pte, 0.5)}
    (tdir / f"metrics_{a.name}.json").write_text(json.dumps(out, indent=2))
    for k in ("test_at_val_fpr5_threshold", "test_at_val_bestf1_threshold"):
        r = out[k]
        print(f"{a.name} {k}: AUC={r['roc_auc']:.4f} PR-AUC={r['pr_auc']:.4f} P={r['precision']:.3f} R={r['recall']:.3f} MCC={r['mcc']:.4f} test_FPR={r['fpr']:.4f} (val_FPR={r['validation_fpr_at_threshold']:.4f})")


if __name__ == "__main__":
    main()
