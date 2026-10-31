"""Temporal split + encoding. Writes arrays for Python and CSVs for Java.

Outputs (data/processed):
  arrays.npz              X_train, y_train, X_val, y_val, X_test, y_test (float32/int8)
  meta.json               feature names, categorical maps, dropped columns, split sizes
  train.csv/val.csv/test.csv   numeric feature matrix + label (last column 'label'), for Java
  bench_rows.csv          fixed sequence of TEST rows replayed by every latency arm (measured calls)
  warmup_rows.csv         fixed sequence of VALIDATION rows used only for warm-up calls
results/tables/preprocessing_audit.json   which data each preprocessing / model choice was derived from

Usage:  python s02_prepare_data.py [--csv path]
"""
import argparse
import json
import numpy as np
import pandas as pd
from common import (load_config, resolve, temporal_split, fit_encoders, apply_encoders, bench_matrix,
                    warmup_matrix, train_class_weight, resolve_class_weight)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    d = cfg["data"]
    path = a.csv or resolve(cfg, d["baf_csv"])
    out = resolve(cfg, d["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(path)
    tr, va, te = temporal_split(df, cfg)
    feats, cat_maps, dropped = fit_encoders(tr, d["label"], d["time_col"])
    arr = {}
    for name, part in (("train", tr), ("val", va), ("test", te)):
        X = apply_encoders(part, feats, cat_maps)
        y = part[d["label"]].to_numpy(dtype=np.int8)
        arr[f"X_{name}"], arr[f"y_{name}"] = X, y
        pd.DataFrame(np.column_stack([X, y]), columns=feats + ["label"]).to_csv(out / f"{name}.csv", index=False, float_format="%.9g")
        print(f"{name:5s}: {len(y):8d} rows, {int(y.sum()):6d} fraud ({100*y.mean():.3f}%)")
    np.savez_compressed(out / "arrays.npz", **arr)
    bm = bench_matrix(arr["X_test"], cfg["benchmark"]["seed"])
    pd.DataFrame(bm, columns=feats).to_csv(out / "bench_rows.csv", index=False, float_format="%.9g")
    wm = warmup_matrix(arr["X_val"], cfg["benchmark"]["seed"])
    pd.DataFrame(wm, columns=feats).to_csv(out / "warmup_rows.csv", index=False, float_format="%.9g")
    ytr = arr["y_train"]
    n_pos = int(ytr.sum()); n_neg = int(len(ytr) - n_pos)
    cw, cw_src = resolve_class_weight(cfg, ytr)
    meta = {
        "features": feats, "categorical_maps": cat_maps, "dropped_columns": dropped,
        "train_months": d["train_months"], "val_months": d["val_months"], "test_months": d["test_months"],
        "n_train": int(len(arr["y_train"])), "n_val": int(len(arr["y_val"])), "n_test": int(len(arr["y_test"])),
        "n_features": len(feats),
        "train_fraud": n_pos, "train_legit": n_neg,
        "class_weight_train_only": train_class_weight(ytr), "class_weight_used": cw,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    audit = {
        "temporal_split": {"train_months": d["train_months"], "validation_months": d["val_months"],
                           "test_months": d["test_months"]},
        "categorical_encoding": "ordinal; levels taken from TRAINING months only; unseen levels -> -1",
        "constant_columns_dropped": {"columns": dropped, "decided_on": "training months only"},
        "missing_values": "BAF sentinel -1 kept as is (no imputation fitted on any data)",
        "fraud_class_weight": {"value": cw, "provenance": cw_src,
                               "train_legit": n_neg, "train_fraud": n_pos,
                               "ratio_exact": round(n_neg / n_pos, 4)},
        "hyperparameters": {"source": "fixed in python/config.yaml before any experiment; not tuned on validation or test",
                            **{k: v for k, v in cfg["model"].items() if k != "fraud_class_weight"}},
        "decision_thresholds": "chosen on the VALIDATION month only (5% FPR target; best F1), then applied to test",
        "benchmark_rows": "bench_rows.csv: fixed-seed sequence of TEST rows (measured calls)",
        "warmup_rows": "warmup_rows.csv: fixed-seed sequence of VALIDATION rows (warm-up calls only)",
    }
    tdir = resolve(cfg, cfg["paths"]["results_dir"]) / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "preprocessing_audit.json").write_text(json.dumps(audit, indent=2))
    print(f"fraud class weight (training months only): {n_neg}/{n_pos} = {n_neg/n_pos:.2f} -> {cw}")
    print(f"{len(feats)} features kept; dropped constant columns: {dropped}")
    print("categorical:", {c: len(m) for c, m in cat_maps.items()})


if __name__ == "__main__":
    main()
