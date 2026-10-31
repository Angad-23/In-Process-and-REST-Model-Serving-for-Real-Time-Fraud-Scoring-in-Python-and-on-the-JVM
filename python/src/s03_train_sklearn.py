"""Train the scikit-learn Random Forest, evaluate, and export to ONNX.

Thresholds are chosen on the VALIDATION month only, then applied to test.
Two operating points are reported: (a) FPR<=5% and (b) best F1 on validation.
Also writes an ONNX copy of the SAME model, so a JVM can run the identical
trees (no accuracy difference to explain away) via ONNX Runtime.

Fraud class weight: derived from the TRAINING months only (config 'auto').
Sensitivity check:  python s03_train_sklearn.py --fraud-weight 90 --tag w90
  -> writes metrics_sklearn_w90.json and scores only; the main model and ONNX file are NOT touched.
"""
import argparse
import json
import time
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from common import (load_config, resolve, load_processed, evaluate,
                    threshold_at_fpr, threshold_best_f1, resolve_class_weight)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-onnx", action="store_true")
    ap.add_argument("--fraud-weight", default=None, help="override the class weight (sensitivity check only)")
    ap.add_argument("--tag", default="", help="suffix for a sensitivity run; main model/ONNX are not overwritten")
    a = ap.parse_args()
    cfg = load_config(a.config)
    m = cfg["model"]
    npz, meta = load_processed(cfg)
    res_dir = resolve(cfg, cfg["paths"]["results_dir"])
    (res_dir / "models").mkdir(parents=True, exist_ok=True)
    (res_dir / "tables").mkdir(parents=True, exist_ok=True)

    Xtr, ytr, Xva, yva, Xte, yte = (npz[k] for k in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"))
    cw, cw_src = resolve_class_weight(cfg, ytr, a.fraud_weight)
    name = "sklearn" + (f"_{a.tag}" if a.tag else "")
    print(f"fraud class weight = {cw}  ({cw_src})")
    clf = RandomForestClassifier(
        n_estimators=m["n_estimators"], max_depth=m["max_depth"],
        max_leaf_nodes=m["max_leaf_nodes"], min_samples_leaf=m["min_samples_leaf"],
        max_features=m["max_features"], class_weight={0: 1, 1: cw},
        random_state=m["random_state"], n_jobs=-1)
    t0 = time.time()
    clf.fit(Xtr, ytr)
    train_s = time.time() - t0

    pva = clf.predict_proba(Xva)[:, 1]
    pte = clf.predict_proba(Xte)[:, 1]
    thr_fpr = threshold_at_fpr(yva, pva, 0.05)
    thr_f1 = threshold_best_f1(yva, pva)
    ev_fpr = evaluate(yte, pte, thr_fpr)
    ev_fpr["validation_fpr_at_threshold"] = evaluate(yva, pva, thr_fpr)["fpr"]
    ev_f1 = evaluate(yte, pte, thr_f1)
    ev_f1["validation_fpr_at_threshold"] = evaluate(yva, pva, thr_f1)["fpr"]
    out = {
        "model": "sklearn RandomForest", "params": {**m, "fraud_class_weight": cw},
        "class_weight_provenance": cw_src, "train_seconds": train_s,
        "n_features": meta["n_features"],
        "test_at_val_fpr5_threshold": ev_fpr,
        "test_at_val_bestf1_threshold": ev_f1,
        "test_at_default_0.5": evaluate(yte, pte, 0.5),
    }
    (res_dir / "tables" / f"metrics_{name}.json").write_text(json.dumps(out, indent=2))
    import pandas as pd
    pd.DataFrame({"label": yva, "score": pva}).to_csv(res_dir / "tables" / f"scores_{name}_val.csv", index=False)
    pd.DataFrame({"label": yte, "score": pte}).to_csv(res_dir / "tables" / f"scores_{name}_test.csv", index=False)
    if not a.tag:
        joblib.dump(clf, res_dir / "models" / "rf_sklearn.joblib")
        json.dump({"fpr5": thr_fpr, "bestf1": thr_f1, "fraud_class_weight": cw},
                  open(res_dir / "models" / "thresholds_sklearn.json", "w"))
    for k in ("test_at_val_fpr5_threshold", "test_at_val_bestf1_threshold"):
        r = out[k]
        print(f"{name} {k}: AUC={r['roc_auc']:.4f} PR-AUC={r['pr_auc']:.4f} P={r['precision']:.3f} "
              f"R={r['recall']:.3f} MCC={r['mcc']:.4f} test_FPR={r['fpr']:.4f} "
              f"(val_FPR={r['validation_fpr_at_threshold']:.4f})")

    if not a.skip_onnx and not a.tag:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        import onnxruntime as ort
        onx = convert_sklearn(clf, initial_types=[("input", FloatTensorType([None, Xtr.shape[1]]))],
                              options={id(clf): {"zipmap": False}}, target_opset=15)
        path = res_dir / "models" / "rf_sklearn.onnx"
        path.write_bytes(onx.SerializeToString())
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        sub = Xte[:5000]
        p_onnx = sess.run(None, {"input": sub})[1][:, 1]
        diff = float(np.max(np.abs(p_onnx - clf.predict_proba(sub)[:, 1])))
        print(f"ONNX parity: max |p_onnx - p_sklearn| on {len(sub)} rows = {diff:.2e}")
        (res_dir / "tables" / "onnx_parity.json").write_text(json.dumps({"max_abs_diff": diff, "rows": len(sub)}))
        if diff > 1e-4:
            print("WARNING: parity above 1e-4; investigate before using the ONNX arm in the paper.")


if __name__ == "__main__":
    main()
