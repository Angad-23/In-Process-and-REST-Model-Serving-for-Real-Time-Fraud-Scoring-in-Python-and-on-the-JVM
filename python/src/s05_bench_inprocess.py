"""In-process single-row latency (no network): the model-cost floor for Python.

Arms (run ONE arm per process with --arm, so a campaign can randomize the order of arms):
  sklearn_njobs1     -> py_sklearn_njobs1     RandomForest.predict_proba, n_jobs=1
  sklearn_njobsall   -> py_sklearn_njobsall   same model, n_jobs=-1 (what the old paper accidentally used)
  onnxruntime        -> py_onnxruntime        the ONNX copy of the same model via ONNX Runtime (1 intra-op thread)

Protocol (identical for every arm, Python and Java):
  * warm-up: --warmup calls (default config.benchmark.warmup_requests) on VALIDATION rows
    (warmup_rows.csv), discarded; the measured TEST rows are never used for warm-up
  * measured: the first --n rows of bench_rows.csv (TEST month), one transaction per call,
    each call timed with time.perf_counter_ns()
  * the fraud probability returned by every measured call is saved (<label>_scores.csv) for parity checks
"""
import argparse
import time
import joblib
import numpy as np
from common import (load_config, resolve, sample_rows, warmup_rows, write_raw, write_scores,
                    raw_dir, disable_power_throttling)

ARMS = {"sklearn_njobs1": "py_sklearn_njobs1", "sklearn_njobsall": "py_sklearn_njobsall",
        "onnxruntime": "py_onnxruntime"}


def time_calls(fn, rows, warm):
    for r in warm:
        fn(r[None, :])
    lat = np.empty(len(rows))
    scores = np.empty(len(rows))
    for i, r in enumerate(rows):
        x = r[None, :]
        t0 = time.perf_counter_ns()
        p = fn(x)
        lat[i] = (time.perf_counter_ns() - t0) / 1000.0
        scores[i] = p
    return lat, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--arm", choices=list(ARMS) + ["all"], default="all")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--raw-dir", default=None, help="output folder (default results/raw)")
    a = ap.parse_args()
    disable_power_throttling()
    cfg = load_config(a.config)
    b = cfg["benchmark"]
    n = a.n or b["measured_requests"]
    nw = b["warmup_requests"] if a.warmup is None else a.warmup
    rows = sample_rows(cfg, n)
    warm = warmup_rows(cfg, nw)
    out = raw_dir(cfg, a.raw_dir)
    mdir = resolve(cfg, cfg["paths"]["results_dir"]) / "models"
    extra = {"batch_size": 1, "warmup": nw, "warmup_source": "validation month (warmup_rows.csv)",
             "measured_source": "test month (bench_rows.csv)", "seed": b["seed"], "client": "python in-process"}
    todo = list(ARMS) if a.arm == "all" else [a.arm]

    for arm in todo:
        label = ARMS[arm]
        if arm.startswith("sklearn"):
            clf = joblib.load(mdir / "rf_sklearn.joblib")
            nj = 1 if arm == "sklearn_njobs1" else -1
            clf.set_params(n_jobs=nj)
            lat, sc = time_calls(lambda x: clf.predict_proba(x)[0, 1], rows, warm)
            write_raw(cfg, label, lat, {**extra, "n_jobs": nj}, out)
        else:
            import onnxruntime as ort
            so = ort.SessionOptions(); so.intra_op_num_threads = 1
            sess = ort.InferenceSession(str(mdir / "rf_sklearn.onnx"), so, providers=["CPUExecutionProvider"])
            lat, sc = time_calls(lambda x: sess.run(None, {"input": x})[1][0, 1], rows, warm)
            write_raw(cfg, label, lat, {**extra, "intra_op_threads": 1, "onnxruntime": ort.__version__}, out)
        write_scores(out, label, sc)


if __name__ == "__main__":
    main()
