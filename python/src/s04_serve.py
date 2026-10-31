"""REST inference service (the "Python bridge").

Modes (pick one; report which you used in the paper):
  --server flask-dev   Flask/Werkzeug development server, single-threaded (threaded=False), which speaks
                       HTTP/1.0 and closes the connection after every response; logs every request.
                       Reproduces the old paper's baseline; NOT for production.
  --server waitress    multi-threaded production WSGI server (works on Windows), HTTP/1.1 keep-alive.

--n-jobs sets the model's INFERENCE thread count. n_jobs=-1 on a single row spawns
threads with large overhead; n_jobs=1 is the fair setting for single-transaction scoring.
--backend onnx serves the ONNX copy through ONNX Runtime instead of scikit-learn.

Endpoint: POST /predict   {"features": [f1, ..., fD]}   ->   {"prob": 0.0123}
          GET  /health    -> {"ok": true, "config": {...}}   (exact server settings, recorded by every client)

Server-side timing (reviewer request: separate client, transport, server and model time).
Every /predict response carries these headers, in microseconds, measured with time.perf_counter_ns():
  X-Parse-us    request.get_json + conversion to a float32 array
  X-Model-us    the model call only (predict_proba or ONNX Runtime run)
  X-Handler-us  the whole view function, up to the moment the response object is built
The client subtracts X-Handler-us from its measured round trip to obtain the time spent outside the
handler: network stack (loopback), the WSGI server and the framework (request routing, response writing).
"""
import argparse
import json
import platform
import sys
import time
from importlib import metadata

import joblib
import numpy as np
from flask import Flask, Response, request
from common import load_config, resolve, disable_power_throttling


def _ver(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def create_app(model_file="rf_sklearn.joblib", n_jobs=1, backend="sklearn", config=None, server_info=None):
    cfg = load_config(config)
    mdir = resolve(cfg, cfg["paths"]["results_dir"]) / "models"
    app = Flask(__name__)

    if backend == "onnx":
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(n_jobs) if int(n_jobs) > 0 else 0
        sess = ort.InferenceSession(str(mdir / "rf_sklearn.onnx"), so, providers=["CPUExecutionProvider"])

        def score(x):
            return float(sess.run(None, {"input": x})[1][0, 1])
    else:
        clf = joblib.load(mdir / model_file)
        clf.set_params(n_jobs=int(n_jobs))

        def score(x):
            return float(clf.predict_proba(x)[0, 1])

    info = {"backend": backend, "n_jobs": int(n_jobs), "python": sys.version.split()[0],
            "platform": platform.platform(), "flask": _ver("flask"), "werkzeug": _ver("werkzeug"),
            "waitress": _ver("waitress"), "scikit-learn": _ver("scikit-learn"),
            "onnxruntime": _ver("onnxruntime"), **(server_info or {})}
    model_us = []

    @app.post("/predict")
    def predict():
        t0 = time.perf_counter_ns()
        x = np.asarray(request.get_json(force=True)["features"], dtype=np.float32).reshape(1, -1)
        t1 = time.perf_counter_ns()
        p = score(x)
        t2 = time.perf_counter_ns()
        body = json.dumps({"prob": p})
        t3 = time.perf_counter_ns()
        model_us.append((t2 - t1) / 1000.0)
        if len(model_us) % 1000 == 0:
            print(f"[server] requests={len(model_us)} model-only median over last 1000 = "
                  f"{np.median(model_us[-1000:]):.1f}us", flush=True)
        return Response(body, mimetype="application/json", headers={
            "X-Parse-us": f"{(t1 - t0) / 1000.0:.3f}",
            "X-Model-us": f"{(t2 - t1) / 1000.0:.3f}",
            "X-Handler-us": f"{(t3 - t0) / 1000.0:.3f}"})

    @app.get("/health")
    def health():
        return Response(json.dumps({"ok": True, "config": info}), mimetype="application/json")

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", choices=["flask-dev", "waitress"], default="waitress")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--backend", choices=["sklearn", "onnx"], default="sklearn")
    ap.add_argument("--threads", type=int, default=4, help="waitress worker threads")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    disable_power_throttling()
    cfg = load_config(a.config)
    host = a.host or cfg["benchmark"]["service_host"]
    port = a.port or cfg["benchmark"]["service_port"]
    if a.server == "flask-dev":
        sinfo = {"server": "flask-dev (werkzeug run_simple)", "threaded": False, "http": "HTTP/1.0, connection closed after each response",
                 "request_logging": "werkzeug logs every request (stdout redirected to a log file by run_campaign.py)"}
    else:
        sinfo = {"server": "waitress", "threads": a.threads, "http": "HTTP/1.1 keep-alive",
                 "waitress_settings": "defaults except threads (connection_limit=100, channel_timeout=120 s)"}
    sinfo.update({"host": host, "port": port})
    app = create_app("rf_sklearn.joblib", a.n_jobs, a.backend, a.config, sinfo)
    print(f"serving backend={a.backend} n_jobs={a.n_jobs} server={a.server} on {host}:{port}", flush=True)
    if a.server == "flask-dev":
        app.run(host=host, port=port, threaded=False)
    else:
        from waitress import serve
        serve(app, host=host, port=port, threads=a.threads)


if __name__ == "__main__":
    main()
