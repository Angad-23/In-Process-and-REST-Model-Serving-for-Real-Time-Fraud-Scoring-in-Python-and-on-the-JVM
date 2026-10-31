"""Round-trip latency against the REST service (s04_serve.py must be running).

  python s06_bench_rest.py --label rest_waitress_sklearn_keepalive
  python s06_bench_rest.py --label rest_waitress_sklearn_nokeepalive --no-keepalive

Client settings (recorded in the JSON sidecar):
  * python-requests; keep-alive = one requests.Session (HTTP/1.1, pooled connection);
    --no-keepalive = requests.post() per call, i.e. a new TCP connection per request
  * JSON body {"features": [...29 float32 values...]}, Content-Type application/json, timeout 30 s
  * URL uses 127.0.0.1 (not "localhost": on Windows that may try IPv6 first and add delay)

Per measured call the client records (microseconds, perf_counter_ns):
  total        = client_serialize + round_trip + client_parse      (the latency reported per arm)
  round_trip   = from sending the request to receiving the full response
  server_handler / server_parse / model    from the X-*-us response headers (see s04_serve.py)
  outside_handler = round_trip - server_handler  (loopback network stack + WSGI server + framework)
Written to <label>.csv (total), <label>_breakdown.csv (all components) and <label>_scores.csv.
"""
import argparse
import json
import time
from importlib import metadata

import numpy as np
import requests
from common import (load_config, sample_rows, warmup_rows, write_raw, write_scores, raw_dir,
                    disable_power_throttling)

HEADERS = {"Content-Type": "application/json"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--no-keepalive", action="store_true")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--server-desc", default="", help="free text: how the server was started")
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    disable_power_throttling()
    cfg = load_config(a.config)
    b = cfg["benchmark"]
    n = a.n or b["measured_requests"]
    nw = b["warmup_requests"] if a.warmup is None else a.warmup
    base = f"http://{a.host or b['service_host']}:{a.port or b['service_port']}"
    url = base + "/predict"
    rows = sample_rows(cfg, n)
    warm = warmup_rows(cfg, nw)
    out = raw_dir(cfg, a.raw_dir)
    server_cfg = requests.get(base + "/health", timeout=10).json().get("config", {})

    sess = None if a.no_keepalive else requests.Session()
    post = (lambda data: requests.post(url, data=data, headers=HEADERS, timeout=30)) if sess is None \
        else (lambda data: sess.post(url, data=data, headers=HEADERS, timeout=30))

    for r in warm:
        post(json.dumps({"features": r.tolist()})).raise_for_status()

    cols = ("total", "client_serialize", "round_trip", "client_parse",
            "server_handler", "server_parse", "model", "outside_handler")
    bd = np.empty((n, len(cols)))
    scores = np.empty(n)
    for i, r in enumerate(rows):
        body = {"features": r.tolist()}
        t0 = time.perf_counter_ns()
        data = json.dumps(body)
        t1 = time.perf_counter_ns()
        resp = post(data)
        t2 = time.perf_counter_ns()
        p = resp.json()["prob"]
        t3 = time.perf_counter_ns()
        resp.raise_for_status()
        h = resp.headers
        handler = float(h["X-Handler-us"])
        rt = (t2 - t1) / 1000.0
        bd[i] = ((t3 - t0) / 1000.0, (t1 - t0) / 1000.0, rt, (t3 - t2) / 1000.0,
                 handler, float(h["X-Parse-us"]), float(h["X-Model-us"]), rt - handler)
        scores[i] = p

    client = {"library": "python-requests", "requests": metadata.version("requests"),
              "urllib3": metadata.version("urllib3"), "keepalive": not a.no_keepalive,
              "connection": "one requests.Session (pooled, HTTP/1.1)" if sess else "new TCP connection per request",
              "timeout_s": 30, "body": "JSON, 29 float32 features"}
    write_raw(cfg, a.label, bd[:, 0], {"batch_size": 1, "warmup": nw,
                                       "warmup_source": "validation month (warmup_rows.csv)",
                                       "measured_source": "test month (bench_rows.csv)", "seed": b["seed"],
                                       "keepalive": not a.no_keepalive, "server_desc": a.server_desc, "url": url,
                                       "client": client, "server_config": server_cfg}, out)
    np.savetxt(out / f"{a.label}_breakdown.csv", bd, delimiter=",", header=",".join(cols), comments="", fmt="%.3f")
    write_scores(out, a.label, scores)
    med = np.median(bd, axis=0)
    print("  breakdown medians (us): " + ", ".join(f"{c}={m:.1f}" for c, m in zip(cols, med)))


if __name__ == "__main__":
    main()
