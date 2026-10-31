"""Run a benchmark CAMPAIGN: several complete, independent runs of all arms, each run in a randomized order.

A run executes every selected arm once (each arm in its own fresh process; every REST server is started
fresh for its arms and stopped afterwards). The order of the eight blocks (five in-process arms and three
server groups) and the order of arms inside each server group are shuffled per run with a logged seed,
so no arm always runs first, last, or after a particular other arm.

Usage (from the project root, virtual environment active):
  python scripts/run_campaign.py --campaign main --runs 10
  python scripts/run_campaign.py --campaign main --dry-run                 # print the schedule only
  python scripts/run_campaign.py --campaign sens_warmup20k --runs 3 --warmup 20000 --arms A1,A3,A4,A5
  python scripts/run_campaign.py --campaign sens_nopin --runs 3 --no-pin
  python scripts/run_campaign.py --campaign main --runs 4 --start-run 7    # continue an interrupted campaign

Outputs: results/campaigns/<campaign>/
  campaign.json   settings, environment, versions, power status, git commit, start/end time
  schedule.csv    one row per arm execution: run, position, arm, start/end time, duration, exit code, affinity
  run_01/raw/     latency (<label>.csv), REST breakdown, scores and JSON sidecars of run 1, and so on
  logs/           stdout/stderr of every client and server process
Analyse with:  python python/src/s09_campaign_stats.py --campaign main
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import random
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import psutil
import requests

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "python" / "src"
sys.path.insert(0, str(SRC))
from common import load_config, env_info, disable_power_throttling  # noqa: E402

DATA = ROOT / "data" / "processed"
MODELS = ROOT / "results" / "models"
TABLES = ROOT / "results" / "tables"
WIN = sys.platform == "win32"

# Arm registry. IDs match the paper (Table II).
ARMS = {
    "A1": dict(label="py_sklearn_njobs1", kind="py_inproc", arm="sklearn_njobs1"),
    "A2": dict(label="py_sklearn_njobsall", kind="py_inproc", arm="sklearn_njobsall"),
    "A3": dict(label="py_onnxruntime", kind="py_inproc", arm="onnxruntime"),
    "A4": dict(label="jvm_onnx_inprocess", kind="java_inproc", arm="onnx", model="rf_sklearn.onnx"),
    "A5": dict(label="jvm_smile_inprocess", kind="java_inproc", arm="smile", model="rf_smile.ser"),
    "A6": dict(label="rest_flaskdev_njobsall_nokeepalive", kind="py_rest", server="S1", keepalive=False),
    "A7": dict(label="rest_waitress_sklearn_keepalive", kind="py_rest", server="S2", keepalive=True),
    "A8": dict(label="rest_waitress_sklearn_nokeepalive", kind="py_rest", server="S2", keepalive=False),
    "A9": dict(label="jvm_rest_waitress_sklearn", kind="java_rest", server="S2"),
    "A10": dict(label="rest_waitress_onnx_keepalive", kind="py_rest", server="S3", keepalive=True),
    "A11": dict(label="jvm_rest_waitress_onnx", kind="java_rest", server="S3"),
}
SERVER_DESC = {"S1": "flask-dev, sklearn n_jobs=-1", "S2": "waitress, sklearn n_jobs=1", "S3": "waitress, onnxruntime 1 thread"}


def server_args(sid: str, threads: int) -> list[str]:
    return {"S1": ["--server", "flask-dev", "--backend", "sklearn", "--n-jobs", "-1"],
            "S2": ["--server", "waitress", "--backend", "sklearn", "--n-jobs", "1", "--threads", str(threads)],
            "S3": ["--server", "waitress", "--backend", "onnx", "--n-jobs", "1", "--threads", str(threads)]}[sid]


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def pom_property(name: str) -> str | None:
    m = re.search(rf"<{re.escape(name)}>([^<]+)</{re.escape(name)}>", (ROOT / "java" / "pom.xml").read_text())
    return m.group(1).strip() if m else None


def power_status() -> dict:
    info: dict = {}
    try:
        b = psutil.sensors_battery()
        info["on_ac_power"] = None if b is None else bool(b.power_plugged)
        info["battery_percent"] = None if b is None else b.percent
    except Exception:
        info["on_ac_power"] = None
    if WIN:
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Power\PowerThrottling")
            info["PowerThrottlingOff"] = winreg.QueryValueEx(k, "PowerThrottlingOff")[0]
        except OSError:
            info["PowerThrottlingOff"] = None
        try:
            info["power_scheme"] = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True,
                                                  text=True, timeout=10).stdout.strip()
        except Exception:
            info["power_scheme"] = None
    return info


def java_version(java: str) -> str:
    r = subprocess.run([java, "-version"], capture_output=True, text=True)
    return " | ".join(l for l in r.stderr.splitlines() if "JAVA_TOOL_OPTIONS" not in l)


def git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return r.stdout.strip() + (" (uncommitted changes)" if dirty else "") if r.returncode == 0 else None
    except Exception:
        return None


def port_open(host: str, port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


class Runner:
    def __init__(self, a, cfg):
        self.a, self.cfg = a, cfg
        b, c = cfg["benchmark"], cfg.get("campaign", {})
        self.host, self.port = b["service_host"], b["service_port"]
        self.n = a.n or b["measured_requests"]
        self.warmup = b["warmup_requests"] if a.warmup is None else a.warmup
        self.threads = c.get("waitress_threads", 4)
        self.heap = c.get("java_heap", "4g")
        self.pause = c.get("pause_between_arms_s", 5) if a.pause is None else a.pause
        self.cpus = None if a.no_pin else c.get("cpus", [0, 1, 2, 3])
        self.py = sys.executable
        self.java = a.java or shutil.which("java") or "java"
        self.jar = Path(a.jar) if a.jar else ROOT / "java" / "target" / "bench.jar"
        self.dir = ROOT / "results" / "campaigns" / a.campaign
        self.logs = self.dir / "logs"
        self.flags = subprocess.HIGH_PRIORITY_CLASS if (WIN and not a.no_pin) else 0

    # ------------------------------------------------------------------ process control
    def pin_self(self):
        me = psutil.Process()
        if self.cpus is None:
            try:
                me.cpu_affinity(list(range(psutil.cpu_count())))
            except Exception:
                pass
        else:
            me.cpu_affinity(self.cpus)          # children inherit the affinity mask (Windows and Linux)
            if WIN:
                me.nice(psutil.HIGH_PRIORITY_CLASS)
        disable_power_throttling()
        if WIN:   # keep the system awake for as long as this process runs (reverts automatically on exit)
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

    def spawn(self, cmd, log: Path, cwd=SRC):
        f = open(log, "w", encoding="utf-8")
        p = subprocess.Popen([str(c) for c in cmd], cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
                             creationflags=self.flags)
        if WIN:
            disable_power_throttling(p.pid)
        try:
            aff = psutil.Process(p.pid).cpu_affinity()
        except Exception:
            aff = None
        return p, f, aff

    def start_server(self, sid: str, tag: str):
        deadline = time.time() + 20
        while port_open(self.host, self.port) and time.time() < deadline:
            time.sleep(0.5)
        if port_open(self.host, self.port):
            raise RuntimeError(f"port {self.port} is still in use; close whatever is listening on it and retry")
        cmd = [self.py, "s04_serve.py", *server_args(sid, self.threads)]
        p, f, aff = self.spawn(cmd, self.logs / f"{tag}_server_{sid}.log")
        t0 = time.time()
        while time.time() - t0 < 120:
            if p.poll() is not None:
                raise RuntimeError(f"server {sid} exited early; see {self.logs / f'{tag}_server_{sid}.log'}")
            try:
                if requests.get(f"http://{self.host}:{self.port}/health", timeout=2).ok:
                    return p, f, aff
            except requests.RequestException:
                pass
            time.sleep(0.5)
        self.stop(p, f)
        raise RuntimeError(f"server {sid} did not answer /health within 120 s")

    @staticmethod
    def stop(p, f):
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(15)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(15)
        f.close()

    # ------------------------------------------------------------------ commands
    def cmd(self, aid: str, run_dir: Path):
        s = ARMS[aid]
        raw = run_dir / "raw"
        if s["kind"] == "py_inproc":
            return [self.py, "s05_bench_inprocess.py", "--arm", s["arm"], "--warmup", self.warmup,
                    "--n", self.n, "--raw-dir", raw]
        if s["kind"] == "py_rest":
            c = [self.py, "s06_bench_rest.py", "--label", s["label"], "--warmup", self.warmup, "--n", self.n,
                 "--raw-dir", raw, "--server-desc", SERVER_DESC[s["server"]]]
            return c + ([] if s["keepalive"] else ["--no-keepalive"])
        java = [self.java, f"-Xmx{self.heap}", "-XX:+UseG1GC", "-jar", self.jar, "latency",
                "--label", s["label"], "--rows", DATA / "bench_rows.csv", "--warmuprows", DATA / "warmup_rows.csv",
                "--warmup", self.warmup, "--n", self.n, "--out", run_dir]
        if s["kind"] == "java_inproc":
            return java + ["--arm", s["arm"], "--model", MODELS / s["model"]]
        return java + ["--arm", "rest", "--url", f"http://{self.host}:{self.port}/predict"]

    # ------------------------------------------------------------------ schedule
    def schedule(self, arms: list[str], run: int) -> list[tuple[str, list[str]]]:
        """Blocks in randomized order: ('-', [inproc arm]) or (server id, [arms of that server, shuffled])."""
        rng = random.Random(self.a.seed * 1000 + run)
        blocks: list[tuple[str, list[str]]] = [("-", [x]) for x in arms if "server" not in ARMS[x]]
        for sid in ("S1", "S2", "S3"):
            members = [x for x in arms if ARMS[x].get("server") == sid]
            if members:
                rng.shuffle(members)
                blocks.append((sid, members))
        rng.shuffle(blocks)
        return blocks

    # ------------------------------------------------------------------ checks
    def preflight(self, arms: list[str]) -> dict:
        problems, warnings = [], []
        for f in ("bench_rows.csv", "warmup_rows.csv", "meta.json", "arrays.npz"):
            if not (DATA / f).exists():
                problems.append(f"missing data/processed/{f}: run python/src/s02_prepare_data.py")
        meta = json.loads((DATA / "meta.json").read_text()) if (DATA / "meta.json").exists() else {}
        cw = meta.get("class_weight_used")
        needs_sk = any(ARMS[x]["kind"] != "java_inproc" or ARMS[x]["arm"] == "onnx" for x in arms)
        if needs_sk:
            for f in ("rf_sklearn.joblib", "rf_sklearn.onnx", "thresholds_sklearn.json"):
                if not (MODELS / f).exists():
                    problems.append(f"missing results/models/{f}: run python/src/s03_train_sklearn.py")
            th = MODELS / "thresholds_sklearn.json"
            if th.exists() and cw is not None and json.loads(th.read_text()).get("fraud_class_weight") != cw:
                problems.append(f"sklearn model was not trained with the training-only class weight {cw}: rerun s03_train_sklearn.py")
        if "A5" in arms:
            info = TABLES / "train_smile_info.json"
            if not (MODELS / "rf_smile.ser").exists():
                problems.append("missing results/models/rf_smile.ser: run the Java 'train' command")
            elif not info.exists() or json.loads(info.read_text()).get("fraud_weight") != cw:
                problems.append(f"Smile model was not trained with the training-only weight {cw}: rerun 'java -jar target/bench.jar train'")
        if any(ARMS[x]["kind"].startswith("java") for x in arms):
            if not self.jar.exists():
                problems.append(f"missing {self.jar}: run 'mvn -q package' in the java folder")
        try:
            from importlib import metadata
            py_ort = metadata.version("onnxruntime")
        except Exception:
            py_ort = None
        jv_ort = pom_property("onnxruntime.version")
        if any(x in arms for x in ("A3", "A4")) and py_ort != jv_ort:
            msg = f"ONNX Runtime versions differ: Python {py_ort} vs Java {jv_ort} (pom.xml)"
            (warnings if self.a.allow_version_mismatch else problems).append(msg)
        if any("server" in ARMS[x] for x in arms) and port_open(self.host, self.port):
            problems.append(f"something is already listening on {self.host}:{self.port}; close it first")
        pw = power_status()
        if pw.get("on_ac_power") is False:
            warnings.append("laptop is on BATTERY: plug in the charger")
        if WIN and pw.get("PowerThrottlingOff") != 1 and not self.a.no_pin:
            warnings.append("registry PowerThrottlingOff is not 1 (per-process opt-out is still applied)")
        return {"problems": problems, "warnings": warnings, "power": pw, "py_ort": py_ort, "java_ort": jv_ort,
                "class_weight": cw}

    # ------------------------------------------------------------------ main loop
    def run(self, arms: list[str]):
        runs = list(range(self.a.start_run, self.a.start_run + self.a.runs))
        if self.a.dry_run:
            for r in runs:
                order = [x for _, blk in self.schedule(arms, r) for x in blk]
                print(f"run {r:02d}: " + "  ".join(order))
            return 0
        pf = self.preflight(arms)
        for w in pf["warnings"]:
            print("WARNING:", w)
        if pf["problems"]:
            print("\nCannot start:")
            for p in pf["problems"]:
                print("  -", p)
            return 2
        existing = [r for r in runs if (self.dir / f"run_{r:02d}").exists()]
        if existing and not self.a.overwrite:
            print(f"run folders already exist for runs {existing} in {self.dir}; use --start-run or --overwrite")
            return 2
        self.logs.mkdir(parents=True, exist_ok=True)
        self.pin_self()
        meta_path = self.dir / "campaign.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"sessions": []}
        session = {"started": now(), "runs": runs, "arms": arms, "seed": self.a.seed, "n_measured": self.n,
                   "warmup": self.warmup, "warmup_source": "validation month (warmup_rows.csv)",
                   "pinned_cpus": self.cpus, "priority": "high" if (WIN and self.cpus is not None) else "normal",
                   "pause_between_arms_s": self.pause, "waitress_threads": self.threads, "java_heap": self.heap,
                   "class_weight": pf["class_weight"], "onnxruntime_python": pf["py_ort"],
                   "onnxruntime_java": pf["java_ort"], "smile": pom_property("smile.version"),
                   "java": java_version(self.java), "env": env_info(), "power": pf["power"],
                   "warnings": pf["warnings"], "git_commit": git_commit(), "platform": platform.platform(),
                   "argv": sys.argv[1:]}
        meta["sessions"].append(session)
        meta_path.write_text(json.dumps(meta, indent=2))

        sched_path = self.dir / "schedule.csv"
        new = not sched_path.exists()
        sf = open(sched_path, "a", newline="", encoding="utf-8")
        sw = csv.writer(sf)
        if new:
            sw.writerow(["run", "position", "arm_id", "label", "server", "start", "end", "duration_s",
                         "exit_code", "client_affinity", "server_affinity"])
        failures = []
        try:
            if not self.a.no_machine_warmup:
                print("machine warm-up (discarded) ...", flush=True)
                d = self.dir / "warmup_discard"
                p, f, _ = self.spawn(self.cmd("A1", d)[:-4] + ["--n", 500, "--raw-dir", d / "raw"],
                                     self.logs / "machine_warmup.log")
                p.wait(); f.close()
            for r in runs:
                run_dir = self.dir / f"run_{r:02d}"
                (run_dir / "raw").mkdir(parents=True, exist_ok=True)
                blocks = self.schedule(arms, r)
                total = sum(len(b) for _, b in blocks)
                pos = 0
                print(f"\n=== run {r:02d}  order: " + "  ".join(x for _, b in blocks for x in b), flush=True)
                for sid, members in blocks:
                    server = None
                    try:
                        if sid != "-":
                            server = self.start_server(sid, f"run{r:02d}")
                        for aid in members:
                            pos += 1
                            tag = f"run{r:02d}_{pos:02d}_{aid}"
                            start = now(); t0 = time.time()
                            p, f, aff = self.spawn(self.cmd(aid, run_dir), self.logs / f"{tag}.log",
                                                   cwd=SRC if ARMS[aid]["kind"].startswith("py") else ROOT / "java")
                            code = p.wait(); f.close()
                            dur = time.time() - t0
                            sw.writerow([r, pos, aid, ARMS[aid]["label"], sid if sid != "-" else "", start, now(),
                                         f"{dur:.1f}", code, aff, server[2] if server else ""])
                            sf.flush()
                            summary = next((l.strip() for l in (self.logs / f"{tag}.log").read_text(
                                encoding="utf-8", errors="replace").splitlines()
                                if l.startswith(ARMS[aid]["label"] + ":")), "")
                            print(f"[run {r:02d} {pos:2d}/{total}] {aid:3s} {summary or 'FAILED (exit %d)' % code}",
                                  flush=True)
                            if code != 0:
                                failures.append(tag)
                            time.sleep(self.pause)
                    except RuntimeError as e:
                        print(f"[run {r:02d}] block {sid} failed: {e}", flush=True)
                        failures.append(f"run{r:02d}_{sid}")
                    finally:
                        if server:
                            self.stop(server[0], server[1])
        finally:
            sf.close()
            session["finished"] = now()
            session["failures"] = failures
            meta_path.write_text(json.dumps(meta, indent=2))
        print(f"\nDone. Results in {self.dir}")
        if failures:
            print("FAILED arms (see logs):", ", ".join(failures))
            return 1
        print(f"Next: python python/src/s09_campaign_stats.py --campaign {self.a.campaign}")
        return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", required=True, help="name of the output folder under results/campaigns")
    ap.add_argument("--runs", type=int, default=None, help="number of complete runs (default: config campaign.runs)")
    ap.add_argument("--start-run", type=int, default=1)
    ap.add_argument("--arms", default="all", help="comma list of arm IDs (A1..A11) or 'all'")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n", type=int, default=None, help="measured calls per arm (default config)")
    ap.add_argument("--warmup", type=int, default=None, help="warm-up calls per arm (default config)")
    ap.add_argument("--pause", type=float, default=None)
    ap.add_argument("--no-pin", action="store_true", help="sensitivity check: no CPU pinning, normal priority")
    ap.add_argument("--no-machine-warmup", action="store_true")
    ap.add_argument("--java", default=None)
    ap.add_argument("--jar", default=None)
    ap.add_argument("--allow-version-mismatch", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    c = cfg.get("campaign", {})
    a.runs = a.runs or c.get("runs", 10)
    a.seed = c.get("seed", 1) if a.seed is None else a.seed
    arms = list(ARMS) if a.arms == "all" else [x.strip().upper() for x in a.arms.split(",")]
    bad = [x for x in arms if x not in ARMS]
    if bad:
        raise SystemExit(f"unknown arm IDs: {bad}")
    sys.exit(Runner(a, cfg).run(arms))


if __name__ == "__main__":
    main()
