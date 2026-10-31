"""Quick self-test on a small SYNTHETIC file: prepare -> train -> one short campaign of the Python arms -> stats.
Run:  python -m pytest python/tests -q     (or: python python/tests/test_pipeline.py)
Never report numbers from this test. It cleans up after itself.
SAFETY: it overwrites data/processed and results/models, so it SKIPS when real data or trained models exist.
Run it only in a fresh clone.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
ROOT = SRC.parents[1]


def run(*args, cwd=SRC):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=cwd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_smoke():
    import pytest
    if (ROOT / "data" / "raw" / "Base.csv").exists() or any((ROOT / "results" / "models").glob("rf_*")):
        pytest.skip("real data or trained models present; run the self-test only in a fresh clone")
    csv = ROOT / "data" / "raw" / "Base_synthetic.csv"
    camp = ROOT / "results" / "campaigns" / "_selftest"
    tables = ROOT / "results" / "tables"
    backup = Path(tempfile.mkdtemp()) / "tables"
    shutil.copytree(tables, backup)          # committed result tables are restored afterwards
    try:
        run("s00_make_synthetic_baf.py", "--rows", "30000", "--out", csv)
        out = run("s02_prepare_data.py", "--csv", csv)
        assert "training months only" in out
        meta = json.loads((ROOT / "data" / "processed" / "meta.json").read_text())
        assert meta["class_weight_used"] == meta["class_weight_train_only"]
        assert (ROOT / "data" / "processed" / "warmup_rows.csv").exists()
        out = run("s03_train_sklearn.py")
        assert "ONNX parity" in out and "test_FPR=" in out
        run(ROOT / "scripts" / "run_campaign.py", "--campaign", "_selftest", "--runs", "2", "--arms", "A1,A3",
            "--n", "50", "--warmup", "20", "--pause", "0", "--no-machine-warmup", "--allow-version-mismatch", cwd=ROOT)
        out = run("s09_campaign_stats.py", "--campaign", "_selftest")
        assert "engine: sklearn vs ONNX Runtime" in out
        per = (camp / "analysis" / "per_run.csv").read_text()
        assert per.count("\n") == 1 + 2 * 2   # header + 2 runs x 2 arms
    finally:
        csv.unlink(missing_ok=True)
        shutil.rmtree(camp, ignore_errors=True)
        for f in (ROOT / "data" / "processed").glob("*"):
            if f.name != ".gitkeep":
                f.unlink()
        for f in (ROOT / "results" / "models").glob("*"):
            if f.name != ".gitkeep":
                f.unlink()
        shutil.rmtree(tables, ignore_errors=True)
        shutil.copytree(backup, tables)
        shutil.rmtree(backup.parent, ignore_errors=True)


if __name__ == "__main__":
    test_smoke(); print("smoke test passed")
