"""WITHIN-RUN statistics + figures from the raw latency files of ONE run (one column 'latency_us').

NOTE: the confidence intervals here describe uncertainty WITHIN a single run (resampling its calls); they do
not capture variation BETWEEN independent runs. The paper's main evidence is run-level:
see s09_campaign_stats.py.  Use --raw-dir results/campaigns/<name>/run_01/raw to analyse one run.

Works for Python AND Java arms: both write the same format.

Outputs (results/tables, results/figures):
  latency_summary.csv / .md     n, mean, sd, median, P95, P99, P99.9, min, max, bootstrap 95% CIs
  latency_comparisons.csv / .md ratio of medians (with CI), Mann-Whitney U p, Cliff's delta
  latency_ecdf.png, latency_box.png

Bootstrap uses a moving-block scheme (default block=20) because consecutive timings
are autocorrelated (GC, JIT, CPU frequency), so plain i.i.d. resampling gives CIs that are too narrow.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_config, resolve


def block_resample(x, rng, block):
    n = len(x)
    if block <= 1:
        return x[rng.integers(0, n, n)]
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, nb)
    return np.concatenate([x[s:s + block] for s in starts])[:n]


def boot_ci(x, fn, rng, B, block):
    vals = np.array([fn(block_resample(x, rng, block)) for _ in range(B)])
    return np.percentile(vals, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--reference", default="py_onnxruntime", help="arm used as denominator/reference")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--raw-dir", default=None, help="folder with raw files (default results/raw)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    rdir = resolve(cfg, cfg["paths"]["results_dir"])
    src = Path(a.raw_dir).resolve() if a.raw_dir else rdir / "raw"
    files = sorted(f for f in src.glob("*.csv") if not f.stem.endswith(("_scores", "_breakdown")))
    data = {f.stem: pd.read_csv(f)["latency_us"].to_numpy() for f in files if f.stem not in a.exclude}
    if not data:
        raise SystemExit("no raw latency files found in results/raw")
    rng = np.random.default_rng(12345)

    rows = []
    for name, x in data.items():
        ci_med = boot_ci(x, np.median, rng, a.B, a.block)
        ci_p99 = boot_ci(x, lambda v: np.percentile(v, 99), rng, a.B, a.block)
        rows.append({"arm": name, "n": len(x), "mean_us": x.mean(), "sd_us": x.std(ddof=1),
                     "median_us": np.median(x), "median_ci_lo": ci_med[0], "median_ci_hi": ci_med[1],
                     "p95_us": np.percentile(x, 95), "p99_us": np.percentile(x, 99),
                     "p99_ci_lo": ci_p99[0], "p99_ci_hi": ci_p99[1],
                     "p99.9_us": np.percentile(x, 99.9), "min_us": x.min(), "max_us": x.max(),
                     "cv": x.std(ddof=1) / x.mean()})
    summ = pd.DataFrame(rows).round(2)
    (rdir / "tables").mkdir(exist_ok=True, parents=True)
    (rdir / "figures").mkdir(exist_ok=True, parents=True)
    summ.to_csv(rdir / "tables" / "latency_summary.csv", index=False)
    (rdir / "tables" / "latency_summary.md").write_text(summ.to_markdown(index=False))

    comp = []
    if a.reference in data:
        ref = data[a.reference]
        for name, x in data.items():
            if name == a.reference:
                continue
            u, p = stats.mannwhitneyu(x, ref, alternative="two-sided")
            delta = 2 * u / (len(x) * len(ref)) - 1          # Cliff's delta (x vs ref)
            ratios = np.array([np.median(block_resample(x, rng, a.block)) /
                               np.median(block_resample(ref, rng, a.block)) for _ in range(a.B)])
            lo, hi = np.percentile(ratios, [2.5, 97.5])
            comp.append({"arm": name, "reference": a.reference,
                         "median_ratio_arm_over_ref": np.median(x) / np.median(ref),
                         "ratio_ci_lo": lo, "ratio_ci_hi": hi,
                         "mannwhitney_p": p, "cliffs_delta": delta})
        cdf = pd.DataFrame(comp).round(4)
        cdf.to_csv(rdir / "tables" / "latency_comparisons.csv", index=False)
        (rdir / "tables" / "latency_comparisons.md").write_text(cdf.to_markdown(index=False))
        print(cdf.to_string(index=False))
    else:
        print(f"reference arm '{a.reference}' not found; skipping pairwise comparisons. Arms: {list(data)}")

    # figures
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, x in data.items():
        xs = np.sort(x)
        ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), label=name, lw=1.4)
    ax.set_xscale("log"); ax.set_xlabel("latency per request (µs, log scale)"); ax.set_ylabel("ECDF")
    ax.grid(alpha=.3, which="both"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(rdir / "figures" / "latency_ecdf.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.boxplot(list(data.values()), showfliers=False, whis=(1, 99))
    ax.set_xticks(range(1, len(data) + 1)); ax.set_xticklabels(list(data.keys()))
    ax.set_yscale("log"); ax.set_ylabel("latency per request (µs, log scale)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7); ax.grid(alpha=.3, axis="y", which="both")
    fig.tight_layout(); fig.savefig(rdir / "figures" / "latency_box.png", dpi=300); plt.close(fig)
    print("\n", summ[["arm", "n", "median_us", "p95_us", "p99_us"]].to_string(index=False))


if __name__ == "__main__":
    main()
