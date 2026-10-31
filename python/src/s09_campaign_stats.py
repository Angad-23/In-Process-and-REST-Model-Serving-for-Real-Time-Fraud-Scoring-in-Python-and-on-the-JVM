"""Run-level statistics for a benchmark campaign (results/campaigns/<name>/run_XX/raw).

The unit of analysis is the COMPLETE RUN (Kalibera & Jones 2013; Georges et al. 2007): each run yields one
summary per arm (median, p95, p99 of its 5,000 calls), and uncertainty is computed ACROSS runs. The
5,000 calls inside a run are serially dependent repeated measurements, not independent observations.

Outputs in results/campaigns/<name>/analysis/:
  per_run.csv            every run x arm: n, median, p95, p99, mean, ratio of medians to the reference arm
                         in the same run, position of the arm in that run's randomized order
  summary.csv / .md      per arm across runs: median and range of run medians, between-run CV, median p95/p99,
                         geometric-mean ratio to the reference with a 95% t-interval over runs
  contrasts.csv / .md    the paper's comparisons (engine, runtime, deployment overhead, keep-alive, ...):
                         per-run ratios, geometric mean, 95% CI, min-max, #runs with ratio > 1,
                         Wilcoxon signed-rank test on log ratios across runs
  rest_breakdown.csv/.md REST arms: client serialize, round trip, server handler, model, outside handler,
                         client parse (median over runs of each run's median)
  parity.csv             max |score - score of A1 (sklearn)| per run and arm (Smile compared with itself)
  order_effects.csv      Spearman correlation between an arm's position in the run and its log median
  per_run_medians.png    run medians of every arm (log scale)
Optional --baseline <campaign>: compares this (sensitivity) campaign with the baseline campaign per arm.

Usage:  python s09_campaign_stats.py --campaign main
        python s09_campaign_stats.py --campaign sens_nopin --baseline main
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_config, resolve

IDS = {"py_sklearn_njobs1": "A1", "py_sklearn_njobsall": "A2", "py_onnxruntime": "A3",
       "jvm_onnx_inprocess": "A4", "jvm_smile_inprocess": "A5", "rest_flaskdev_njobsall_nokeepalive": "A6",
       "rest_waitress_sklearn_keepalive": "A7", "rest_waitress_sklearn_nokeepalive": "A8",
       "jvm_rest_waitress_sklearn": "A9", "rest_waitress_onnx_keepalive": "A10", "jvm_rest_waitress_onnx": "A11"}
LABEL = {v: k for k, v in IDS.items()}
ORDER = [f"A{i}" for i in range(1, 12)]

# (name, numerator, denominator, statistic) -- what each comparison answers is described in the paper
CONTRASTS = [
    ("engine: sklearn vs ONNX Runtime, same Python process", "A1", "A3", "median"),
    ("runtime: ONNX Runtime in JVM vs in CPython (same model, same ORT version)", "A4", "A3", "median"),
    ("JVM-native Smile vs ONNX Runtime in CPython", "A5", "A3", "median"),
    ("deployment overhead, ONNX, Java client (A11 vs A4)", "A11", "A4", "median"),
    ("deployment overhead, ONNX, Python client (A10 vs A3)", "A10", "A3", "median"),
    ("deployment overhead, sklearn, Python client (A7 vs A1)", "A7", "A1", "median"),
    ("client language against the same ONNX server (A10 vs A11)", "A10", "A11", "median"),
    ("client language against the same sklearn server (A7 vs A9)", "A7", "A9", "median"),
    ("no keep-alive vs keep-alive, median (A8 vs A7)", "A8", "A7", "median"),
    ("no keep-alive vs keep-alive, p99 (A8 vs A7)", "A8", "A7", "p99"),
    ("n_jobs=-1 vs n_jobs=1 in-process (A2 vs A1)", "A2", "A1", "median"),
    ("naive replica vs tuned sklearn service (A6 vs A7)", "A6", "A7", "median"),
    ("naive replica vs tuned ONNX service (A6 vs A10)", "A6", "A10", "median"),
]
SKLEARN_FAMILY = ["A1", "A2", "A3", "A4", "A6", "A7", "A8", "A9", "A10", "A11"]


def run_dirs(cdir: Path):
    return sorted(d for d in cdir.glob("run_*") if (d / "raw").is_dir())


def load_campaign(cdir: Path):
    rows, scores, breakdowns = [], {}, {}
    sched = pd.read_csv(cdir / "schedule.csv") if (cdir / "schedule.csv").exists() else None
    for d in run_dirs(cdir):
        run = int(d.name.split("_")[1])
        for f in sorted((d / "raw").glob("*.csv")):
            stem = f.stem
            if stem.endswith("_scores") or stem.endswith("_breakdown") or stem not in IDS:
                continue
            x = pd.read_csv(f)["latency_us"].to_numpy()
            aid = IDS[stem]
            pos = None
            if sched is not None:
                m = sched[(sched.run == run) & (sched.arm_id == aid)]
                pos = int(m.position.iloc[-1]) if len(m) else None
            rows.append({"run": run, "arm": aid, "label": stem, "n": len(x), "median_us": np.median(x),
                         "p95_us": np.percentile(x, 95), "p99_us": np.percentile(x, 99), "mean_us": x.mean(),
                         "position": pos})
            sf = d / "raw" / f"{stem}_scores.csv"
            if sf.exists():
                scores[(run, aid)] = pd.read_csv(sf)["score"].to_numpy()
            bf = d / "raw" / f"{stem}_breakdown.csv"
            if bf.exists():
                breakdowns[(run, aid)] = pd.read_csv(bf).median()
    return pd.DataFrame(rows), scores, breakdowns


def gm_ci(r):
    """Geometric mean of ratios with a 95% t-interval on the log scale (runs = independent observations)."""
    lr = np.log(np.asarray(r, dtype=float))
    k = len(lr)
    gm = float(np.exp(lr.mean()))
    if k < 2:
        return gm, np.nan, np.nan
    h = stats.t.ppf(0.975, k - 1) * lr.std(ddof=1) / np.sqrt(k)
    return gm, float(np.exp(lr.mean() - h)), float(np.exp(lr.mean() + h))


def wilcoxon_p(r):
    lr = np.log(np.asarray(r, dtype=float))
    if len(lr) < 2 or np.allclose(lr, 0):
        return np.nan
    try:
        return float(stats.wilcoxon(lr, alternative="two-sided").pvalue)
    except ValueError:
        return np.nan


def fmt_md(df):
    return df.to_markdown(index=False, floatfmt=".4g")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--reference", default="A3", help="reference arm ID (default A3, ONNX Runtime in CPython)")
    ap.add_argument("--baseline", default=None, help="compare against this campaign (sensitivity analyses)")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    croot = resolve(cfg, cfg["paths"]["results_dir"]) / "campaigns"
    cdir = croot / a.campaign
    out = cdir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    per, scores, bds = load_campaign(cdir)
    if per.empty:
        raise SystemExit(f"no run folders with raw files in {cdir}")
    ref = a.reference
    refmed = per[per.arm == ref].set_index("run")["median_us"]
    per["ratio_to_ref"] = per.apply(lambda r: r.median_us / refmed.get(r.run, np.nan), axis=1)
    per["arm_order"] = per.arm.map({x: i for i, x in enumerate(ORDER)})
    per = per.sort_values(["arm_order", "run"]).drop(columns="arm_order")
    per.round(3).to_csv(out / "per_run.csv", index=False)
    n_runs = per.run.nunique()

    # ---- summary per arm
    summ = []
    for aid in [x for x in ORDER if x in set(per.arm)]:
        s = per[per.arm == aid]
        med = s.median_us.to_numpy()
        rat = s.ratio_to_ref.dropna().to_numpy()
        gm, lo, hi = gm_ci(rat) if len(rat) else (np.nan,) * 3
        summ.append({"arm": aid, "label": LABEL[aid], "runs": len(s),
                     "median_of_run_medians_us": np.median(med), "min_run_median_us": med.min(),
                     "max_run_median_us": med.max(), "between_run_cv": med.std(ddof=1) / med.mean() if len(med) > 1 else np.nan,
                     "median_p95_us": s.p95_us.median(), "median_p99_us": s.p99_us.median(),
                     f"gm_ratio_to_{ref}": gm, "ratio_ci_lo": lo, "ratio_ci_hi": hi,
                     "ratio_min": rat.min() if len(rat) else np.nan, "ratio_max": rat.max() if len(rat) else np.nan})
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(fmt_md(summ))

    # ---- contrasts (paired by run)
    piv = {stat: per.pivot(index="run", columns="arm", values=f"{stat}_us") for stat in ("median", "p99")}
    con = []
    for name, num, den, stat in CONTRASTS:
        p = piv[stat]
        if num not in p or den not in p:
            continue
        r = (p[num] / p[den]).dropna()
        if r.empty:
            continue
        gm, lo, hi = gm_ci(r)
        con.append({"contrast": name, "numerator": num, "denominator": den, "statistic": stat, "runs": len(r),
                    "gm_ratio": gm, "ci_lo": lo, "ci_hi": hi, "min": r.min(), "max": r.max(),
                    "runs_ratio_gt_1": int((r > 1).sum()), "wilcoxon_p": wilcoxon_p(r),
                    "median_difference_us": float(np.median(p[num][r.index] - p[den][r.index])),
                    "per_run_ratios": " ".join(f"{v:.3g}" for v in r)})
    con = pd.DataFrame(con)
    con.to_csv(out / "contrasts.csv", index=False)
    (out / "contrasts.md").write_text(fmt_md(con.drop(columns="per_run_ratios")) if not con.empty else "")

    # ---- REST breakdown
    if bds:
        b = pd.DataFrame([{**v.to_dict(), "run": k[0], "arm": k[1]} for k, v in bds.items()])
        cols = ["total", "client_serialize", "round_trip", "client_parse", "server_handler", "server_parse",
                "model", "outside_handler"]
        rb = b.groupby("arm")[cols].median()
        rb = rb.loc[[x for x in ORDER if x in rb.index]].reset_index()
        rb.insert(1, "runs", b.groupby("arm").size().reindex(rb.arm).to_numpy())
        rb.to_csv(out / "rest_breakdown.csv", index=False)
        (out / "rest_breakdown.md").write_text(
            "Median over runs of each run's median, microseconds. outside_handler = round_trip - server_handler.\n\n"
            + fmt_md(rb))

    # ---- parity
    par = []
    runs = sorted(per.run.unique())
    smile_ref = next((scores[(r, "A5")] for r in runs if (r, "A5") in scores), None)
    for r in runs:
        base = scores.get((r, "A1"))
        for aid in ORDER:
            sc = scores.get((r, aid))
            if sc is None:
                continue
            if aid == "A5":
                ref_s, against = smile_ref, "A5 (first run)"
            elif aid in SKLEARN_FAMILY and aid != "A1":
                ref_s, against = base, "A1"
            else:
                continue
            if ref_s is None or len(ref_s) != len(sc):
                continue
            par.append({"run": r, "arm": aid, "compared_with": against,
                        "max_abs_diff": float(np.max(np.abs(sc - ref_s))), "score_sum": float(sc.sum())})
    par = pd.DataFrame(par)
    par.to_csv(out / "parity.csv", index=False)

    # ---- order effects
    oe = []
    for aid in [x for x in ORDER if x in set(per.arm)]:
        s = per[(per.arm == aid) & per.position.notna()]
        if len(s) >= 4 and s.position.nunique() > 1:
            rho, p = stats.spearmanr(s.position, np.log(s.median_us))
            oe.append({"arm": aid, "runs": len(s), "positions_used": s.position.nunique(),
                       "spearman_rho": rho, "p_value": p})
    pd.DataFrame(oe).to_csv(out / "order_effects.csv", index=False)

    # ---- figure
    arms = [x for x in ORDER if x in set(per.arm)]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, aid in enumerate(arms):
        y = per[per.arm == aid].median_us
        ax.scatter(y, np.full(len(y), i) + np.random.default_rng(i).uniform(-.15, .15, len(y)), s=12)
    ax.set_yticks(range(len(arms))); ax.set_yticklabels([f"{x} {LABEL[x]}" for x in arms], fontsize=7)
    ax.invert_yaxis(); ax.set_xscale("log"); ax.set_xlabel("run median latency per transaction (µs, log scale)")
    ax.grid(alpha=.3, axis="x", which="both"); ax.set_title(f"{a.campaign}: {n_runs} runs, one dot per run", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "per_run_medians.png", dpi=300); plt.close(fig)

    # ---- sensitivity vs baseline campaign
    if a.baseline:
        bper, _, _ = load_campaign(croot / a.baseline)
        rows = []
        for aid in [x for x in ORDER if x in set(per.arm) and x in set(bper.arm)]:
            m1, m0 = per[per.arm == aid].median_us, bper[bper.arm == aid].median_us
            rows.append({"arm": aid, "runs": len(m1), "baseline_runs": len(m0),
                         "median_of_run_medians_us": m1.median(), "baseline_median_us": m0.median(),
                         "ratio_to_baseline": m1.median() / m0.median(),
                         "between_run_cv": m1.std(ddof=1) / m1.mean() if len(m1) > 1 else np.nan,
                         "baseline_between_run_cv": m0.std(ddof=1) / m0.mean() if len(m0) > 1 else np.nan,
                         "mannwhitney_p_runs": stats.mannwhitneyu(m1, m0).pvalue if len(m1) > 1 and len(m0) > 1 else np.nan})
        sens = pd.DataFrame(rows)
        sens.to_csv(out / f"sensitivity_vs_{a.baseline}.csv", index=False)
        (out / f"sensitivity_vs_{a.baseline}.md").write_text(fmt_md(sens))
        print(f"\nSensitivity vs {a.baseline}:\n" + sens.to_string(index=False))

    print(f"{a.campaign}: {n_runs} runs, {per.arm.nunique()} arms\n")
    print(summ[["arm", "runs", "median_of_run_medians_us", "min_run_median_us", "max_run_median_us",
                "between_run_cv", f"gm_ratio_to_{ref}", "ratio_ci_lo", "ratio_ci_hi"]].to_string(index=False))
    if not con.empty:
        print("\n" + con[["contrast", "runs", "gm_ratio", "ci_lo", "ci_hi", "min", "max", "runs_ratio_gt_1",
                          "wilcoxon_p"]].to_string(index=False))
    if not par.empty:
        print(f"\nparity: max |diff| over all runs and sklearn-family arms = "
              f"{par[par.arm != 'A5'].max_abs_diff.max():.2e}; Smile across runs = "
              f"{par[par.arm == 'A5'].max_abs_diff.max() if (par.arm == 'A5').any() else float('nan'):.2e}")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
