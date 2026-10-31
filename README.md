# Cross-Language ML Inference Latency for JVM Fraud-Scoring Stacks

A reproducible benchmark that separates **where the time goes** when a Java/JVM banking stack scores a
transaction with a model trained in Python: in-process Python, a REST bridge (several server configurations),
in-process JVM inference of a Smile forest, and in-process JVM inference of the *same* scikit-learn forest via ONNX Runtime.

This is the rebuilt version of the earlier "JVM-Native Real-Time Fraud Detection" manuscript. The old design compared an
in-process JVM call with a full HTTP round trip through Flask's development server, so it could not say how much of the
gap came from REST and how much from the model library. This project fixes that. See `docs/01_project_plan.md` for the
list of reviewer-level problems it addresses.

> **Status:** code scaffold complete; **no real results yet.** Every number must come from running the protocol in
> `docs/03_experiment_protocol.md` on your machine. Nothing in this repo contains reportable results.

## What is in the box

```
jvm-fraud-benchmark/
├── README.md                     this file
├── python/
│   ├── config.yaml               ONE place for split, hyperparameters, benchmark sizes
│   ├── requirements.txt
│   ├── src/
│   │   ├── common.py             loading, temporal split, encoding, metrics, benchmark helpers
│   │   ├── s00_make_synthetic_baf.py   fake data with BAF layout (smoke test only)
│   │   ├── s01_inspect_data.py         facts about the data (month range, dtypes, -1 counts)
│   │   ├── s02_prepare_data.py         temporal split + encoding; writes CSVs for Java too
│   │   ├── s03_train_sklearn.py        train, evaluate, export ONNX, parity check
│   │   ├── s04_serve.py                REST service (Flask dev | waitress; sklearn | onnx)
│   │   ├── s05_bench_inprocess.py      in-process Python latency arms
│   │   ├── s06_bench_rest.py           REST round-trip latency (Python client)
│   │   ├── s07_stats.py                WITHIN-run CIs, Mann-Whitney, Cliff's delta, ECDF/box plots
│   │   ├── s08_evaluate_scores.py      identical metrics for any model (Smile scores from Java), incl. test FPR
│   │   └── s09_campaign_stats.py       RUN-level statistics for a campaign (main evidence)
│   └── tests/test_pipeline.py    smoke test on synthetic data
├── java/
│   ├── pom.xml                   Smile 3.1.1, ONNX Runtime 1.30.0 (= Python version), shaded jar
│   └── src/main/java/bench/      Main, TrainSmile, LatencyBench, SmileArm, OnnxArm, RestArm, Csv, Arm
├── data/        raw/ (put Base.csv here), processed/ (generated)
├── results/     raw/ (per-request latencies), tables/, figures/, models/
├── docs/        plan, methodology, protocol, threats, reference audit, paper outline, checklist, notebook, known gaps
├── paper/       manuscript work area
└── scripts/     run_campaign.py (randomized multi-run campaigns), resubmission_A/B_*.ps1, run_python_steps.ps1 / .sh
```

## Resubmission workflow (October 2026)

The IEEE Access reviewers' concerns and the code changes that address them are listed in
`docs/11_resubmission_changes.md`. To reproduce the resubmitted results:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\resubmission_A_prepare_and_train.ps1   # data + models (~20 min)
.venv\Scripts\python.exe scripts\run_campaign.py --campaign main --dry-run           # shows the randomized order
powershell -ExecutionPolicy Bypass -File scripts\resubmission_B_campaigns.ps1          # all campaigns (~3-4 h)
```
Results: `results/campaigns/<campaign>/analysis/`.

## Quick start (Windows 11)

Prerequisites: Python 3.10+, JDK 21, Maven 3.9+ (`java -version`, `mvn -v`, `python --version` must all work).

```powershell
# 0. get the data: download Base.csv from Kaggle into data\raw\   (see data\README.md)

# 1. Python pipeline (creates .venv, installs packages, inspects, splits, trains, exports ONNX)
.\scripts\run_python_steps.ps1
```
Open `results\tables\data_inspection.txt`, check the month range, and adjust `train_months / val_months / test_months`
in `python\config.yaml` if the range is not 0-7. Then re-run the script.

```powershell
# 2. Java: build, train the Smile forest on the same split
cd java
mvn -q package
java -Xmx8g -XX:+UseG1GC -jar target\bench.jar train
cd ..
cd python\src
python s08_evaluate_scores.py --name smile        # same metric code as scikit-learn
cd ..\..
```

Latency arms, statistics and the full run order are in **`docs/03_experiment_protocol.md`**. The short version:

| Arm label | What it measures | Command (from `python\src` unless noted) |
|---|---|---|
| `py_sklearn_njobs1` / `py_sklearn_njobsall` / `py_onnxruntime` | in-process Python, no network | `python s05_bench_inprocess.py` |
| `rest_flaskdev_nokeepalive_njobsall` | replica of the old baseline | server: `python s04_serve.py --server flask-dev --n-jobs -1`, client: `python s06_bench_rest.py --label ... --no-keepalive` |
| `rest_waitress_keepalive_njobs1` / `rest_waitress_nokeepalive_njobs1` | tuned REST bridge, with and without connection reuse | server: `python s04_serve.py --server waitress --n-jobs 1` |
| `rest_waitress_keepalive_onnx` | tuned bridge + ONNX backend | server: `... --backend onnx` |
| `jvm_rest_client_waitress` | Java client calling the Python service (the real bridge) | `java -jar target\bench.jar latency --arm rest --label jvm_rest_client_waitress` (from `java`) |
| `jvm_smile_inprocess` | Smile forest inside the JVM | `java -Xmx8g -XX:+UseG1GC -jar target\bench.jar latency --arm smile` |
| `jvm_onnx_inprocess` | same sklearn forest via ONNX Runtime in the JVM | `... latency --arm onnx` |

Then `python s07_stats.py --reference jvm_onnx_inprocess`.

## Design decisions that matter

1. **Temporal split by month**, validation month used only for thresholds, test months untouched. No random shuffle.
2. **Encoders fitted on train only.** BAF's `-1` missing marker is kept as-is for tree models.
3. **Identical rows in identical order** for every latency arm: Python writes `bench_rows.csv`, Java replays it.
4. **Same trees where it matters:** the ONNX arm runs the exact scikit-learn forest, so its predictions match to ~1e-7
   and latency differences cannot be blamed on accuracy differences. The Smile forest is a *different implementation*
   and is reported as such (its accuracy is measured, not assumed equal).
5. **Raw per-request latencies are saved**, and statistics use block bootstrap CIs, Mann-Whitney U and Cliff's delta.
6. **Warm-up:** 20,000 discarded calls for JVM in-process arms (not 5). Measured: 5,000 requests per arm (configurable).
7. **Metrics that fit imbalance:** ROC-AUC, PR-AUC, TPR at 1% and 5% FPR, precision/recall/MCC at validation-chosen thresholds.
   Accuracy is reported only next to the majority-class baseline.

## Testing status (honest)

| Component | Tested here? |
|---|---|
| All Python scripts (`s00`-`s08`), on synthetic data | Yes: full pipeline, REST servers, stats, figures |
| Java `Csv`, `LatencyBench`, `RestArm`, `Main` (compiled with plain `javac`) | Yes: ran against the Python service; scores matched scikit-learn to 9 digits |
| Java `TrainSmile`, `SmileArm`, `OnnxArm`, `pom.xml` | **No.** Maven Central was unreachable from the build sandbox. Written against Smile 3.1.1 and ONNX Runtime 1.17.1 APIs from memory. See `docs/09_known_gaps.md` |

## License
Code: MIT (`LICENSE`). Data: not included; BAF is CC BY-NC-SA 4.0 per its Kaggle page. See `data/README.md`.
