# Resubmission changes (IEEE Access, manuscript Access-2026-49207)

Decision of 25 Oct 2026: rejected with resubmission allowed (one resubmission only). This file maps every
reviewer concern to the code change that addresses it and to the output that provides the evidence.
Use it when writing the response to reviewers.

| # | Concern (reviewer) | Code change | Evidence produced |
|---|---|---|---|
| 1 | Class weight derived from the full dataset (R1, R2) | `fraud_class_weight: auto` = round(#legit/#fraud) in the training months only, computed identically in Python (`common.train_class_weight`) and Java (`TrainSmile --fraudweight auto`); both models retrained; runner refuses stale models | `data/processed/meta.json`, `results/tables/preprocessing_audit.json`, `train_smile_info.json`, `metrics_sklearn.json` |
| 1b | All preprocessing / model choices from training data only (R1) | Audit file written by `s02_prepare_data.py` | `results/tables/preprocessing_audit.json` |
| 1c | Sensitivity: old weight 90 (R1) | `s03_train_sklearn.py --fraud-weight 90 --tag w90`; `bench.jar train --fraudweight 90 --tag w90` | `metrics_sklearn_w90.json`, `metrics_smile_w90.json` |
| 2 | Runs, not calls, are the unit of evidence; report every run (R1) | `scripts/run_campaign.py` (10 complete runs); `s09_campaign_stats.py` (run-level statistics: geometric-mean ratios with t-intervals over runs, Wilcoxon signed-rank over runs) | `campaigns/main/analysis/per_run.csv`, `summary.*`, `contrasts.*` |
| 3 | Randomize / counterbalance arm order; clarify the schedule (R1) | Every run shuffles the 8 blocks (5 in-process arms, 3 server groups) and the arms inside each server group, seeded and logged | `campaigns/main/schedule.csv` (start/end of every arm), `order_effects.csv` |
| 4 | Separate warm-up data; equal Python/JVM warm-up (R1) | `warmup_rows.csv` from the VALIDATION month; 2,000 warm-up calls for every arm (Python and Java) | sidecar JSON of every arm (`warmup`, `warmup_source`) |
| 4b | Sensitivity: long warm-up (R1) | campaign `sens_warmup20k` (20,000 warm-up calls; A1, A3, A4, A5) | `campaigns/sens_warmup20k/analysis/sensitivity_vs_main.*` |
| 5 | Separate client, transport, server and inference time; document settings (R1) | `s04_serve.py` returns X-Parse-us / X-Model-us / X-Handler-us; both clients record serialize, round trip, parse; `/health` reports exact server settings, saved in every sidecar | `*_breakdown.csv`, `campaigns/main/analysis/rest_breakdown.*`, sidecar `server_config`, `client` |
| 6 | Actual test FPR at the validation threshold (R1, R2) | `evaluate()` reports `fpr`; scripts print test FPR next to validation FPR | `metrics_*.json` (`fpr`, `validation_fpr_at_threshold`) |
| 7 | Different ONNX Runtime versions (R2) | Java 1.30.0 = Python 1.30.0 (`pom.xml`, `requirements.txt`); runner refuses a mismatch | `campaign.json` (`onnxruntime_python`, `onnxruntime_java`) |
| 8 | Sensitivity: processor pinning (R1) | campaign `sens_nopin` (`--no-pin`: no affinity, normal priority) | `campaigns/sens_nopin/analysis/sensitivity_vs_main.*` |
| 9 | Within-run bootstrap intervals are not between-run uncertainty (R2) | `s07_stats.py` labelled as within-run; main evidence from `s09` | - |
| 10 | Parity in every arm | every arm (Python and Java) saves the score of every measured call | `campaigns/main/analysis/parity.csv` |

Text-only concerns (manuscript): limit claims to the tested model/dataset/laptop/Windows/loopback/single request;
"no consistent speed advantage" instead of "never faster"; "measured deployment overhead" instead of isolated
transport cost; remove template placeholders; verify software/web citations; IEEE math formatting.

## Additional fixes found during the code audit
* The old self-test deleted trained models and committed result tables; it now skips when real data or models exist
  and restores `results/tables`.
* Every Python and Java process opts out of Windows power throttling itself (in addition to the registry setting),
  and the runner keeps Windows awake while it runs.
* Server and client output goes to log files (no console windows to click on).
