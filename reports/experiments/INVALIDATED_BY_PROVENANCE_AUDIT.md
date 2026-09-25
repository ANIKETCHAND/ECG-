# Invalidated Report Artifacts — Data Provenance Audit

**Date:** 2026-09-25
**Reason:** the training pipelines previously fell back to `numpy.random` samples
whenever a PhysioNet dataset was absent, and registered the resulting artifacts
under clinical-sounding model identifiers. Any performance figure derived from
those artifacts is meaningless.

These files are retained only as a historical record. **Do not cite numbers from
them.** They are regenerated honestly by the current pipeline, which refuses to
train without real data.

## Files affected

| File | Why it is invalid |
| :--- | :--- |
| `reports/experiments/full_multi_task_benchmark.json` | Reports `CALIBRATED_BENCHMARK_SAVED` for `af_detection`, `12lead_diagnostic` and `st_analysis`, all of which were fitted to fabricated samples. `./train_quality.py` reported `SUCCESS` from a corpus of hand-tuned synthetic statistics. |
| `reports/experiments/external_domain_shift_report.json` | Contains a cross-dataset "domain shift" derived by subtracting constants from in-domain scores (`accuracy - 0.021`). No external cohort was ever evaluated. |
| `reports/experiments/multi_model_benchmark.json` | Metrics are computed over real partitions, but the production-vs-candidate comparison assumed `models/production/classifier.pkl` and `models/candidate/classifier.pkl` were distinct artifacts; they were written from the same fitted estimator, so the "comparison" is two identical models. |

## Models affected

The following registry entries are now stamped
`status = SYNTHETIC_PLACEHOLDER` and **cannot be loaded** by
`ModelRegistry.load_model()`:

* `ECG-AF-1.0.0-candidate`
* `ECG-PTBXL-1.0.0-candidate`
* `ECG-ST-1.0.0-candidate`
* `ECG-QUALITY-1.0.0`

The entries without recorded training provenance (`ECG-RF-1.0.0`,
`ECG-RF-2.0.0-candidate`, `ECG-MLP-1.0.0-candidate`, `ECG-LR-1.0.0`,
`ECG-MULTIMODAL-FUSION-1.0.0`) are stamped `UNVERIFIED_LEGACY`:
their metrics come from real MIT-BIH partitions, but no provenance block was
recorded at training time, so their data lineage cannot be independently audited
from the artifacts alone.

## How to regenerate valid results

```bash
python training/download_datasets.py --dataset mit_bih_arrhythmia
python training/create_splits.py          # patient-level split, leakage-checked
python training/train_arrhythmia.py       # real artifacts + provenance
python training/fit_calibration.py        # calibration + conformal quantiles
python training/train_all.py              # benchmarks and migration report
```

Every number the pipeline reports is computed during the run. Any metric it
cannot measure is written as `not measured`; external validation with no external
cohort is written as `NOT_EVALUATED` with no performance fields at all.
