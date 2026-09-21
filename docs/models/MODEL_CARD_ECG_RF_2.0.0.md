# Model Card: ECG-RF-2.0.0-candidate

## Model Details
- **Developer:** ECG Guardian Autonomous ML Pipeline
- **Model Date:** September 2026
- **Model Version:** 2.0.0-candidate
- **Model Type:** Balanced Ensemble Random Forest Classifier
- **Input Representation:** 33 morphological and interval features extracted from 360 Hz single-lead ECG beat windows (-200ms to +400ms around R-peak).
- **Output Classes:** `Normal`, `PAC`, `PVC`, `Other`
- **Primary Training Objective:** High-sensitivity detection of Premature Ventricular Contractions (PVC) and Premature Atrial Contractions (PAC).

## Intended Use
- **Primary Intended Use:** Educational, algorithmic research, and physician review copilot in simulated hospital environments.
- **Out-of-Scope Use:** Autonomous diagnostic decisions, primary triage without physician confirmation, pediatric ECG analysis without specialized training.

## Training Data & Provenance
- **Dataset Source:** MIT-BIH Arrhythmia Database (PhysioNet).
- **Patient Isolation:** Train patients (['100', '106', '208', '213']), Validation patient (200), Test patients (['101', '119']). Zero beat-level cross-contamination.
- **Preprocessing:** Butterworth bandpass filter (0.5–40 Hz), notch filter (50 Hz), lead-wise z-score normalization. Scaler statistics fit strictly on train split.

## Performance Metrics (Independent Test Split)
- **Accuracy:** 98.33%
- **Weighted F1 Score:** 98.31%
- **PVC Sensitivity (Recall):** 99.67%
- **PVC Specificity:** 98.06%
- **AUROC (Weighted):** 0.9973
- **AUPRC (Weighted):** 0.9966
- **Expected Calibration Error (ECE):** 0.0989

## Regulatory & Ethical Disclaimers
> [!WARNING]
> This model is an investigative, non-clinical research software component. It is **NOT** approved by CDSCO, US FDA, or CE Mark notified bodies for diagnostic use. Probability values represent algorithmic softmax outputs and must NOT be interpreted as true clinical diagnostic certainty.
