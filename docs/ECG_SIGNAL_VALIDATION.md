# Biomedical ECG Signal Processing & Waveform Validation

## 1. Input Processing Architecture

ECG Guardian supports two distinct classes of electrocardiographic input:
1. **Native Digital Waveforms**: Direct numerical voltage samples (CSV, JSON array, WFDB/MIT-BIH, DICOM-ECG, EDF).
2. **Optical Waveform Extractions**: Scanned paper ECG strips, photographs, and PDF document renderings.

These inputs are **never treated as equivalent**. Native digital signals preserve calibrated millivolt scales and microsecond-level timing accuracy, whereas optical extractions are subject to camera distortion, perspective skew, and lack of physical grid calibration.

---

## 2. Ingress Signal Quality & Validation Pipeline

Before inference or measurement extraction, all incoming signals traverse the Biomedical Validation Gate (`src/biomedical/signal_quality.py`):

```
Raw Input Signal 
       │
       ▼
1. Sampling Frequency Normalization (Resampled to 360 Hz standard via SciPy polyphase)
       │
       ▼
2. Baseline Wander Removal (0.5 Hz highpass Butterworth / median dual-window filter)
       │
       ▼
3. High-Frequency & Powerline Suppression (50/60 Hz notch filter + 45 Hz lowpass filter)
       │
       ▼
4. Signal Quality Index (SQI) Assessment
   ├── Flatline Detection: Zero-variance or dead-lead test
   ├── Clipping Detection: ADC saturation test (threshold: > 98% dynamic range)
   ├── SNR & High-Frequency Noise Ratio: Spectral power analysis
   └── P-QRS-T Physiological Plausibility: Heart rate between 30 and 240 bpm
       │
       ▼
Validated Signal Object -> Passed to Pan-Tompkins QRS Extractor & Model Tier
```

---

## 3. Optical Waveform Extraction Overhaul (BIO-01 Fix)

### 3.1 Historical Defect Analysis
In legacy versions, optical ECG extraction in `src/ecg_input/waveform_extractor.py` suffered from two severe engineering defects:
1. **Width-Based Duration Guessing**:
   ```python
   # Legacy code:
   assumed_duration_sec = 10.0 if w >= 1000 else 2.5
   ```
   Pixel width of an image varies with camera resolution and zoom level, not physical paper strip speed (25 mm/s).
2. **Arbitrary Peak Rescaling to 1.5 mV**:
   ```python
   # Legacy code:
   final_signal = (final_signal / max_peak) * 1.5
   ```
   Forcing the highest peak to 1.5 mV artificially inflated or suppressed clinical voltages, completely invalidating ST-segment elevation/depression measurements and Left Ventricular Hypertrophy (Sokolow-Lyon) criteria.

### 3.2 Implemented Scientific Remediation
1. **Preservation of Relative Signal Morphology**:
   - Arbitrary 1.5 mV scaling has been removed.
   - For uncalibrated images, the waveform is centered around its median baseline and standardized via z-score normalization (`signal = (signal - median) / std`).
2. **Mandatory Uncalibrated Warning & Feature Disabling**:
   - The extraction result explicitly sets:
     - `is_calibrated: false`
     - `can_evaluate_st: false`
     - `calibration_status: "UNCALIBRATED_OPTICAL_EXTRACT"`
     - `clinical_warning: "ST-segment elevation and amplitude criteria withheld: Missing physical grid calibration."`
3. **Clinical Policy**:
   - Diagnostic decisions requiring absolute millivolt precision (STEMI, ST-depression ischemia, LVH) are barred when `can_evaluate_st` is false.
   - Only rhythm interpretation (rate, regularity, ectopy) is permitted on optical strips pending physical grid detection calibration.

---

## 4. Dynamic Morphological Fiducial Detection (BIO-02)

### 4.1 Remediation of Fixed Q-Point Calculation
- **Historical Defect**: QT interval calculation previously assumed `q_point = t_pre` (hardcoded at R - 50ms), producing constant interval distortions regardless of actual Q-wave morphology, PR interval variation, or intraventricular conduction delays.
- **Dynamic Q-Onset Algorithm**:
  1. Identifies the pre-R search window `[R - 70ms, R]`.
  2. Computes the minimum voltage inflection point (Q trough).
  3. Traces backward along the first derivative to locate the baseline departure inflection where `|d/dt| < 0.10 * max(|d/dt|)`.
  4. Measures the true QT interval from the dynamic Q-onset to the T-wave offset baseline return.
- **Validation**: Tested against standard MIT-BIH sinus beats, yielding physiological QT intervals in the 360–440 ms range and correct Bazett/Fridericia rate-corrected intervals.
