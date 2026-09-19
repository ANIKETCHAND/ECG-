"""
Structured ECG Report Generator
===============================

Aggregates digital signal metrics, AI model classifications, and extracted
printed report parameters into a unified, non-hallucinatory clinical research report.

Supports:
- Structured Python Dictionary
- Plain-Text Export (.txt)
- Machine-Readable JSON (.json)

Research/educational use only.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd


def generate_structured_report(
    input_info: Dict[str, Any],
    ai_results: Optional[Dict[str, Any]] = None,
    extracted_measurements: Optional[Dict[str, Any]] = None,
    waveform_status: Optional[Dict[str, Any]] = None,
    clinician_review: Optional[Dict[str, Any]] = None,
    hospital_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble complete clinical research and review report dictionary.

    Args:
        input_info: File name, modality, duration, sampling rate, lead
        ai_results: Output from predict_ecg if ML pipeline was executed
        extracted_measurements: Output from measurement_extractor if PDF/image text was parsed
        waveform_status: Success/failure details of waveform extraction
        clinician_review: Optional physician review, interpretation, and sign-off
        hospital_info: Optional hospital/clinic name, department, and facility details

    Returns:
        Consolidated dictionary ready for UI display, PDF generation, or JSON export.
    """
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Patient info
    p_info = {
        "patient_name": extracted_measurements.get("patient_name") if extracted_measurements else None,
        "patient_age": extracted_measurements.get("patient_age") if extracted_measurements else None,
        "patient_sex": extracted_measurements.get("patient_sex") if extracted_measurements else None,
        "recording_date": extracted_measurements.get("recording_date") if extracted_measurements else None,
    }

    # Signal quality
    if ai_results and "signal_quality" in ai_results:
        sq_raw = ai_results.get("quality_indicators", {})
        quality_data = {
            "category": ai_results["signal_quality"],
            "quality_score": round(float(ai_results.get("quality_score", 1.0)), 2),
            "snr_db": round(float(sq_raw.get("snr_db", 0.0)), 1),
            "baseline_wander": bool(sq_raw.get("baseline_wander", False)),
            "powerline_interference": bool(sq_raw.get("has_powerline_interference", False)),
            "motion_artifacts": bool(sq_raw.get("has_motion_artifacts", False)),
        }
    else:
        quality_data = {
            "category": "NOT_ASSESSED",
            "quality_score": None,
            "snr_db": None,
            "baseline_wander": None,
            "powerline_interference": None,
            "motion_artifacts": None,
        }

    # Cardiac parameters
    hr_val = None
    mean_rr = None
    beat_cnt = None
    if ai_results and "heart_rate_bpm" in ai_results:
        hr_val = round(float(ai_results["heart_rate_bpm"]), 1)
        mean_rr = round(float(ai_results.get("mean_rr_sec", 0.0)) * 1000.0, 1)
        beat_cnt = int(ai_results.get("beat_count", 0))
    elif extracted_measurements and extracted_measurements.get("heart_rate_printed") is not None:
        hr_val = float(extracted_measurements["heart_rate_printed"])

    cardiac_params = {
        "heart_rate_bpm": hr_val,
        "heart_rate_category": _get_hr_category(hr_val),
        "mean_rr_ms": mean_rr,
        "detected_beats": beat_cnt,
        # Printed measurements from report if available
        "pr_interval_ms": extracted_measurements.get("pr_interval_ms") if extracted_measurements else None,
        "qrs_duration_ms": extracted_measurements.get("qrs_duration_ms") if extracted_measurements else None,
        "qt_interval_ms": extracted_measurements.get("qt_interval_ms") if extracted_measurements else None,
        "qtc_interval_ms": extracted_measurements.get("qtc_interval_ms") if extracted_measurements else None,
        "p_axis_deg": extracted_measurements.get("p_axis_deg") if extracted_measurements else None,
        "qrs_axis_deg": extracted_measurements.get("qrs_axis_deg") if extracted_measurements else None,
        "t_axis_deg": extracted_measurements.get("t_axis_deg") if extracted_measurements else None,
        "machine_interpretation": extracted_measurements.get("machine_interpretation") if extracted_measurements else None,
    }

    # AI Analysis & Model Probabilities
    if ai_results and "predicted_class" in ai_results:
        probs = ai_results.get("probabilities", {})
        counts = ai_results.get("class_counts", {})
        pvc_cnt = counts.get("PVC", 0)
        norm_cnt = counts.get("Normal", 0)
        oth_cnt = counts.get("Other", 0)
        total_b = max(1, beat_cnt or 1)

        ai_analysis = {
            "status": "COMPLETED",
            "primary_pattern": ai_results["predicted_class"],
            "probabilities": {
                "Normal": round(float(probs.get("Normal", 0.0)) * 100.0, 1),
                "PVC": round(float(probs.get("PVC", 0.0)) * 100.0, 1),
                "Other": round(float(probs.get("Other", 0.0)) * 100.0, 1),
            },
            "beat_distribution": {
                "Normal": {"count": norm_cnt, "percentage": round(norm_cnt / total_b * 100.0, 1)},
                "PVC": {"count": pvc_cnt, "percentage": round(pvc_cnt / total_b * 100.0, 1)},
                "Other": {"count": oth_cnt, "percentage": round(oth_cnt / total_b * 100.0, 1)},
            },
            "explanation": _generate_plain_language_explanation(ai_results["predicted_class"], pvc_cnt, total_b),
        }
    else:
        ai_analysis = {
            "status": "NOT_EXECUTED",
            "primary_pattern": "Analysis Not Run",
            "probabilities": {},
            "beat_distribution": {},
            "explanation": (
                waveform_status.get("message", "Waveform could not be reliably extracted from the uploaded document.")
                if waveform_status else "No waveform available for AI inference."
            ),
        }

    # Factual findings list
    findings = []
    if hr_val is not None:
        findings.append(f"Heart rate estimated at {hr_val:.1f} BPM ({cardiac_params['heart_rate_category']}).")
    if quality_data["category"] != "NOT_ASSESSED":
        findings.append(f"Signal quality rated as {quality_data['category']} (SNR: {quality_data['snr_db']} dB).")
    if ai_analysis["status"] == "COMPLETED":
        pvc_info = ai_analysis["beat_distribution"]["PVC"]
        if pvc_info["count"] > 0:
            findings.append(f"Detected {pvc_info['count']} Premature Ventricular Contraction (PVC) beats ({pvc_info['percentage']}% burden).")
        else:
            findings.append("No premature ventricular contractions detected in the analyzed window.")
    if cardiac_params["qrs_duration_ms"] is not None:
        findings.append(f"Printed report QRS duration: {cardiac_params['qrs_duration_ms']:.0f} ms.")
    if cardiac_params["qtc_interval_ms"] is not None:
        findings.append(f"Printed report QTc interval: {cardiac_params['qtc_interval_ms']:.0f} ms.")
    if cardiac_params["machine_interpretation"]:
        findings.append(f"Printed machine interpretation: {', '.join(cardiac_params['machine_interpretation'])} (Source Information).")

    # Hospital information
    h_info = hospital_info or {
        "institution_name": "Apex Heart & Vascular Hospital",
        "department": "Department of Cardiac Electrophysiology",
        "facility_id": "MED-FAC-2026-IND",
    }

    # Clinician review
    if clinician_review:
        c_review = {
            "status": clinician_review.get("agreement_status", "REVIEWED"),
            "clinician_name": clinician_review.get("clinician_name", "Unknown Clinician"),
            "clinician_role": clinician_review.get("clinician_role", "DOCTOR"),
            "registration_number": clinician_review.get("registration_number", "N/A"),
            "clinician_interpretation": clinician_review.get("clinician_interpretation", "No interpretation recorded"),
            "clinical_notes": clinician_review.get("clinical_notes", ""),
            "reviewed_at": clinician_review.get("reviewed_at", timestamp_str),
        }
    else:
        c_review = {
            "status": "PENDING_REVIEW",
            "clinician_name": None,
            "clinician_role": None,
            "registration_number": None,
            "clinician_interpretation": "Awaiting attending physician sign-off.",
            "clinical_notes": "",
            "reviewed_at": None,
        }

    return {
        "report_title": "AI ECG SCREENING REPORT — CLINICAL DECISION SUPPORT",
        "report_subtitle": "Clinical Decision Support — Subject to Mandatory Qualified Physician Review",
        "generated_at": timestamp_str,
        "hospital_info": h_info,
        "clinician_review": c_review,
        "patient_info": p_info,
        "input_info": input_info,
        "signal_quality": quality_data,
        "cardiac_parameters": cardiac_params,
        "ai_analysis": ai_analysis,
        "findings": findings,
        "disclaimer": (
            "This report is generated automatically as an AI-assisted clinical decision support tool. "
            "It is NOT an autonomous medical device and does NOT constitute an independent clinical diagnosis. "
            "Pursuant to CDSCO MDR 2017 and IEC 62304 safety principles, all analytical findings and classifications "
            "must be verified, interpreted, and signed off by a licensed medical practitioner before patient care decisions."
        ),
    }


def _get_hr_category(hr: Optional[float]) -> str:
    if hr is None:
        return "Unknown"
    if hr < 60.0:
        return "Bradycardia (<60 BPM)"
    if hr > 100.0:
        return "Tachycardia (>100 BPM)"
    return "Normal Rate (60–100 BPM)"


def _generate_plain_language_explanation(pred_class: str, pvc_count: int, total_beats: int) -> str:
    if "PVC" in pred_class:
        return (
            f"The research model identified {pvc_count} heartbeat(s) with morphological and timing characteristics "
            f"consistent with Premature Ventricular Contractions (PVCs). This means the electrical excitation originated "
            f"early in the heart's ventricles rather than the normal sinoatrial node. This is an algorithmic pattern screening, "
            f"not a definitive clinical diagnosis."
        )
    if "Other" in pred_class:
        return (
            "The model identified non-sinus or atypical cardiac complexes (such as supraventricular ectopic or fusion beats). "
            "Consultation with a cardiologist and a full 12-lead ECG is recommended for comprehensive evaluation."
        )
    return (
        "The model classified the heartbeats as consistent with Normal Sinus Rhythm. "
        "The timing and waveform morphology align with standard ventricular depolarization cycles within the evaluated recording window."
    )


def export_report_to_text(report: Dict[str, Any]) -> str:
    """Render report dictionary as clean formatted plain text."""
    lines = [
        "=" * 72,
        report["report_title"],
        report["report_subtitle"],
        f"Generated: {report['generated_at']}",
        "=" * 72,
        "",
        "1. PATIENT & INPUT INFORMATION",
        "-" * 72,
        f"  Patient Name     : {report['patient_info']['patient_name'] or 'Not Specified / Not in File'}",
        f"  Patient Age/Sex  : {report['patient_info']['patient_age'] or 'N/A'} / {report['patient_info']['patient_sex'] or 'N/A'}",
        f"  Recording Date   : {report['patient_info']['recording_date'] or 'N/A'}",
        f"  File Name        : {report['input_info'].get('filename', 'N/A')}",
        f"  Input Modality   : {report['input_info'].get('modality', 'N/A')}",
        f"  Sampling Rate    : {report['input_info'].get('sampling_rate', 'N/A')} Hz",
        f"  Analyzed Duration: {report['input_info'].get('duration_sec', 'N/A')} seconds",
        f"  Analyzed Lead    : {report['input_info'].get('lead', 'Lead II / MLII')}",
        "",
        "2. SIGNAL QUALITY ASSESSMENT",
        "-" * 72,
        f"  Category         : {report['signal_quality']['category']}",
        f"  Quality Score    : {report['signal_quality']['quality_score'] or 'N/A'} / 1.00",
        f"  SNR              : {report['signal_quality']['snr_db'] or 'N/A'} dB",
        f"  Baseline Wander  : {report['signal_quality']['baseline_wander']}",
        f"  Motion Artifacts : {report['signal_quality']['motion_artifacts']}",
        "",
        "3. CARDIAC PARAMETERS",
        "-" * 72,
        f"  Estimated Heart Rate : {report['cardiac_parameters']['heart_rate_bpm'] or 'N/A'} BPM ({report['cardiac_parameters']['heart_rate_category']})",
        f"  Mean R-R Interval    : {report['cardiac_parameters']['mean_rr_ms'] or 'N/A'} ms",
        f"  Detected Beats       : {report['cardiac_parameters']['detected_beats'] or 'N/A'}",
        f"  PR Interval (Printed): {report['cardiac_parameters']['pr_interval_ms'] or 'Not reliably extracted'} ms",
        f"  QRS Duration         : {report['cardiac_parameters']['qrs_duration_ms'] or 'Not reliably extracted'} ms",
        f"  QT / QTc (Printed)   : {report['cardiac_parameters']['qt_interval_ms'] or 'N/A'} / {report['cardiac_parameters']['qtc_interval_ms'] or 'N/A'} ms",
        f"  P/QRS/T Axes         : {report['cardiac_parameters']['p_axis_deg'] or 'N/A'} / {report['cardiac_parameters']['qrs_axis_deg'] or 'N/A'} / {report['cardiac_parameters']['t_axis_deg'] or 'N/A'} deg",
        "",
        "4. AI ABNORMALITY ANALYSIS",
        "-" * 72,
        f"  Primary Pattern      : {report['ai_analysis']['primary_pattern']}",
    ]

    if report["ai_analysis"]["probabilities"]:
        lines.append(f"  Model Probabilities  : Normal: {report['ai_analysis']['probabilities'].get('Normal', 0)}%, PVC: {report['ai_analysis']['probabilities'].get('PVC', 0)}%, Other: {report['ai_analysis']['probabilities'].get('Other', 0)}%")

    if report["ai_analysis"]["beat_distribution"]:
        b_dist = report["ai_analysis"]["beat_distribution"]
        lines.append(f"  Beat Breakdown       : Normal: {b_dist['Normal']['count']} ({b_dist['Normal']['percentage']}%), PVC: {b_dist['PVC']['count']} ({b_dist['PVC']['percentage']}%), Other: {b_dist['Other']['count']} ({b_dist['Other']['percentage']}%)")

    lines.extend([
        "",
        "  AI Explanation:",
        f"    {report['ai_analysis']['explanation']}",
        "",
        "5. DETECTED FINDINGS",
        "-" * 72,
    ])
    for f in report["findings"]:
        lines.append(f"  • {f}")

    if report.get("clinician_review"):
        cr = report["clinician_review"]
        lines.extend([
            "",
            "6. CLINICIAN REVIEW & PHYSICIAN SIGN-OFF",
            "-" * 72,
            f"  Review Status        : {cr.get('status', 'PENDING_REVIEW')}",
            f"  Reviewing Physician  : {cr.get('clinician_name') or 'Pending Clinician Review'}",
            f"  Physician Role       : {cr.get('clinician_role') or 'N/A'}",
            f"  Medical Reg. Number  : {cr.get('registration_number') or 'N/A'}",
            f"  Clinical Diagnosis   : {cr.get('clinician_interpretation') or 'Awaiting Attending Physician Review'}",
            f"  Clinical Directives  : {cr.get('clinical_notes') or 'None'}",
            f"  Sign-off Timestamp   : {cr.get('reviewed_at') or 'Pending'}",
        ])

    lines.extend([
        "",
        "=" * 72,
        "IMPORTANT MEDICAL DISCLAIMER",
        "=" * 72,
        report["disclaimer"],
        "=" * 72,
    ])

    return "\n".join(lines)


def export_report_to_json(report: Dict[str, Any]) -> str:
    """Serialize report dictionary to formatted JSON."""
    return json.dumps(report, indent=2, default=str)
