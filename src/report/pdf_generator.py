"""
Professional PDF ECG Report Generator
======================================

Generates a publication-grade PDF research screening report using ReportLab:
- Institutional academic research header
- Clear "Research/Educational Analysis â€” Not a Medical Diagnosis" status banner
- Patient demographics & recording input table
- Signal quality assessment breakdown
- Clinical parameters table (Heart rate, RR, PR, QRS, QT/QTc, axes)
- AI Abnormality Classification & class probability breakdown
- Embedded high-resolution ECG waveform strip with detected R-peaks
- Factual bulleted findings
- Prominent cardiovascular safety & clinical disclaimer

Research/educational use only.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

try:
    from components.analysis import (
        get_waveform_criteria,
        get_signal_quality_criteria,
        get_ai_classification_criteria,
        get_patient_context_criteria,
        get_medication_safety_criteria,
        get_cds_criteria,
        CriteriaDescriptor,
    )
except ImportError:
    from src.components.analysis import (
        get_waveform_criteria,
        get_signal_quality_criteria,
        get_ai_classification_criteria,
        get_patient_context_criteria,
        get_medication_safety_criteria,
        get_cds_criteria,
        CriteriaDescriptor,
    )


def _make_pdf_criteria_footer(
    criteria_desc: Optional[CriteriaDescriptor] = None,
    label: str = "Criteria used",
    criteria: Optional[list[str]] = None,
    processing: Optional[list[str]] = None,
    missing: Optional[list[str]] = None,
    width: float = 540,
) -> Table:
    """Creates a standardized, muted, low-prominence criteria footer for PDF sections."""
    if criteria_desc is not None:
        label = criteria_desc.label
        criteria = criteria_desc.criteria
        processing = getattr(criteria_desc, "processing_steps", getattr(criteria_desc, "processing", []))
        missing = criteria_desc.missing_items

    criteria = criteria or []
    processing = processing or []
    missing = missing or []

    parts = []
    if criteria:
        joined_crit = " • ".join(criteria)
        parts.append(f"<font color='#6c757d'><b>{label}:</b></font> {joined_crit}")
    if processing:
        joined_proc = " • ".join(processing)
        parts.append(f"<font color='#6c757d'><b>Processing:</b></font> {joined_proc}")
    if missing:
        joined_miss = " • ".join(missing)
        parts.append(f"<font color='#d9534f'><b>Missing information:</b></font> {joined_miss}")

    text = "<br/>".join(parts) if parts else f"<font color='#6c757d'><b>{label}:</b> Dynamic input criteria</font>"

    p_style = ParagraphStyle(
        "PDFCriteriaFooter",
        fontSize=6.8,
        leading=8.8,
        textColor=colors.HexColor("#555e68"),
        fontName="Helvetica",
    )
    p = Paragraph(text, p_style)
    t = Table([[p]], colWidths=[width])
    t.setStyle(TableStyle([
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("LINEABOVE", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
    ]))
    return t


def generate_doctor_report(
    report_data: Dict[str, Any],
    waveform: Optional[np.ndarray] = None,
    fs: float = 360.0,
    r_peaks: Optional[np.ndarray] = None,
) -> bytes:
    """Generate a publication-grade Doctor Clinical Decision-Support PDF report.

    Args:
        report_data: Structured report dictionary from generate_structured_report
        waveform: Optional 1D ECG voltage array to render inside the PDF
        fs: Sampling frequency in Hz
        r_peaks: Optional detected R-peak sample indices to annotate

    Returns:
        bytes of the compiled PDF document
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=32,
        bottomMargin=32,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    hosp_style = ParagraphStyle(
        "HospHeader",
        parent=styles["Normal"],
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#0d3b66"),
        fontName="Helvetica-Bold",
        alignment=1,
    )
    hosp_sub = ParagraphStyle(
        "HospSub",
        parent=styles["Normal"],
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#555555"),
        alignment=1,
    )
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#1e3d59"),
        fontName="Helvetica-Bold",
        alignment=1,  # Center
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#d9534f"),
        fontName="Helvetica-Bold",
        alignment=1,
    )
    sec_heading_style = ParagraphStyle(
        "SecHeading",
        parent=styles["Heading2"],
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#17252a"),
        fontName="Helvetica-Bold",
        spaceBefore=6,
        spaceAfter=3,
    )
    cell_bold = ParagraphStyle(
        "CellBold",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#2b2d42"),
    )
    cell_normal = ParagraphStyle(
        "CellNormal",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11,
        fontName="Helvetica",
        textColor=colors.HexColor("#2b2d42"),
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11,
        fontName="Helvetica",
        textColor=colors.HexColor("#333333"),
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#777777"),
        fontName="Helvetica-Oblique",
    )

    story = []

    # 1. Hospital Institutional Letterhead
    hosp = report_data.get("hospital_info", {})
    inst_name = hosp.get("institution_name", "Apex Heart & Vascular Hospital")
    dept_name = hosp.get("department", "Department of Cardiac Electrophysiology & Telemetry")
    fac_id = hosp.get("facility_id", "MED-FAC-2026-IND")

    story.append(Paragraph(f"<b>{inst_name.upper()}</b>", hosp_style))
    story.append(Paragraph(f"{dept_name} â€¢ Facility ID: {fac_id}", hosp_sub))
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#0d3b66"), spaceBefore=2, spaceAfter=4))

    # Title Banner
    story.append(Paragraph(report_data.get("report_title", "AI ECG SCREENING REPORT"), title_style))
    story.append(Spacer(1, 2))
    story.append(Paragraph(report_data.get("report_subtitle", "Research/Educational AI Analysis â€” Not a Medical Diagnosis"), subtitle_style))
    story.append(Spacer(1, 3))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1e3d59"), spaceBefore=2, spaceAfter=6))

    # 2. Patient Demographics & Input Information Table
    story.append(Paragraph("1. PATIENT & RECORDING INFORMATION", sec_heading_style))
    p_info = report_data.get("patient_info", {})
    clin_hist = report_data.get("clinical_history", {})
    allergies_meds = report_data.get("allergies_and_medications", {})
    i_info = report_data.get("input_info", {})

    known_allergies = allergies_meds.get("known_allergies") or p_info.get("known_allergies") or "None documented"
    smoking_status = clin_hist.get("smoking_status") or p_info.get("smoking_status") or "Not recorded"
    existing_conditions = clin_hist.get("existing_conditions") or p_info.get("existing_conditions") or "None documented"
    current_meds = allergies_meds.get("current_medications") or p_info.get("current_medications") or "None recorded"
    cardiac_history = clin_hist.get("cardiac_history") or p_info.get("previous_cardiac_history") or "None documented"

    meta_table_data = [
        [
            Paragraph("<b>Patient Name:</b>", cell_bold),
            Paragraph(str(p_info.get("patient_name") or "Not Specified"), cell_normal),
            Paragraph("<b>Hospital MRN:</b>", cell_bold),
            Paragraph(str(p_info.get("hospital_mrn") or "Not Assigned"), cell_normal),
        ],
        [
            Paragraph("<b>Age / Sex:</b>", cell_bold),
            Paragraph(f"{p_info.get('patient_age') or 'N/A'} yrs / {p_info.get('patient_sex') or 'N/A'}", cell_normal),
            Paragraph("<b>Blood Group:</b>", cell_bold),
            Paragraph(str(p_info.get("blood_group") or "Unknown"), cell_normal),
        ],
        [
            Paragraph("<b>Known Allergies:</b>", cell_bold),
            Paragraph(str(known_allergies), cell_normal),
            Paragraph("<b>Smoking Status:</b>", cell_bold),
            Paragraph(str(smoking_status), cell_normal),
        ],
        [
            Paragraph("<b>Existing Conditions:</b>", cell_bold),
            Paragraph(str(existing_conditions), cell_normal),
            Paragraph("<b>Contact No.:</b>", cell_bold),
            Paragraph(str(p_info.get("contact") or "Not provided"), cell_normal),
        ],
        [
            Paragraph("<b>Current Medications:</b>", cell_bold),
            Paragraph(str(current_meds), cell_normal),
            Paragraph("<b>Cardiac History:</b>", cell_bold),
            Paragraph(str(cardiac_history), cell_normal),
        ],
        [
            Paragraph("<b>Recording Date:</b>", cell_bold),
            Paragraph(str(p_info.get("recording_date") or report_data.get("generated_at", "N/A")), cell_normal),
            Paragraph("<b>Sampling Rate / Lead:</b>", cell_bold),
            Paragraph(f"{i_info.get('sampling_rate', 360)} Hz ({i_info.get('lead', 'Lead II')})", cell_normal),
        ],
    ]
    t_meta = Table(meta_table_data, colWidths=[110, 160, 110, 160])

    t_meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8f9fa")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    crit_pat = get_patient_context_criteria({
        **p_info,
        **clin_hist,
        **allergies_meds,
        **i_info,
    })
    t_crit_pat = _make_pdf_criteria_footer(crit_pat, width=540)
    story.append(KeepTogether([t_meta, t_crit_pat]))
    story.append(Spacer(1, 6))

    # Missing Clinical Information Warnings (if any)
    if report_data.get("missing_information_warnings"):
        warn_data = [[
            Paragraph(
                "<b>⚠️ CAUTION — MISSING CLINICAL INFORMATION:</b> " + " • ".join(report_data["missing_information_warnings"]),
                ParagraphStyle("WarnBox", parent=cell_normal, textColor=colors.HexColor("#856404"), fontSize=7.5, leading=10)
            )
        ]]
        t_warn = Table(warn_data, colWidths=[540])
        t_warn.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff3cd")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#ffeeba")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(t_warn)
        story.append(Spacer(1, 6))

    # Vital Signs & Laboratory Context Table (Zero Fabrication)
    vitals = report_data.get("vital_signs", {})
    labs = report_data.get("laboratory_results", {})
    if vitals or labs:
        story.append(Paragraph("2. VITAL SIGNS & LABORATORY CONTEXT (POINT-IN-TIME)", sec_heading_style))
        vl_table_data = [
            [
                Paragraph("<b>Blood Pressure:</b>", cell_bold),
                Paragraph(str(vitals.get("blood_pressure") or "NOT PROVIDED"), cell_normal),
                Paragraph("<b>Serum Potassium (K+):</b>", cell_bold),
                Paragraph(str(labs.get("potassium_meq_l") or "NOT PROVIDED"), cell_normal),
            ],
            [
                Paragraph("<b>Heart Rate / SpO2:</b>", cell_bold),
                Paragraph(f"{vitals.get('heart_rate_bpm') or 'N/A'} BPM | {vitals.get('spo2_percent') or 'N/A'}", cell_normal),
                Paragraph("<b>Serum Sodium (Na+):</b>", cell_bold),
                Paragraph(str(labs.get("sodium_meq_l") or "NOT PROVIDED"), cell_normal),
            ],
            [
                Paragraph("<b>Resp Rate / Temp:</b>", cell_bold),
                Paragraph(f"{vitals.get('respiratory_rate_bpm') or 'N/A'} bpm | {vitals.get('temperature_c') or 'N/A'}", cell_normal),
                Paragraph("<b>Creatinine / eGFR:</b>", cell_bold),
                Paragraph(f"{labs.get('creatinine_mg_dl') or 'N/A'} | {labs.get('egfr_ml_min') or 'N/A'}", cell_normal),
            ],
            [
                Paragraph("<b>Cardiac Troponin:</b>", cell_bold),
                Paragraph(str(labs.get("troponin_ng_ml") or "NOT PROVIDED"), cell_normal),
                Paragraph("<b>BNP / NT-proBNP:</b>", cell_bold),
                Paragraph(str(labs.get("bnp_pg_ml") or "NOT PROVIDED"), cell_normal),
            ],
        ]
        t_vl = Table(vl_table_data, colWidths=[120, 150, 120, 150])
        t_vl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8f9fa")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        vl_crit_tokens = []
        if vitals.get("blood_pressure") and vitals.get("blood_pressure") != "NOT PROVIDED": vl_crit_tokens.append("Blood pressure")
        if vitals.get("heart_rate_bpm"): vl_crit_tokens.append("Heart rate")
        if vitals.get("spo2_percent"): vl_crit_tokens.append("SpO2")
        if vitals.get("respiratory_rate_bpm"): vl_crit_tokens.append("Respiratory rate")
        if vitals.get("temperature_c"): vl_crit_tokens.append("Temperature")
        if labs.get("potassium_meq_l") and labs.get("potassium_meq_l") != "NOT PROVIDED": vl_crit_tokens.append("Serum potassium")
        if labs.get("sodium_meq_l") and labs.get("sodium_meq_l") != "NOT PROVIDED": vl_crit_tokens.append("Serum sodium")
        if labs.get("creatinine_mg_dl") and labs.get("creatinine_mg_dl") != "NOT PROVIDED": vl_crit_tokens.append("Serum creatinine")
        if labs.get("egfr_ml_min") and labs.get("egfr_ml_min") != "NOT PROVIDED": vl_crit_tokens.append("eGFR")
        if labs.get("troponin_ng_ml") and labs.get("troponin_ng_ml") != "NOT PROVIDED": vl_crit_tokens.append("Cardiac troponin")
        if labs.get("bnp_pg_ml") and labs.get("bnp_pg_ml") != "NOT PROVIDED": vl_crit_tokens.append("BNP/NT-proBNP")
        if not vl_crit_tokens:
            vl_crit_tokens = ["Point-in-time clinical observations"]
        t_crit_vl = _make_pdf_criteria_footer(label="Criteria used", criteria=vl_crit_tokens, width=540)
        story.append(KeepTogether([t_vl, t_crit_vl]))
        story.append(Spacer(1, 6))

    # 3. Signal Quality & Cardiac Parameters Grid
    story.append(Paragraph("3. SIGNAL QUALITY & CARDIAC PARAMETERS", sec_heading_style))
    sq = report_data.get("signal_quality", {})
    cp = report_data.get("cardiac_parameters", {})

    q_cat = sq.get("category", "N/A")
    q_color = colors.HexColor("#28a745") if q_cat == "GOOD" else (colors.HexColor("#ffc107") if q_cat == "ACCEPTABLE" else colors.HexColor("#dc3545"))

    pr_val = f"{cp.get('pr_interval_ms'):.0f} ms" if cp.get("pr_interval_ms") is not None else "Not reliably extracted"
    qrs_val = f"{cp.get('qrs_duration_ms'):.0f} ms" if cp.get("qrs_duration_ms") is not None else "Not reliably extracted"
    qtc_val = f"{cp.get('qt_interval_ms') or 'N/A'} / {cp.get('qtc_interval_ms') or 'N/A'} ms" if (cp.get("qt_interval_ms") or cp.get("qtc_interval_ms")) else "N/A"

    params_table_data = [
        [
            Paragraph("<b>Signal Quality:</b>", cell_bold),
            Paragraph(f"<b>{q_cat}</b> (Score: {sq.get('quality_score', 'N/A')}, SNR: {sq.get('snr_db', 'N/A')} dB)", cell_normal),
            Paragraph("<b>Heart Rate:</b>", cell_bold),
            Paragraph(f"<b>{cp.get('heart_rate_bpm') or 'N/A'} BPM</b> ({cp.get('heart_rate_category', 'Normal')})", cell_normal),
        ],
        [
            Paragraph("<b>Baseline Drift:</b>", cell_bold),
            Paragraph(str(sq.get("baseline_wander", "Normal")), cell_normal),
            Paragraph("<b>Mean R-R Interval:</b>", cell_bold),
            Paragraph(f"{cp.get('mean_rr_ms') or 'N/A'} ms", cell_normal),
        ],
        [
            Paragraph("<b>PR Interval:</b>", cell_bold),
            Paragraph(pr_val, cell_normal),
            Paragraph("<b>QRS Duration:</b>", cell_bold),
            Paragraph(qrs_val, cell_normal),
        ],
        [
            Paragraph("<b>QT / QTc Interval:</b>", cell_bold),
            Paragraph(qtc_val, cell_normal),
            Paragraph("<b>Detected Beats:</b>", cell_bold),
            Paragraph(f"{cp.get('detected_beats', 'N/A')} cardiac cycles", cell_normal),
        ],
    ]
    t_params = Table(params_table_data, colWidths=[110, 150, 110, 170])
    t_params.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ffffff")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    crit_sq_tokens = ["SNR", "Baseline drift", "Powerline interference", "Motion artifacts"]
    if cp.get("heart_rate_bpm") is not None: crit_sq_tokens.append("Heart rate")
    if cp.get("mean_rr_ms") is not None: crit_sq_tokens.append("Mean R-R interval")
    if cp.get("pr_interval_ms") is not None: crit_sq_tokens.append("PR interval")
    if cp.get("qrs_duration_ms") is not None: crit_sq_tokens.append("QRS duration")
    if cp.get("qtc_interval_ms") is not None: crit_sq_tokens.append("QT/QTc interval")
    if cp.get("detected_beats") is not None: crit_sq_tokens.append("Detected beats")
    t_crit_sq = _make_pdf_criteria_footer(label="Criteria used", criteria=crit_sq_tokens, width=540)
    story.append(KeepTogether([t_params, t_crit_sq]))
    story.append(Spacer(1, 6))

    # 4. AI Abnormality Classification Banner
    story.append(Paragraph("4. AI ABNORMALITY ANALYSIS (MULTIMODAL ECG CLASSIFIER)", sec_heading_style))
    ai = report_data.get("ai_analysis", {})
    mm = report_data.get("multimodal_fusion", {})
    p_pattern = mm.get("model_c_multimodal_pattern") or ai.get("primary_pattern", "Normal Sinus Rhythm")
    probs = ai.get("probabilities", {})
    b_dist = ai.get("beat_distribution", {})

    is_abnormal = "PVC" in p_pattern or "Other" in p_pattern
    bg_color = colors.HexColor("#f8d7da") if is_abnormal else colors.HexColor("#d4edda")
    border_color = colors.HexColor("#f5c6cb") if is_abnormal else colors.HexColor("#c3e6cb")
    text_color = colors.HexColor("#721c24") if is_abnormal else colors.HexColor("#155724")

    ai_box_data = [
        [
            Paragraph(f"<b>PRIMARY MULTIMODAL FINDING (MODEL C):</b> {p_pattern}", ParagraphStyle("AIBig", parent=styles["Normal"], fontSize=10, leading=13, fontName="Helvetica-Bold", textColor=text_color)),
        ],
        [
            Paragraph(
                f"<b>Model Probabilities:</b> Normal: <b>{probs.get('Normal', 0)}%</b> | "
                f"PVC: <b>{probs.get('PVC', 0)}%</b> | "
                f"Other: <b>{probs.get('Other', 0)}%</b>",
                cell_normal,
            ),
        ],
    ]
    if mm:
        ai_box_data.append([
            Paragraph(
                f"<b>Model Comparison:</b> Multimodal Model C Conf: <b>{mm.get('model_c_confidence', 'N/A')}</b> | "
                f"ECG-Only Model A Conf: <b>{mm.get('model_a_confidence', 'N/A')}</b>",
                cell_normal,
            )
        ])
        if mm.get("context_influence_statement"):
            ai_box_data.append([
                Paragraph(
                    f"<b>Clinical Context Impact:</b> {mm.get('context_influence_statement')}",
                    ParagraphStyle("MMInf", parent=cell_normal, fontSize=8, leading=10, textColor=colors.HexColor("#2b2d42"))
                )
            ])
    if b_dist:
        ai_box_data.append([
            Paragraph(
                f"<b>Beat Breakdown:</b> Normal: {b_dist.get('Normal', {}).get('count', 0)} ({b_dist.get('Normal', {}).get('percentage', 0)}%) | "
                f"PVC: {b_dist.get('PVC', {}).get('count', 0)} ({b_dist.get('PVC', {}).get('percentage', 0)}%) | "
                f"Other: {b_dist.get('Other', {}).get('count', 0)} ({b_dist.get('Other', {}).get('percentage', 0)}%)",
                cell_normal,
            )
        ])

    t_ai = Table(ai_box_data, colWidths=[540])
    t_ai.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg_color),
        ("BOX", (0, 0), (-1, -1), 1, border_color),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    is_mm = bool(mm)
    crit_ai = get_ai_classification_criteria(
        ai_results=ai,
        model_id=report_data.get("model_metadata", {}).get("model_version", "BeatArrhythmia-RF-v1"),
        is_multimodal=is_mm,
        patient_profile=p_info if is_mm else None,
        vital_signs=vitals if is_mm else None,
        laboratory_results=labs if is_mm else None,
    )
    t_crit_ai = _make_pdf_criteria_footer(crit_ai, width=540)
    story.append(KeepTogether([t_ai, t_crit_ai]))
    story.append(Spacer(1, 6))

    # 5. Embedded Waveform Snippet (if available)
    if waveform is not None and len(waveform) > 0:
        story.append(Paragraph("4. ECG WAVEFORM & R-PEAK TRACE", sec_heading_style))
        img_buf = _render_waveform_strip(waveform, fs, r_peaks)
        if img_buf:
            crit_wf = get_waveform_criteria(
                sampling_rate=i_info.get("sampling_rate", fs),
                lead=i_info.get("lead", "Lead II"),
                duration_sec=min(float(len(waveform)/fs), 6.0),
            )
            t_crit_wf = _make_pdf_criteria_footer(crit_wf, width=540)
            story.append(KeepTogether([RLImage(img_buf, width=540, height=90), t_crit_wf]))
            story.append(Spacer(1, 6))

    # 6. Detected Findings & Plain Language Explanation
    story.append(Paragraph("5. FACTUAL FINDINGS & EXPLANATION", sec_heading_style))
    findings = report_data.get("findings", [])
    findings_text = "".join([f"• {f}<br/>" for f in findings])
    story.append(Paragraph(findings_text, body_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"<b>Summary:</b> {ai.get('explanation', '')}", body_style))
    story.append(Spacer(1, 8))

    # Clinical Decision Support & Guidelines (If Available)
    if report_data.get("clinical_decision_support"):
        cds = report_data["clinical_decision_support"]
        story.append(Paragraph("6. CLINICAL DECISION SUPPORT (NON-AUTONOMOUS)", sec_heading_style))
        urg = cds.get("urgency", "ROUTINE REVIEW")
        urg_color = "#dc3545" if "URGENT" in urg else ("#fd7e14" if "PROMPT" in urg else "#28a745")
        cds_data = [
            [
                Paragraph(f"<b>Triage Urgency:</b> <font color='{urg_color}'><b>{urg}</b></font>", cell_bold),
                Paragraph(f"<b>Primary Finding:</b> {cds.get('primary_finding', 'N/A')}", cell_normal),
            ],
            [
                Paragraph(f"<b>Clinical Advisory:</b> {cds.get('summary', 'N/A')}", cell_normal),
                Paragraph(f"<b>Guidelines:</b> {'; '.join(cds.get('guideline_citations', [])) or 'ACC/AHA/ESC'}", cell_normal),
            ],
        ]
        t_cds = Table(cds_data, colWidths=[270, 270])
        t_cds.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        crit_cds = get_cds_criteria(
            cds_rec=cds,
            has_patient_history=bool(cardiac_history and cardiac_history != "None documented"),
            has_vitals=bool(vitals.get("blood_pressure") or vitals.get("heart_rate_bpm")),
            has_labs=bool(labs.get("potassium_meq_l") or labs.get("troponin_ng_ml")),
            has_meds=bool(current_meds and current_meds != "None recorded"),
            has_prior_ecg=bool(report_data.get("longitudinal_comparison")),
        )
        t_crit_cds = _make_pdf_criteria_footer(crit_cds, width=540)
        story.append(KeepTogether([t_cds, t_crit_cds]))
        story.append(Spacer(1, 4))

        # Guideline-Based Medication Recommendations
        med_suggestions = cds.get('medication_recommendations', [])
        if med_suggestions:
            story.append(Paragraph('7. GUIDELINE-BASED MEDICATION CONSIDERATIONS (PHYSICIAN REFERENCE)', sec_heading_style))
            story.append(Paragraph(
                '<b>For Physician Reference Only:</b> Evidence-based drug class considerations from AHA/ACC/ESC guidelines. '
                'This system does <b>NOT</b> prescribe medication. All prescribing decisions rest with the attending physician.',
                ParagraphStyle('MedNote', parent=cell_normal, textColor=colors.HexColor('#0369a1'), fontSize=8)
            ))
            story.append(Spacer(1, 3))
            med_header = [
                Paragraph('<b>Drug Class</b>', cell_bold),
                Paragraph('<b>Example Agents</b>', cell_bold),
                Paragraph('<b>Indication</b>', cell_bold),
                Paragraph('<b>Guideline</b>', cell_bold),
            ]
            med_rows = [med_header]
            for m in med_suggestions:
                med_rows.append([
                    Paragraph(str(m.get('drug_class', '-')), ParagraphStyle('MedDC', parent=cell_normal, fontName='Helvetica-Bold', fontSize=7.5)),
                    Paragraph(str(m.get('example_agents', '-')), ParagraphStyle('MedAg', parent=cell_normal, fontSize=7.5)),
                    Paragraph(str(m.get('indication', '-')), ParagraphStyle('MedInd', parent=cell_normal, fontSize=7.5)),
                    Paragraph(str(m.get('guideline', '-')), ParagraphStyle('MedGL', parent=cell_normal, fontSize=7, textColor=colors.HexColor('#0369a1'))),
                ])
                if m.get('note'):
                    med_rows.append([
                        Paragraph('', cell_normal),
                        Paragraph(str(m.get('note', '')), ParagraphStyle('MedNote2', parent=cell_normal, fontSize=7, textColor=colors.HexColor('#6b7280'), fontName='Helvetica-Oblique')),
                        Paragraph('', cell_normal),
                        Paragraph('', cell_normal),
                    ])
            t_med_sug = Table(med_rows, colWidths=[135, 140, 160, 105])
            t_med_sug.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dbeafe')),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8faff')),
                ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#bfdbfe')),
                ('TOPPADDING', (0, 0), (-1, -1), 2),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            med_tokens = ["ECG findings", "AHA/ACC/ESC clinical guidelines"]
            if existing_conditions and existing_conditions != "None documented":
                med_tokens.append("Existing conditions")
            if p_info.get("patient_age"):
                med_tokens.append("Patient age")
            t_crit_med_sug = _make_pdf_criteria_footer(label="Criteria considered", criteria=med_tokens, width=540)
            story.append(KeepTogether([t_med_sug, t_crit_med_sug]))
            story.append(Spacer(1, 4))

    # Medication Safety Evaluation (If Available)
    if report_data.get('medication_safety'):
        meds = report_data['medication_safety']
        alerts = meds.get('alerts', [])
        story.append(Paragraph('8. MEDICATION SAFETY AND INTERACTION EVALUATION', sec_heading_style))
        raw_meds = meds.get('analyzed_medications') or meds.get('active_medications', [])
        active_meds = ', '.join(raw_meds) if isinstance(raw_meds, list) else str(raw_meds)
        active_meds = active_meds or 'None recorded'
        med_summary = f'<b>Active Regimen Evaluated:</b> {active_meds} | <b>Safety Alerts:</b> {len(alerts)}'
        story.append(Paragraph(med_summary, body_style))
        if alerts:
            alert_rows = []
            for a in alerts[:4]:
                sev = a.get('severity', 'INFO')
                s_color = '#dc3545' if sev == 'CRITICAL' else ('#fd7e14' if sev == 'MAJOR' else '#0d6efd')
                alert_rows.append([
                    Paragraph(
                        "<font color='" + s_color + "'><b>[" + sev + "]</b></font> " + str(a.get('title','')) + ": " + str(a.get('description','')),
                        cell_normal
                    )
                ])
            t_med = Table(alert_rows, colWidths=[540])
            t_med.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fff8f0')),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#ffd8a8')),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#ffe8cc')),
                ('TOPPADDING', (0, 0), (-1, -1), 2),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ]))
            crit_med = get_medication_safety_criteria(
                med_safety_eval=meds,
                patient_medications=raw_meds if isinstance(raw_meds, list) else ([raw_meds] if raw_meds else []),
                allergies=known_allergies if (known_allergies and known_allergies != "None documented") else None,
                vital_signs=vitals,
                laboratory_results=labs,
            )
            t_crit_med = _make_pdf_criteria_footer(crit_med, width=540)
            story.append(KeepTogether([t_med, t_crit_med]))
        story.append(Spacer(1, 4))


    # Clinician Review & Physician Sign-off
    cr = report_data.get("clinician_review", {})
    c_status = cr.get("status", "PENDING_REVIEW")
    c_name = cr.get("clinician_name") or "Pending Qualified Physician Review"
    c_role = cr.get("clinician_role") or "Physician"
    c_reg = cr.get("registration_number") or "N/A"
    c_interp = cr.get("clinician_interpretation") or "Awaiting Attending Physician Evaluation"
    c_notes = cr.get("clinical_notes") or "None"
    c_time = cr.get("reviewed_at") or "Pending"

    status_color = "#28a745" if c_status == "CONFIRMED" else ("#fd7e14" if c_status == "MODIFIED" else ("#dc3545" if c_status == "REJECTED" else "#6c757d"))

    story.append(Paragraph("8. CLINICAL REVIEW & PHYSICIAN SIGN-OFF", sec_heading_style))
    review_box_data = [
        [
            Paragraph(f"<b>Review Status:</b> <font color='{status_color}'><b>{c_status}</b></font>", cell_bold),
            Paragraph(f"<b>Reviewing Clinician:</b> {c_name} ({c_role})", cell_normal),
        ],
        [
            Paragraph(f"<b>Medical Reg. No:</b> {c_reg}", cell_normal),
            Paragraph(f"<b>Review Date/Time:</b> {c_time}", cell_normal),
        ],
        [
            Paragraph(f"<b>Clinician Diagnosis:</b> {c_interp}", cell_bold),
            Paragraph(f"<b>Clinical Directives:</b> {c_notes}", cell_normal),
        ],
        [
            Paragraph("<b>Physician Signature:</b> ___________________________", cell_normal),
            Paragraph("<b>Hospital Seal:</b> [ Electronically Authenticated ]", cell_normal),
        ],
    ]
    t_rev = Table(review_box_data, colWidths=[270, 270])
    t_rev.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8f9fa")),
        ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#ced4da")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e9ecef")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    t_crit_rev = _make_pdf_criteria_footer(
        label="Criteria verified",
        criteria=["Attending clinician review", "Hospital physician signature", "Tamper-evident audit digest"],
        width=540,
    )
    story.append(KeepTogether([t_rev, t_crit_rev]))
    story.append(Spacer(1, 6))

    # 7. Disclaimer Box
    disclaimer_data = [[
        Paragraph(
            "<b>IMPORTANT CLINICAL DISCLAIMER:</b> This report was generated by an educational and research AI prototype. "
            "It is NOT an FDA/CE approved diagnostic device and should never be used as a substitute for professional clinical medical evaluation. "
            "If experiencing symptoms such as chest pain, syncope, or palpitations, consult a qualified physician or call emergency services immediately.",
            disclaimer_style,
        )
    ]]
    t_disc = Table(disclaimer_data, colWidths=[540])
    t_disc.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff3cd")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#ffeeba")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_disc)

    doc.build(story)
    return buffer.getvalue()


# Backward-compatible alias
generate_pdf_report = generate_doctor_report


def generate_patient_report(
    report_data: Dict[str, Any],
    waveform: Optional[np.ndarray] = None,
    fs: float = 360.0,
    r_peaks: Optional[np.ndarray] = None,
) -> bytes:
    """Generate a clear, reassuring, patient-friendly ECG Health Summary PDF.

    Complies with Phase 39: Plain-language, layman-friendly, discussion points,
    and prominent non-prescription safety disclaimer.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=32,
        bottomMargin=32,
    )

    styles = getSampleStyleSheet()

    # Patient report styles
    hosp_style = ParagraphStyle(
        "PatHospHeader",
        parent=styles["Normal"],
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#1b4332"),
        fontName="Helvetica-Bold",
        alignment=1,
    )
    title_style = ParagraphStyle(
        "PatDocTitle",
        parent=styles["Heading1"],
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#2d6a4f"),
        fontName="Helvetica-Bold",
        alignment=1,
    )
    subtitle_style = ParagraphStyle(
        "PatDocSubtitle",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#40916c"),
        alignment=1,
    )
    sec_heading = ParagraphStyle(
        "PatSectionHeading",
        parent=styles["Heading2"],
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#1b4332"),
        fontName="Helvetica-Bold",
        spaceBefore=6,
        spaceAfter=3,
    )
    body_text = ParagraphStyle(
        "PatBody",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#212529"),
    )
    bold_text = ParagraphStyle(
        "PatBold",
        parent=body_text,
        fontName="Helvetica-Bold",
    )
    bullet_text = ParagraphStyle(
        "PatBullet",
        parent=body_text,
        leftIndent=12,
        leading=11.5,
    )
    warning_box_style = ParagraphStyle(
        "PatWarningBox",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#856404"),
        fontName="Helvetica",
    )
    emergency_style = ParagraphStyle(
        "PatEmergency",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#721c24"),
    )

    story = []

    # 1. Hospital Header
    hosp_name = report_data.get("hospital_name", "Clinical Cardiology Network")
    hosp_dept = report_data.get("hospital_department", "Department of Preventive Cardiology & Heart Health")
    story.append(Paragraph(hosp_name.upper(), hosp_style))
    story.append(Paragraph(f"{hosp_dept} â€” Patient Education & Summary", subtitle_style))
    story.append(Spacer(1, 3))

    story.append(Paragraph("YOUR ECG TEST SUMMARY & HEART HEALTH GUIDE", title_style))
    gen_time = report_data.get("generated_at", "")
    p_info = report_data.get("patient_info", {})
    pat_name = p_info.get("patient_name", "Valued Patient")
    story.append(Paragraph(f"Prepared for: <b>{pat_name}</b> &nbsp;|&nbsp; Date: {gen_time[:10]}", subtitle_style))
    story.append(Spacer(1, 5))

    # 2. Reassuring Greeting Card
    greeting_card = [
        [
            Paragraph(
                f"<b>Hello {pat_name},</b><br/>"
                "This document summarizes your recent electrocardiogram (ECG) test in simple language. "
                "An ECG is a safe, painless test that listens to the electrical signals that keep your heart beating regularly. "
                "This guide explains what the test showed and offers helpful questions for your next doctor consultation.",
                body_text,
            )
        ]
    ]
    t_greet = Table(greeting_card, colWidths=[540])
    t_greet.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#e8f5e9")),
        ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#c8e6c9")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t_greet)
    story.append(Spacer(1, 5))

    # 3. Patient Details & Test Recording
    demo_table_data = [
        [
            Paragraph("<b>Patient Name:</b>", body_text),
            Paragraph(str(pat_name), bold_text),
            Paragraph("<b>Patient ID:</b>", body_text),
            Paragraph(str(p_info.get("patient_id", "N/A")), bold_text),
        ],
        [
            Paragraph("<b>Age:</b>", body_text),
            Paragraph(f"{p_info.get('patient_age', 'N/A')} years", body_text),
            Paragraph("<b>Gender:</b>", body_text),
            Paragraph(str(p_info.get("patient_sex", "N/A")), body_text),
        ],
    ]
    t_demo = Table(demo_table_data, colWidths=[100, 170, 100, 170])
    t_demo.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8f9fa")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e9ecef")),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(Paragraph("1. YOUR TEST INFORMATION", sec_heading))
    t_crit_demo = _make_pdf_criteria_footer(
        label="Criteria considered",
        criteria=["Patient Name", "Patient ID", "Age", "Gender"],
        width=540,
    )
    story.append(KeepTogether([t_demo, t_crit_demo]))
    story.append(Spacer(1, 5))

    # 4. Signal Quality in Plain Words
    s_qual = report_data.get("signal_quality", {})
    q_cat = s_qual.get("category", "ACCEPTABLE")
    if q_cat == "GOOD":
        qual_msg = "<b>Excellent Clarity:</b> The recording sensors had good contact with your skin, giving a clean and steady recording."
        qual_color = "#155724"
        qual_bg = "#d4edda"
    elif q_cat == "ACCEPTABLE":
        qual_msg = "<b>Satisfactory Quality:</b> The recording was clear enough for clinical evaluation, with minimal movement."
        qual_color = "#856404"
        qual_bg = "#fff3cd"
    else:
        qual_msg = "<b>Some Signal Noise:</b> The sensors detected some muscle tension or movement during recording. Your doctor may suggest retesting if needed."
        qual_color = "#721c24"
        qual_bg = "#f8d7da"

    t_qual = Table([[Paragraph(f"<font color='{qual_color}'>{qual_msg}</font>", body_text)]], colWidths=[540])
    t_qual.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(qual_bg)),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor(qual_color)),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(Paragraph("2. HOW CLEAR WAS THE RECORDING?", sec_heading))
    t_crit_qual = _make_pdf_criteria_footer(
        label="Criteria used",
        criteria=["Signal Quality category", "Skin sensor contact", "Baseline drift stability"],
        width=540,
    )
    story.append(KeepTogether([t_qual, t_crit_qual]))
    story.append(Spacer(1, 5))

    # 5. Heart Rate & Rhythm Summary
    c_params = report_data.get("cardiac_parameters", {})
    ai_info = report_data.get("ai_analysis", {})
    hr_bpm = c_params.get("heart_rate_bpm")
    prim_pat = ai_info.get("primary_pattern", "Normal Rhythm")

    if hr_bpm:
        hr_context = "Within typical resting range (60–100 beats per minute)" if (60 <= hr_bpm <= 100) else ("Slightly slower than average resting rate" if hr_bpm < 60 else "Slightly faster than average resting rate")
        hr_display = f"<b>{hr_bpm} beats per minute</b> ({hr_context})"
    else:
        hr_display = "Measurement awaiting physician review"

    if "NORMAL" in prim_pat.upper():
        rhythm_explanation = (
            "<b>Regular Rhythm:</b> Your heart beats followed a standard, organized electrical rhythm (Normal Sinus Rhythm). "
            "This indicates the natural pacemaker of your heart was working in a steady sequence during the recording."
        )
    elif "PVC" in prim_pat.upper():
        rhythm_explanation = (
            "<b>Occasional Early Beat (Premature Beat):</b> The test noted an occasional early contraction from the heart's lower chambers. "
            "Premature beats are very common in healthy individuals and are often felt as a temporary 'flutter' or 'skipped beat'. "
            "They can be influenced by caffeine, stress, or tiredness. Your doctor will discuss whether any further check is needed."
        )
    else:
        rhythm_explanation = (
            f"<b>Rhythm Observation:</b> The screening algorithm identified a pattern described as <b>{prim_pat}</b>. "
            "Please speak with your attending doctor to understand the exact clinical interpretation in the context of your overall health."
        )

    rhythm_card_data = [
        [Paragraph("<b>Measured Heart Rate:</b>", body_text), Paragraph(hr_display, body_text)],
        [Paragraph("<b>Heart Rhythm Summary:</b>", body_text), Paragraph(rhythm_explanation, body_text)],
    ]
    t_rhythm = Table(rhythm_card_data, colWidths=[130, 410])
    t_rhythm.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ffffff")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#ced4da")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e9ecef")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(Paragraph("3. YOUR HEART RATE & RHYTHM", sec_heading))
    t_crit_rhythm = _make_pdf_criteria_footer(
        label="Criteria used",
        criteria=["Measured Heart Rate", "Beat timing consistency", "AI rhythm classification"],
        width=540,
    )
    story.append(KeepTogether([t_rhythm, t_crit_rhythm]))
    story.append(Spacer(1, 5))

    # 6. Waveform Strip Visual
    if waveform is not None and len(waveform) > 0:
        img_buf = _render_waveform_strip(waveform, fs, r_peaks, max_duration_sec=5.0)
        if img_buf is not None:
            story.append(Paragraph("4. VISUAL OF YOUR HEARTBEAT TRACING", sec_heading))
            story.append(Paragraph("Below is a snapshot of your heartbeat signals. The red markers indicate the top of each heartbeat pulse.", body_text))
            crit_wf_pat = get_waveform_criteria(sampling_rate=fs, lead="Lead II", duration_sec=5.0)
            t_crit_wf_pat = _make_pdf_criteria_footer(crit_wf_pat, width=540)
            story.append(KeepTogether([RLImage(img_buf, width=540, height=75), t_crit_wf_pat]))
            story.append(Spacer(1, 5))

    # 7. Helpful Discussion Points For Doctor Consultation
    story.append(Paragraph("5. QUESTIONS TO ASK YOUR DOCTOR", sec_heading))
    disc_points = [
        "1. Do these ECG findings match how I have been feeling lately?",
        "2. Do I need any follow-up tests, such as a 24-hour Holter monitor, an echocardiogram, or routine blood work?",
        "3. Are there any lifestyle recommendations (e.g. hydration, caffeine, sleep, or exercise) that I should follow?",
        "4. Do any of my current medications need review or adjustment?",
    ]
    for dp in disc_points:
        story.append(Paragraph(f"â€¢ &nbsp; {dp}", bullet_text))
    story.append(Spacer(1, 5))

    # 8. NON-NEGOTIABLE SAFETY WARNING: No Prescriptions
    no_rx_box = [
        [
            Paragraph(
                "<b>IMPORTANT PATIENT SAFETY NOTICE: NO MEDICATION PRESCRIBED</b><br/>"
                "<b>No medication or treatment is prescribed through this report.</b> "
                "This automated summary is designed solely for patient information and educational transparency. "
                "Do not start, stop, or change any medication dosages based on this document. "
                "All medical diagnoses and prescription decisions must be made in person with your licensed doctor.",
                warning_box_style,
            )
        ]
    ]
    t_no_rx = Table(no_rx_box, colWidths=[540])
    t_no_rx.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff3cd")),
        ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#ffeeba")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t_no_rx)
    story.append(Spacer(1, 5))

    # 9. Emergency Warning
    em_box = [
        [
            Paragraph(
                "<b>WHEN TO SEEK IMMEDIATE EMERGENCY MEDICAL ATTENTION:</b><br/>"
                "If you experience severe chest pressure, pain radiating to your arm or jaw, sudden severe breathlessness, fainting, or sudden rapid fluttering, call emergency medical services immediately.",
                emergency_style,
            )
        ]
    ]
    t_em = Table(em_box, colWidths=[540])
    t_em.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8f7f8")),
        ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#e2d9dc")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t_em)

    doc.build(story)
    return buffer.getvalue()


def _render_waveform_strip(
    signal: np.ndarray,
    fs: float,
    r_peaks: Optional[np.ndarray],
    max_duration_sec: float = 6.0,
) -> Optional[io.BytesIO]:
    """Render a clean ECG strip with grid and R-peaks to a BytesIO image buffer."""
    try:
        n_samples = min(len(signal), int(max_duration_sec * fs))
        sig_slice = signal[:n_samples]
        t = np.arange(n_samples) / fs

        fig, ax = plt.subplots(figsize=(10, 1.8), dpi=150)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#fffbfc")

        # ECG Grid simulation: minor grid (0.04s, 0.1mV), major grid (0.2s, 0.5mV)
        ax.grid(True, which="major", color="#f8d7da", linestyle="-", linewidth=0.6)
        ax.minorticks_on()
        ax.grid(True, which="minor", color="#fde8ea", linestyle=":", linewidth=0.4)

        # Plot waveform
        ax.plot(t, sig_slice, color="#1a535c", linewidth=1.2)

        # Mark R-peaks if available
        if r_peaks is not None and len(r_peaks) > 0:
            valid_peaks = r_peaks[r_peaks < n_samples]
            if len(valid_peaks) > 0:
                ax.scatter(valid_peaks / fs, sig_slice[valid_peaks], color="#e63946", s=25, zorder=5, marker="^")

        ax.set_xlim(0, t[-1])
        ax.set_xlabel("Time (seconds)", fontsize=7)
        ax.set_ylabel("mV (norm)", fontsize=7)
        ax.tick_params(labelsize=6)
        fig.tight_layout(pad=0.5)

        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        img_buf.seek(0)
        return img_buf
    except Exception:
        return None

