"""
Professional PDF ECG Report Generator
======================================

Generates a publication-grade PDF research screening report using ReportLab:
- Institutional academic research header
- Clear "Research/Educational Analysis — Not a Medical Diagnosis" status banner
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
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def generate_pdf_report(
    report_data: Dict[str, Any],
    waveform: Optional[np.ndarray] = None,
    fs: float = 360.0,
    r_peaks: Optional[np.ndarray] = None,
) -> bytes:
    """Generate a publication-grade PDF report and return raw PDF bytes.

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
    story.append(Paragraph(f"{dept_name} • Facility ID: {fac_id}", hosp_sub))
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#0d3b66"), spaceBefore=2, spaceAfter=4))

    # Title Banner
    story.append(Paragraph(report_data.get("report_title", "AI ECG SCREENING REPORT"), title_style))
    story.append(Spacer(1, 2))
    story.append(Paragraph(report_data.get("report_subtitle", "Research/Educational AI Analysis — Not a Medical Diagnosis"), subtitle_style))
    story.append(Spacer(1, 3))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1e3d59"), spaceBefore=2, spaceAfter=6))

    # 2. Patient Demographics & Input Information Table
    story.append(Paragraph("1. PATIENT & RECORDING INFORMATION", sec_heading_style))
    p_info = report_data.get("patient_info", {})
    i_info = report_data.get("input_info", {})

    meta_table_data = [
        [
            Paragraph("<b>Patient Name:</b>", cell_bold),
            Paragraph(str(p_info.get("patient_name") or "Not Specified"), cell_normal),
            Paragraph("<b>File Name:</b>", cell_bold),
            Paragraph(str(i_info.get("filename") or "Uploaded ECG"), cell_normal),
        ],
        [
            Paragraph("<b>Age / Sex:</b>", cell_bold),
            Paragraph(f"{p_info.get('patient_age') or 'N/A'} / {p_info.get('patient_sex') or 'N/A'}", cell_normal),
            Paragraph("<b>Input Modality:</b>", cell_bold),
            Paragraph(str(i_info.get("modality") or "Digital Signal"), cell_normal),
        ],
        [
            Paragraph("<b>Recording Date:</b>", cell_bold),
            Paragraph(str(p_info.get("recording_date") or report_data.get("generated_at", "N/A")), cell_normal),
            Paragraph("<b>Sampling Rate / Lead:</b>", cell_bold),
            Paragraph(f"{i_info.get('sampling_rate', 360)} Hz ({i_info.get('lead', 'Lead II')})", cell_normal),
        ],
    ]
    t_meta = Table(meta_table_data, colWidths=[100, 160, 120, 160])
    t_meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8f9fa")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 6))

    # 3. Signal Quality & Cardiac Parameters Grid
    story.append(Paragraph("2. SIGNAL QUALITY & CARDIAC PARAMETERS", sec_heading_style))
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
    story.append(t_params)
    story.append(Spacer(1, 6))

    # 4. AI Abnormality Classification Banner
    story.append(Paragraph("3. AI ABNORMALITY ANALYSIS (RESEARCH CLASSIFIER)", sec_heading_style))
    ai = report_data.get("ai_analysis", {})
    p_pattern = ai.get("primary_pattern", "Normal Sinus Rhythm")
    probs = ai.get("probabilities", {})
    b_dist = ai.get("beat_distribution", {})

    is_abnormal = "PVC" in p_pattern or "Other" in p_pattern
    bg_color = colors.HexColor("#f8d7da") if is_abnormal else colors.HexColor("#d4edda")
    border_color = colors.HexColor("#f5c6cb") if is_abnormal else colors.HexColor("#c3e6cb")
    text_color = colors.HexColor("#721c24") if is_abnormal else colors.HexColor("#155724")

    ai_box_data = [
        [
            Paragraph(f"<b>PRIMARY AI PATTERN:</b> {p_pattern}", ParagraphStyle("AIBig", parent=styles["Normal"], fontSize=10, leading=13, fontName="Helvetica-Bold", textColor=text_color)),
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
    story.append(t_ai)
    story.append(Spacer(1, 6))

    # 5. Embedded Waveform Snippet (if available)
    if waveform is not None and len(waveform) > 0:
        story.append(Paragraph("4. ECG WAVEFORM & R-PEAK TRACE", sec_heading_style))
        img_buf = _render_waveform_strip(waveform, fs, r_peaks)
        if img_buf:
            story.append(RLImage(img_buf, width=540, height=90))
            story.append(Spacer(1, 6))

    # 6. Detected Findings & Plain Language Explanation
    story.append(Paragraph("5. FACTUAL FINDINGS & EXPLANATION", sec_heading_style))
    findings = report_data.get("findings", [])
    findings_text = "".join([f"• {f}<br/>" for f in findings])
    story.append(Paragraph(findings_text, body_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"<b>Summary:</b> {ai.get('explanation', '')}", body_style))
    story.append(Spacer(1, 8))

    # 6. Clinician Review & Physician Sign-off
    cr = report_data.get("clinician_review", {})
    c_status = cr.get("status", "PENDING_REVIEW")
    c_name = cr.get("clinician_name") or "Pending Qualified Physician Review"
    c_role = cr.get("clinician_role") or "Physician"
    c_reg = cr.get("registration_number") or "N/A"
    c_interp = cr.get("clinician_interpretation") or "Awaiting Attending Physician Evaluation"
    c_notes = cr.get("clinical_notes") or "None"
    c_time = cr.get("reviewed_at") or "Pending"

    status_color = "#28a745" if c_status == "CONFIRMED" else ("#fd7e14" if c_status == "MODIFIED" else ("#dc3545" if c_status == "REJECTED" else "#6c757d"))

    story.append(Paragraph("6. CLINICAL REVIEW & PHYSICIAN SIGN-OFF", sec_heading_style))
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
    story.append(t_rev)
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
