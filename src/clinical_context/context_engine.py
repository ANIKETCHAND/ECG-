"""
Clinical Context Fusion Engine
==============================

Phase 16:
Unifies patient history, acute symptoms, vital signs, laboratory panels,
medication regimens, allergies, and prior ECG comparisons with active ECG findings.

Strict Rules:
- Preserves the discrete source of every clinical statement (Rule 20 & 21).
- Every statement is marked with one of:
  [ECG-DERIVED], [PATIENT-HISTORY], [VITAL-SIGN], [LABORATORY],
  [MEDICATION-RECORD], [AI-GENERATED FINDING], [MACHINE INTERPRETATION],
  [CLINICAL DECISION SUPPORT], [CLINICIAN-AUTHORED], [NOT PROVIDED].
- Never asserts an unrecorded fact as normal (Rule 1).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.clinical.models import (
    ClinicalObservation,
    InformationSource,
    LaboratoryResults,
    Patient,
    Symptoms,
    VitalSigns,
)


@dataclass
class SourcedStatement:
    category: str  # "ECG_FINDING", "PATIENT_CONTEXT", "VITAL_SIGN", "LABORATORY", "MEDICATION", "CDS", "MISSING_INFO"
    statement: str
    source_tag: InformationSource
    timestamp: Optional[str] = None
    is_abnormal: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["source_tag"] = self.source_tag.value
        return d

    def format_line(self) -> str:
        time_suffix = f" (Observed: {self.timestamp[:19].replace('T', ' ')})" if self.timestamp else ""
        return f"• [{self.source_tag.value}] {self.statement}{time_suffix}"


class ClinicalContextFusionEngine:
    """Fuses multi-source medical telemetry with explicit provenance tracking."""

    def fuse_context(
        self,
        ecg_finding: str,
        patient: Optional[Patient] = None,
        as_of_timestamp: Optional[str] = None,
        quality_category: str = "GOOD",
        snr_db: Optional[float] = None,
        heart_rate: Optional[float] = None,
        machine_interpretation: Optional[List[str]] = None,
        prior_ecg_summary: Optional[str] = None,
    ) -> List[SourcedStatement]:
        statements: List[SourcedStatement] = []

        # 1. Primary ECG Findings
        statements.append(
            SourcedStatement(
                category="ECG_FINDING",
                statement=f"Primary rhythm classified as: {ecg_finding}",
                source_tag=InformationSource.AI_GENERATED,
                timestamp=as_of_timestamp,
            )
        )
        if heart_rate is not None:
            statements.append(
                SourcedStatement(
                    category="ECG_FINDING",
                    statement=f"Ventricular rate estimated at {heart_rate:.1f} BPM",
                    source_tag=InformationSource.MEASURED_ECG,
                    timestamp=as_of_timestamp,
                )
            )
        if quality_category != "GOOD":
            statements.append(
                SourcedStatement(
                    category="ECG_FINDING",
                    statement=f"Signal quality categorized as {quality_category} (SNR: {snr_db or 0.0:.1f} dB)",
                    source_tag=InformationSource.MEASURED_ECG,
                    timestamp=as_of_timestamp,
                    is_abnormal=True,
                )
            )

        # 2. Machine Interpretation (if present)
        if machine_interpretation:
            statements.append(
                SourcedStatement(
                    category="ECG_FINDING",
                    statement=f"Machine printed interpretation: {', '.join(machine_interpretation)}",
                    source_tag=InformationSource.MACHINE_GENERATED,
                    timestamp=as_of_timestamp,
                )
            )

        # If patient is unlinked / None
        if patient is None:
            statements.append(
                SourcedStatement(
                    category="MISSING_INFO",
                    statement="Patient demographics, history, and records: NOT PROVIDED",
                    source_tag=InformationSource.NOT_AVAILABLE,
                )
            )
            return statements

        # 3. Patient Demographics & Blood Group
        demo_str = f"Age: {patient.age if patient.age is not None else 'NOT PROVIDED'}, Sex: {patient.sex or 'NOT PROVIDED'}, Blood Group: {patient.blood_group or 'NOT PROVIDED'}"
        statements.append(
            SourcedStatement(
                category="PATIENT_CONTEXT",
                statement=demo_str,
                source_tag=InformationSource.RECORDED_HISTORY,
            )
        )

        # 4. Patient Medical & Cardiac History
        all_conditions = patient.existing_conditions + patient.cardiac_history
        if all_conditions:
            statements.append(
                SourcedStatement(
                    category="PATIENT_CONTEXT",
                    statement=f"Documented conditions: {', '.join(all_conditions)}",
                    source_tag=InformationSource.RECORDED_HISTORY,
                )
            )
        else:
            statements.append(
                SourcedStatement(
                    category="PATIENT_CONTEXT",
                    statement="Documented cardiac and medical history: None documented",
                    source_tag=InformationSource.RECORDED_HISTORY,
                )
            )

        # 5. Acute Symptoms (Point-in-Time filtered)
        if patient.symptoms:
            valid_symps = [s for s in patient.symptoms if not as_of_timestamp or (s.recorded_at and s.recorded_at <= as_of_timestamp)]
            if valid_symps:
                symp_descs = [f"{s.primary_symptom} (Severity: {s.severity})" for s in valid_symps if s.primary_symptom]
                statements.append(
                    SourcedStatement(
                        category="PATIENT_CONTEXT",
                        statement=f"Acute presenting symptoms: {', '.join(symp_descs)}",
                        source_tag=InformationSource.RECORDED_HISTORY,
                        timestamp=valid_symps[-1].recorded_at,
                    )
                )
            else:
                statements.append(
                    SourcedStatement(
                        category="MISSING_INFO",
                        statement="Acute symptoms as of recording time: NOT PROVIDED",
                        source_tag=InformationSource.NOT_AVAILABLE,
                    )
                )
        else:
            statements.append(
                SourcedStatement(
                    category="MISSING_INFO",
                    statement="Acute symptoms: NOT PROVIDED",
                    source_tag=InformationSource.NOT_AVAILABLE,
                )
            )

        # 6. Vital Signs (Point-in-Time filtered)
        vitals = patient.get_latest_vitals_before(as_of_timestamp)
        if vitals and (vitals.systolic_bp_mmhg is not None or vitals.spo2_percent is not None):
            v_parts = []
            if vitals.systolic_bp_mmhg is not None and vitals.diastolic_bp_mmhg is not None:
                v_parts.append(f"BP: {int(vitals.systolic_bp_mmhg)}/{int(vitals.diastolic_bp_mmhg)} mmHg")
            if vitals.spo2_percent is not None:
                v_parts.append(f"SpO2: {vitals.spo2_percent:.0f}%")
            if vitals.respiratory_rate_bpm is not None:
                v_parts.append(f"RR: {vitals.respiratory_rate_bpm:.0f}/min")
            statements.append(
                SourcedStatement(
                    category="VITAL_SIGN",
                    statement=f"Bedside vitals: {', '.join(v_parts)}",
                    source_tag=InformationSource.VITAL_SIGN,
                    timestamp=vitals.recorded_at,
                )
            )
        else:
            statements.append(
                SourcedStatement(
                    category="MISSING_INFO",
                    statement="Bedside vital signs: NOT PROVIDED",
                    source_tag=InformationSource.NOT_AVAILABLE,
                )
            )

        # 7. Laboratory Results (Point-in-Time filtered)
        labs = patient.get_latest_labs_before(as_of_timestamp)
        if labs and (labs.potassium_mmol_l is not None or labs.serum_creatinine_mg_dl is not None or labs.troponin_i_ng_ml is not None):
            l_parts = []
            if labs.potassium_mmol_l is not None:
                l_parts.append(f"Potassium: {labs.potassium_mmol_l:.2f} mmol/L")
            if labs.serum_creatinine_mg_dl is not None:
                l_parts.append(f"Creatinine: {labs.serum_creatinine_mg_dl:.2f} mg/dL")
            if labs.egfr_ml_min is not None:
                l_parts.append(f"eGFR: {labs.egfr_ml_min:.0f} mL/min")
            if labs.troponin_i_ng_ml is not None:
                l_parts.append(f"Troponin I: {labs.troponin_i_ng_ml:.3f} ng/mL")
            statements.append(
                SourcedStatement(
                    category="LABORATORY",
                    statement=f"Recent laboratory panel: {', '.join(l_parts)}",
                    source_tag=InformationSource.LABORATORY,
                    timestamp=labs.recorded_at,
                )
            )
        else:
            statements.append(
                SourcedStatement(
                    category="MISSING_INFO",
                    statement="Recent cardiac & metabolic laboratory data: NOT PROVIDED",
                    source_tag=InformationSource.NOT_AVAILABLE,
                )
            )

        # 8. Medication Regimen (Active at time of ECG)
        active_meds = patient.get_active_medications_at(as_of_timestamp)
        if active_meds:
            med_names = [f"{m.drug_name} {m.dosage or ''}".strip() for m in active_meds]
            statements.append(
                SourcedStatement(
                    category="MEDICATION",
                    statement=f"Active medications: {', '.join(med_names)}",
                    source_tag=InformationSource.MEDICATION_REGIMEN,
                )
            )
        else:
            statements.append(
                SourcedStatement(
                    category="MEDICATION",
                    statement="Active medication regimen: None documented",
                    source_tag=InformationSource.MEDICATION_REGIMEN,
                )
            )

        # 9. Prior ECG Evolution
        if prior_ecg_summary:
            statements.append(
                SourcedStatement(
                    category="ECG_FINDING",
                    statement=f"Longitudinal ECG Evolution: {prior_ecg_summary}",
                    source_tag=InformationSource.MEASURED_ECG,
                )
            )

        return statements


GLOBAL_CONTEXT_FUSION_ENGINE = ClinicalContextFusionEngine()
