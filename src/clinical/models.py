"""
ECG Guardian — Clinical Data Models & Patient Schema
===================================================

Phase 1 & Phase 2:
Defines immutable, strongly-typed clinical data structures for:
- Universal Patient Observation Model (value, unit, recorded_at, source, entered_by)
- Temporal Patient Record (Demographics, Blood Group, Vitals, Labs, Symptoms, Meds, History)
- Multimodal ECG Ingestion & Measurements
- AI Analysis & Machine Interpretation (Decoupled)
- Clinical Context & Medication Safety
- Clinician Review & Final Sealed Report

Strict Safety Rules:
- Rule 1: Never fabricate patient data (explicit "NOT PROVIDED" / "NOT AVAILABLE").
- Rule 2: Decoupled architecture — separate signal findings from patient context.
- Rule 3: Time-aware — observations carry recorded_at to prevent look-ahead bias.
- Rule 20: Every report statement carries an explicit source-attribution tag.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union


# ─────────────────────────────────────────────────────────────────────────────
# Provenance & Source Attribution Types (Phase 20 & 21)
# ─────────────────────────────────────────────────────────────────────────────

class InformationSource(str, Enum):
    MEASURED_ECG = "ECG-DERIVED"
    RECORDED_HISTORY = "PATIENT-HISTORY"
    VITAL_SIGN = "VITAL-SIGN"
    LABORATORY = "LABORATORY"
    MEDICATION_REGIMEN = "MEDICATION-RECORD"
    AI_GENERATED = "AI-GENERATED FINDING"
    MACHINE_GENERATED = "MACHINE INTERPRETATION"
    CLINICAL_DECISION_SUPPORT = "CLINICAL DECISION SUPPORT"
    CLINICIAN_AUTHORED = "CLINICIAN-AUTHORED"
    NOT_AVAILABLE = "NOT PROVIDED"


class QualityCategory(str, Enum):
    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    POOR = "POOR"
    UNUSABLE = "UNUSABLE"
    NOT_ASSESSED = "NOT_ASSESSED"


class ComparisonStatus(str, Enum):
    NEW = "NEW"
    PERSISTENT = "PERSISTENT"
    CHANGED = "CHANGED"
    RESOLVED = "RESOLVED"
    NOT_COMPARABLE = "NOT COMPARABLE"


# ─────────────────────────────────────────────────────────────────────────────
# Universal Clinical Observation Model (Phase 2 & 3)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClinicalObservation:
    """Base observation atom adhering to CDSCO / HL7 FHIR observation model."""
    name: str
    value: Any
    unit: Optional[str] = None
    recorded_at: Optional[str] = None  # ISO 8601 timestamp: YYYY-MM-DDTHH:MM:SS
    source: str = InformationSource.RECORDED_HISTORY.value
    entered_by: Optional[str] = "Clinical Staff"
    reference_range: Optional[str] = None
    is_abnormal: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def display_value(self) -> str:
        if self.value is None or str(self.value).strip() in ("", "None", "null"):
            return "NOT PROVIDED"
        if self.unit:
            return f"{self.value} {self.unit}"
        return str(self.value)


# ─────────────────────────────────────────────────────────────────────────────
# Patient Sub-Entities
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Allergy:
    allergen: str
    reaction: Optional[str] = None
    severity: str = "MODERATE"  # MILD, MODERATE, SEVERE, ANAPHYLACTIC
    recorded_at: Optional[str] = None
    source: str = InformationSource.RECORDED_HISTORY.value
    entered_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Medication:
    drug_name: str
    generic_name: Optional[str] = None
    dosage: Optional[str] = None
    unit: Optional[str] = None
    frequency: Optional[str] = None  # e.g., "OD", "BD", "PRN"
    route: str = "Oral"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    is_active: bool = True
    indication: Optional[str] = None
    recorded_at: Optional[str] = None
    source: str = InformationSource.MEDICATION_REGIMEN.value
    entered_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def is_active_at(self, timestamp_str: Optional[str]) -> bool:
        """Evaluate if medication was active at the given ECG timestamp."""
        if not self.is_active:
            return False
        if not timestamp_str or not self.start_date:
            return self.is_active
        try:
            target_t = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            start_t = datetime.fromisoformat(self.start_date.replace("Z", "+00:00"))
            if target_t < start_t:
                return False
            if self.end_date:
                end_t = datetime.fromisoformat(self.end_date.replace("Z", "+00:00"))
                if target_t > end_t:
                    return False
            return True
        except Exception:
            return self.is_active


@dataclass
class Symptoms:
    primary_symptom: Optional[str] = None  # Chest Pain, Dyspnea, Palpitations, Syncope
    onset_timestamp: Optional[str] = None
    duration_hours: Optional[float] = None
    severity: str = "NOT PROVIDED"  # MILD, MODERATE, SEVERE, NOT PROVIDED
    character: Optional[str] = None  # Crushing, Sharp, Pleuritic, Atypical
    associated_symptoms: List[str] = field(default_factory=list)
    recorded_at: Optional[str] = None
    source: str = InformationSource.RECORDED_HISTORY.value
    entered_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VitalSigns:
    heart_rate_bpm: Optional[float] = None
    systolic_bp_mmhg: Optional[float] = None
    diastolic_bp_mmhg: Optional[float] = None
    spo2_percent: Optional[float] = None
    respiratory_rate_bpm: Optional[float] = None
    temperature_celsius: Optional[float] = None
    recorded_at: Optional[str] = None
    source: str = InformationSource.VITAL_SIGN.value
    entered_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def blood_pressure_str(self) -> str:
        if self.systolic_bp_mmhg is not None and self.diastolic_bp_mmhg is not None:
            return f"{int(self.systolic_bp_mmhg)}/{int(self.diastolic_bp_mmhg)} mmHg"
        return "NOT PROVIDED"


@dataclass
class LaboratoryResults:
    potassium_mmol_l: Optional[float] = None       # 3.5 - 5.0
    magnesium_mg_dl: Optional[float] = None        # 1.7 - 2.2
    serum_creatinine_mg_dl: Optional[float] = None # 0.6 - 1.2
    egfr_ml_min: Optional[float] = None            # > 60
    troponin_i_ng_ml: Optional[float] = None       # < 0.04
    troponin_t_ng_ml: Optional[float] = None       # < 0.01
    bnp_pg_ml: Optional[float] = None              # < 100
    hemoglobin_g_dl: Optional[float] = None        # 12.0 - 17.0
    panel_name: str = "Cardiac & Metabolic Workup"
    recorded_at: Optional[str] = None
    source: str = InformationSource.LABORATORY.value
    entered_by: Optional[str] = None
    additional_observations: Dict[str, ClinicalObservation] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["additional_observations"] = {k: v.to_dict() for k, v in self.additional_observations.items()}
        return res


# ─────────────────────────────────────────────────────────────────────────────
# Patient Record (Phase 2 & Phase 8)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Patient:
    patient_id: str
    hospital_mrn: str
    name: str
    hospital_id: str = "HOSP-APEX"
    age: Optional[int] = None
    date_of_birth: Optional[str] = None
    sex: Optional[str] = None  # "Male", "Female", "Other"
    blood_group: Optional[str] = None  # "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    allergies: List[Allergy] = field(default_factory=list)
    smoking_status: Optional[str] = None  # "Never Smoker", "Former Smoker", "Active Smoker", None
    alcohol_status: Optional[str] = None
    existing_conditions: List[str] = field(default_factory=list)  # e.g., ["Hypertension", "Type 2 Diabetes"]
    cardiac_history: List[str] = field(default_factory=list)      # e.g., ["Prior STEMI (2024)", "CABG"]
    family_history: List[str] = field(default_factory=list)       # e.g., ["Premature CAD in father"]
    current_medications: List[Medication] = field(default_factory=list)
    symptoms: List[Symptoms] = field(default_factory=list)
    vital_signs: List[VitalSigns] = field(default_factory=list)
    laboratory_results: List[LaboratoryResults] = field(default_factory=list)
    previous_ecg_ids: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["allergies"] = [a.to_dict() for a in self.allergies]
        d["current_medications"] = [m.to_dict() for m in self.current_medications]
        d["symptoms"] = [s.to_dict() for s in self.symptoms]
        d["vital_signs"] = [v.to_dict() for v in self.vital_signs]
        d["laboratory_results"] = [l.to_dict() for l in self.laboratory_results]
        return d

    @property
    def bmi(self) -> Optional[float]:
        if self.height_cm and self.weight_kg and self.height_cm > 0:
            h_m = self.height_cm / 100.0
            return round(self.weight_kg / (h_m * h_m), 1)
        return None

    def get_latest_vitals_before(self, cutoff_iso: Optional[str]) -> Optional[VitalSigns]:
        """Time-aware getter: returns vital signs strictly recorded at or before cutoff."""
        if not self.vital_signs:
            return None
        valid = [v for v in self.vital_signs if not cutoff_iso or (v.recorded_at and v.recorded_at <= cutoff_iso)]
        if not valid:
            return None
        return sorted(valid, key=lambda x: x.recorded_at or "")[-1]

    def get_latest_labs_before(self, cutoff_iso: Optional[str]) -> Optional[LaboratoryResults]:
        """Time-aware getter: returns lab panel strictly recorded at or before cutoff."""
        if not self.laboratory_results:
            return None
        valid = [l for l in self.laboratory_results if not cutoff_iso or (l.recorded_at and l.recorded_at <= cutoff_iso)]
        if not valid:
            return None
        return sorted(valid, key=lambda x: x.recorded_at or "")[-1]

    def get_active_medications_at(self, cutoff_iso: Optional[str]) -> List[Medication]:
        """Time-aware getter: returns medications active at the specified timestamp."""
        return [m for m in self.current_medications if m.is_active_at(cutoff_iso)]


# ─────────────────────────────────────────────────────────────────────────────
# ECG Waveform, Measurements & Quality Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ECGRecording:
    record_id: str
    patient_id: Optional[str] = None
    hospital_id: str = "HOSP-APEX"
    recording_timestamp: Optional[str] = None
    sampling_rate: float = 360.0
    lead_names: List[str] = field(default_factory=lambda: ["Lead II"])
    duration_sec: float = 10.0
    source_format: str = "CSV"
    device_model: Optional[str] = "Clinical Diagnostic Holter / ECG Recorder"
    file_hash: Optional[str] = None
    raw_data_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ECGMeasurements:
    heart_rate_bpm: Optional[float] = None
    mean_rr_ms: Optional[float] = None
    pr_interval_ms: Optional[float] = None
    qrs_duration_ms: Optional[float] = None
    qt_interval_ms: Optional[float] = None
    qtc_bazett_ms: Optional[float] = None
    qtc_fridericia_ms: Optional[float] = None
    p_axis_deg: Optional[float] = None
    qrs_axis_deg: Optional[float] = None
    t_axis_deg: Optional[float] = None
    detected_beats_count: Optional[int] = None
    detected_r_peaks: List[int] = field(default_factory=list)
    source: str = InformationSource.MEASURED_ECG.value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ECGQuality:
    category: QualityCategory = QualityCategory.GOOD
    quality_score: float = 100.0
    snr_db: float = 25.0
    baseline_wander: bool = False
    powerline_interference: bool = False
    motion_artifacts: bool = False
    warnings: List[str] = field(default_factory=list)
    source: str = InformationSource.MEASURED_ECG.value

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        return d


# ─────────────────────────────────────────────────────────────────────────────
# AI Model Analysis vs Machine Interpretation (Decoupled — Rule 2 & 21)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AIAnalysis:
    analysis_id: str
    model_id: str = "ECG-RF-1.0.0"
    model_version: str = "1.0.0"
    primary_pattern: str = "Normal Sinus Rhythm"
    model_probabilities: Dict[str, float] = field(default_factory=dict)
    beat_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    aberrant_beat_indices: List[int] = field(default_factory=list)
    pvc_burden_percent: float = 0.0
    confidence_score: float = 0.95
    model_limitations: List[str] = field(default_factory=list)
    analyzed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = InformationSource.AI_GENERATED.value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MachineInterpretation:
    printed_statements: List[str] = field(default_factory=list)
    printed_heart_rate: Optional[float] = None
    printed_intervals: Dict[str, Optional[float]] = field(default_factory=dict)
    source_device: Optional[str] = "Clinical ECG Cart Algorithm"
    source: str = InformationSource.MACHINE_GENERATED.value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Clinical Context & Medication Safety Engines (Phase 16, 17, 18)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SafetyAlert:
    category: str  # "INTERACTION", "CONTRAINDICATION", "ALLERGY_WARNING", "QT_PROLONGATION"
    severity: str  # "CRITICAL", "MAJOR", "MODERATE", "INFO"
    title: str
    description: str
    clinical_recommendation: str
    source_guideline: str
    evidence_level: str = "Level A"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MedicationSafety:
    evaluated_regimen: List[str] = field(default_factory=list)
    alerts: List[SafetyAlert] = field(default_factory=list)
    has_critical_alerts: bool = False
    context_completeness: str = "COMPLETE"  # COMPLETE, PARTIAL, INSUFFICIENT
    missing_context_fields: List[str] = field(default_factory=list)
    source: str = InformationSource.CLINICAL_DECISION_SUPPORT.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluated_regimen": self.evaluated_regimen,
            "has_critical_alerts": self.has_critical_alerts,
            "context_completeness": self.context_completeness,
            "missing_context_fields": self.missing_context_fields,
            "alerts": [a.to_dict() for a in self.alerts],
            "source": self.source,
        }


@dataclass
class ClinicalRecommendation:
    primary_finding: str
    urgency: str = "ROUTINE REVIEW"  # ROUTINE REVIEW, PROMPT REVIEW, URGENT CLINICIAN REVIEW
    clinical_considerations: List[str] = field(default_factory=list)
    suggested_assessments: List[str] = field(default_factory=list)
    contraindication_warnings: List[str] = field(default_factory=list)
    relevant_guidelines: List[str] = field(default_factory=list)
    clinician_action_required: str = "Correlate ECG rhythm with bedside symptoms and patient history."
    missing_patient_information: List[str] = field(default_factory=list)
    source: str = InformationSource.CLINICAL_DECISION_SUPPORT.value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Clinician Review & Final Report (Phase 19, 20, 23)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DoctorReview:
    review_id: str
    analysis_id: str
    clinician_name: str
    clinician_role: str = "CARDIOLOGIST"
    registration_number: Optional[str] = None
    agreement_status: str = "CONFIRMED"  # CONFIRMED, MODIFIED, REJECTED, PENDING
    clinician_diagnosis: Optional[str] = None
    clinical_interpretation: Optional[str] = None
    treatment_plan: Optional[str] = None
    medication_plan: Optional[str] = None
    follow_up_recommendation: Optional[str] = None
    reviewed_at: Optional[str] = None
    digital_signature_hash: Optional[str] = None
    status: str = "SEALED"
    source: str = InformationSource.CLINICIAN_AUTHORED.value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PreviousECGComparison:
    has_previous_ecg: bool = False
    previous_record_id: Optional[str] = None
    previous_timestamp: Optional[str] = None
    delta_heart_rate_bpm: Optional[float] = None
    delta_qrs_duration_ms: Optional[float] = None
    delta_qtc_ms: Optional[float] = None
    rhythm_evolution: ComparisonStatus = ComparisonStatus.NOT_COMPARABLE
    pvc_burden_change: Optional[str] = None
    comparison_summary: str = "No prior ECG records available for comparison."
    source: str = InformationSource.MEASURED_ECG.value

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["rhythm_evolution"] = self.rhythm_evolution.value
        return d


@dataclass
class FinalReport:
    """Consolidated 12-Section Clinical Reference Report."""
    report_id: str
    generated_at: str
    hospital_name: str = "Apex Heart & Vascular Institute"
    hospital_department: str = "Department of Cardiac Electrophysiology & Clinical Decision Support"
    facility_id: str = "HOSP-APEX-01"
    
    # 1. Patient & Recording Information
    patient: Patient = field(default_factory=lambda: Patient(patient_id="ANON", hospital_mrn="MRN-UNKNOWN", name="Anonymous"))
    recording: ECGRecording = field(default_factory=lambda: ECGRecording(record_id="REC-UNKNOWN"))
    
    # 2. Signal Quality & Cardiac Parameters
    quality: ECGQuality = field(default_factory=ECGQuality)
    measurements: ECGMeasurements = field(default_factory=ECGMeasurements)
    
    # 3. AI Abnormality Analysis
    ai_analysis: AIAnalysis = field(default_factory=lambda: AIAnalysis(analysis_id="ANL-UNKNOWN"))
    
    # 4. Waveform & Evidence Path / Metadata
    waveform_sample_count: int = 0
    detected_r_peak_indices: List[int] = field(default_factory=list)
    
    # 5. Factual Findings & Plain Language Explanation
    factual_findings: List[str] = field(default_factory=list)
    plain_explanation: str = ""
    
    # 6. Patient Clinical Context (Vitals, Labs, Symptoms, History)
    active_vitals: Optional[VitalSigns] = None
    active_labs: Optional[LaboratoryResults] = None
    acute_symptoms: Optional[Symptoms] = None
    
    # 7. Clinical Decision Support
    decision_support: ClinicalRecommendation = field(default_factory=lambda: ClinicalRecommendation(primary_finding="Normal"))
    
    # 8. Medication Safety & Interactions
    medication_safety: MedicationSafety = field(default_factory=MedicationSafety)
    
    # 9. AI vs Machine Interpretation Comparison
    machine_interpretation: MachineInterpretation = field(default_factory=MachineInterpretation)
    ai_machine_concordance: str = "CONCORDANT"
    concordance_notes: str = ""
    
    # 10. Previous ECG Comparison
    longitudinal_comparison: PreviousECGComparison = field(default_factory=PreviousECGComparison)
    
    # 11 & 12. Clinical Review & Sign-Off
    clinician_review: Optional[DoctorReview] = None
    
    # Regulatory Disclaimer
    regulatory_disclaimer: str = (
        "SaMD Class B Clinical Decision Support System adhering to CDSCO MDR 2017 & IEC 62304. "
        "Strict Directive: NO RELIABLE INPUT = NO AI RESULT. The analytical outputs and decision support "
        "recommendations are non-autonomous and subject to mandatory review and final prescription by a "
        "qualified medical practitioner."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "hospital_name": self.hospital_name,
            "hospital_department": self.hospital_department,
            "facility_id": self.facility_id,
            "patient": self.patient.to_dict(),
            "recording": self.recording.to_dict(),
            "quality": self.quality.to_dict(),
            "measurements": self.measurements.to_dict(),
            "ai_analysis": self.ai_analysis.to_dict(),
            "waveform_sample_count": self.waveform_sample_count,
            "detected_r_peak_indices": self.detected_r_peak_indices,
            "factual_findings": self.factual_findings,
            "plain_explanation": self.plain_explanation,
            "active_vitals": self.active_vitals.to_dict() if self.active_vitals else None,
            "active_labs": self.active_labs.to_dict() if self.active_labs else None,
            "acute_symptoms": self.acute_symptoms.to_dict() if self.acute_symptoms else None,
            "decision_support": self.decision_support.to_dict(),
            "medication_safety": self.medication_safety.to_dict(),
            "machine_interpretation": self.machine_interpretation.to_dict(),
            "ai_machine_concordance": self.ai_machine_concordance,
            "concordance_notes": self.concordance_notes,
            "longitudinal_comparison": self.longitudinal_comparison.to_dict(),
            "clinician_review": self.clinician_review.to_dict() if self.clinician_review else None,
            "regulatory_disclaimer": self.regulatory_disclaimer,
        }
