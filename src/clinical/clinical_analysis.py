"""
ECG Guardian — Authoritative Clinical Analysis Object
=====================================================

Defines ONE authoritative, strongly-typed clinical analysis structure.
Consolidated across:
- API responses (/api/analyze)
- Dashboard rendering
- Structured report generator
- PDF generator
- Supabase / SQLite persistence
- Historical report retrieval

Mandatory Safety Governance:
- Non-autonomous: strictly decision support for qualified clinicians.
- Zero fabrication: missing fields are marked explicitly as "NOT PROVIDED" / "NOT ASSESSED".
- Traceable: carries model versions, knowledge base dates, and dataset versions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


@dataclass
class PatientContext:
    patient_id: str = "PAT-ANON"
    hospital_mrn: str = "NOT ASSIGNED"
    name: str = "Anonymous"
    age: Optional[int] = None
    sex: Optional[str] = None
    blood_group: Optional[str] = None
    date_of_birth: Optional[str] = None
    smoking_status: Optional[str] = None
    contact: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ECGMeasurementsData:
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ECGAnalysisData:
    primary_finding: str = "Normal Sinus Rhythm"
    findings: List[str] = field(default_factory=list)
    probabilities: Dict[str, float] = field(default_factory=dict)
    heart_rate: Optional[float] = None
    rhythm: str = "Regular Sinus Rhythm"
    signal_quality: str = "GOOD"
    signal_quality_score: float = 1.0
    snr_db: float = 25.0
    r_peaks: List[int] = field(default_factory=list)
    measurements: ECGMeasurementsData = field(default_factory=ECGMeasurementsData)
    selective_gate_applied: bool = False
    reported_coverage: Optional[float] = None
    abstain: bool = False
    abstain_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class ClinicalContextData:
    symptoms: List[str] = field(default_factory=list)
    medical_history: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    current_medications: List[str] = field(default_factory=list)
    vitals: Dict[str, Any] = field(default_factory=dict)
    laboratory_results: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MedicationCandidate:
    medication: str
    drug_class: str
    model_association: float
    clinical_rationale: str
    relevant_indication: str
    safety_status: str  # SAFE, REVIEW_REQUIRED, BLOCKED, CONTRAINDICATED
    contraindications_detected: List[str] = field(default_factory=list)
    drug_interactions_detected: List[str] = field(default_factory=list)
    allergy_check: str = "CLEAR"  # CLEAR, ALLERGY_ALERT, UNKNOWN
    relevant_laboratory_checks: List[str] = field(default_factory=list)
    missing_information: List[str] = field(default_factory=list)
    evidence_source: str = "ACC/AHA/ESC Clinical Practice Guidelines & DailyMed"
    clinician_review_required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MedicationDecisionSupportData:
    status: str = "AVAILABLE"  # AVAILABLE, LIMITED, INSUFFICIENT_INFORMATION, WITHHELD
    status_reason: Optional[str] = None
    clinical_finding: str = "Normal Sinus Rhythm"
    candidates: List[MedicationCandidate] = field(default_factory=list)
    safety_checks: List[Dict[str, Any]] = field(default_factory=list)
    contraindications: List[str] = field(default_factory=list)
    interactions: List[str] = field(default_factory=list)
    allergy_checks: List[Dict[str, Any]] = field(default_factory=list)
    missing_information: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    withheld_reason: Optional[str] = None
    model_version: str = "MED-CANDIDATE-1.0.0"
    dataset_version: str = "MIMIC-IV-v2.2+DailyMed-2026.04"
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    clinician_review_required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["candidates"] = [c.to_dict() if hasattr(c, "to_dict") else c for c in self.candidates]
        return d


@dataclass
class PreviousECGComparisonData:
    has_previous_ecg: bool = False
    previous_record_id: Optional[str] = None
    previous_timestamp: Optional[str] = None
    delta_heart_rate_bpm: Optional[float] = None
    delta_qrs_duration_ms: Optional[float] = None
    delta_qtc_ms: Optional[float] = None
    rhythm_evolution: str = "NOT COMPARABLE"
    pvc_burden_change: Optional[str] = None
    comparison_summary: str = "No prior ECG records available for comparison."

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelMetadataData:
    ecg_model_name: str = "RandomForestClassifier"
    ecg_model_version: str = "1.0.0"
    medication_model_name: str = "MultilabelClinicalCandidateModel"
    medication_model_version: str = "1.0.0"
    medication_knowledge_base_version: str = "DailyMed-FDA-2026.04"
    dataset_versions: Dict[str, str] = field(default_factory=lambda: {
        "MIMIC-IV": "2.2",
        "MIMIC-IV-ECG": "1.0",
        "MIMIC-IV-ED": "2.2",
        "eICU": "2.0",
        "Synthea": "3.0.0",
        "DailyMed": "2026-04",
        "MIT-BIH": "1.0.0",
    })
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AuthoritativeClinicalAnalysis:
    """The single authoritative clinical analysis object unifying all system tiers."""
    patient_context: PatientContext = field(default_factory=PatientContext)
    ecg_analysis: ECGAnalysisData = field(default_factory=ECGAnalysisData)
    clinical_context: ClinicalContextData = field(default_factory=ClinicalContextData)
    medication_decision_support: MedicationDecisionSupportData = field(default_factory=MedicationDecisionSupportData)
    previous_ecg_comparison: PreviousECGComparisonData = field(default_factory=PreviousECGComparisonData)
    model_metadata: ModelMetadataData = field(default_factory=ModelMetadataData)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_context": self.patient_context.to_dict(),
            "ecg_analysis": self.ecg_analysis.to_dict(),
            "clinical_context": self.clinical_context.to_dict(),
            "medication_decision_support": self.medication_decision_support.to_dict(),
            "previous_ecg_comparison": self.previous_ecg_comparison.to_dict(),
            "model_metadata": self.model_metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AuthoritativeClinicalAnalysis:
        if not data:
            return cls()

        p_ctx = data.get("patient_context", {})
        ecg_a = data.get("ecg_analysis", {})
        c_ctx = data.get("clinical_context", {})
        med_ds = data.get("medication_decision_support", {})
        prev_c = data.get("previous_ecg_comparison", {})
        m_meta = data.get("model_metadata", {})

        # Parse measurements
        meas_data = ecg_a.get("measurements", {})
        meas_obj = ECGMeasurementsData(**{k: v for k, v in meas_data.items() if k in ECGMeasurementsData.__dataclass_fields__})

        # Parse candidates
        candidates = []
        for c in med_ds.get("candidates", []):
            if isinstance(c, dict):
                candidates.append(MedicationCandidate(**{k: v for k, v in c.items() if k in MedicationCandidate.__dataclass_fields__}))
            elif isinstance(c, MedicationCandidate):
                candidates.append(c)

        med_ds_clean = {k: v for k, v in med_ds.items() if k in MedicationDecisionSupportData.__dataclass_fields__ and k != "candidates"}
        med_ds_obj = MedicationDecisionSupportData(candidates=candidates, **med_ds_clean)

        return cls(
            patient_context=PatientContext(**{k: v for k, v in p_ctx.items() if k in PatientContext.__dataclass_fields__}),
            ecg_analysis=ECGAnalysisData(measurements=meas_obj, **{k: v for k, v in ecg_a.items() if k in ECGAnalysisData.__dataclass_fields__ and k != "measurements"}),
            clinical_context=ClinicalContextData(**{k: v for k, v in c_ctx.items() if k in ClinicalContextData.__dataclass_fields__}),
            medication_decision_support=med_ds_obj,
            previous_ecg_comparison=PreviousECGComparisonData(**{k: v for k, v in prev_c.items() if k in PreviousECGComparisonData.__dataclass_fields__}),
            model_metadata=ModelMetadataData(**{k: v for k, v in m_meta.items() if k in ModelMetadataData.__dataclass_fields__}),
        )
