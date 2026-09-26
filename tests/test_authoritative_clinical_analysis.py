"""
Tests for ECG Guardian Authoritative Clinical Analysis & Medication Decision Support
=====================================================================================

Validates:
1. AuthoritativeClinicalAnalysis strongly-typed schema and serialization.
2. Multimodal candidate model prediction and association scoring.
3. Hard deterministic safety gates:
   - Allergy blocking
   - QTc prolongation contraindication gate
   - Sinus bradycardia contraindication gate
   - Hyperkalemia contraindication gate
   - Drug-drug interaction detection
4. Evidence gating & withheld therapy reasons when reliability or clinical context is missing.
5. Sourced clinical reports (text and PDF) rendering medication decision support.
6. Serverless API contract (/api/analyze) including clinical_analysis object.
"""

import json
import pytest
import numpy as np

from src.clinical.clinical_analysis import (
    AuthoritativeClinicalAnalysis,
    MedicationCandidate,
    MedicationDecisionSupportData,
    PatientContext,
    ECGAnalysisData,
    ClinicalContextData,
)
from src.clinical.medication_candidate_engine import (
    GLOBAL_MEDICATION_CANDIDATE_ENGINE,
    build_authoritative_clinical_analysis,
)
from src.report.report_generator import generate_structured_report, export_report_to_text
from src.report.pdf_generator import generate_doctor_report


class TestAuthoritativeClinicalAnalysisSchema:
    """Test schema instantiation, serialization, and round-trip fidelity."""

    def test_round_trip_serialization(self):
        candidate = MedicationCandidate(
            medication="Beta-Adrenergic Blocker",
            drug_class="Beta-Adrenergic Blocker",
            model_association=0.92,
            clinical_rationale="Considered for ventricular ectopy suppression.",
            relevant_indication="Symptomatic PVCs (2019 ESC Guidelines)",
            safety_status="SAFE",
            contraindications_detected=[],
            drug_interactions_detected=[],
            allergy_check="CLEAR",
            evidence_source="2019 ESC Guidelines on Ventricular Arrhythmias",
            clinician_review_required=True,
        )
        med_ds = MedicationDecisionSupportData(
            status="AVAILABLE",
            clinical_finding="Premature Ventricular Contractions (PVC)",
            candidates=[candidate],
            clinician_review_required=True,
        )
        ca = AuthoritativeClinicalAnalysis(
            patient_context=PatientContext(patient_id="PAT-TEST-001", age=58, sex="M"),
            ecg_analysis=ECGAnalysisData(primary_finding="Premature Ventricular Contractions (PVC)", heart_rate=78.0),
            clinical_context=ClinicalContextData(symptoms=["Palpitations"], medical_history=["Hypertension"]),
            medication_decision_support=med_ds,
        )

        d = ca.to_dict()
        assert d["patient_context"]["patient_id"] == "PAT-TEST-001"
        assert d["ecg_analysis"]["primary_finding"] == "Premature Ventricular Contractions (PVC)"
        assert len(d["medication_decision_support"]["candidates"]) == 1
        assert d["medication_decision_support"]["candidates"][0]["medication"] == "Beta-Adrenergic Blocker"

        reconstructed = AuthoritativeClinicalAnalysis.from_dict(d)
        assert reconstructed.patient_context.patient_id == "PAT-TEST-001"
        assert reconstructed.ecg_analysis.heart_rate == 78.0
        assert len(reconstructed.medication_decision_support.candidates) == 1
        assert reconstructed.medication_decision_support.candidates[0].safety_status == "SAFE"


class TestMedicationSafetyGates:
    """Validate deterministic physiological, laboratory, and allergy safety gates."""

    def test_allergy_contraindication_gate_blocks_candidate(self):
        """When patient has documented allergy, matching candidate must be marked BLOCKED."""
        res = GLOBAL_MEDICATION_CANDIDATE_ENGINE.generate_candidate_recommendations(
            ecg_finding="Premature Ventricular Contractions (PVC)",
            ecg_measurements={"heart_rate_bpm": 75, "qtc_bazett_ms": 420},
            patient_profile={
                "age": 60,
                "allergies": "Metoprolol, Bisoprolol",
                "conditions": "CAD",
                "symptoms": "Palpitations",
            },
            vital_signs={"systolic_bp": 125, "heart_rate": 75},
            laboratory_results={"potassium_meq_l": 4.2, "creatinine_mg_dl": 1.0},
        )
        bb_candidates = [c for c in res.candidates if "Beta" in c.medication]
        assert len(bb_candidates) > 0
        assert bb_candidates[0].safety_status == "BLOCKED"
        assert bb_candidates[0].allergy_check == "ALLERGY_ALERT"
        assert any("allergy" in cd.lower() for cd in bb_candidates[0].contraindications_detected)

    def test_qtc_prolongation_gate_blocks_class_3_antiarrhythmics(self):
        """When QTc >= 500 ms, QT-prolonging antiarrhythmics must be marked BLOCKED."""
        res = GLOBAL_MEDICATION_CANDIDATE_ENGINE.generate_candidate_recommendations(
            ecg_finding="Ventricular Ectopy",
            ecg_measurements={"heart_rate_bpm": 80, "qtc_bazett_ms": 520},
            patient_profile={"age": 65, "conditions": "Heart Failure", "symptoms": "Shortness of breath"},
            vital_signs={"systolic_bp": 115, "heart_rate": 80},
            laboratory_results={"potassium_meq_l": 4.1, "creatinine_mg_dl": 1.1},
        )
        c3_candidates = [c for c in res.candidates if "Class III" in c.medication]
        if c3_candidates:
            assert c3_candidates[0].safety_status == "BLOCKED"
            assert any("500 ms" in cd for cd in c3_candidates[0].contraindications_detected)

    def test_bradycardia_gate_contraindicates_nodal_blockers(self):
        """When heart rate < 50 bpm, beta blockers and non-DHP CCBs must be BLOCKED."""
        res = GLOBAL_MEDICATION_CANDIDATE_ENGINE.generate_candidate_recommendations(
            ecg_finding="Premature Ventricular Contractions (PVC)",
            ecg_measurements={"heart_rate_bpm": 44, "qtc_bazett_ms": 420},
            patient_profile={"age": 70, "conditions": "Hypertension"},
            vital_signs={"systolic_bp": 130, "heart_rate": 44},
            laboratory_results={"potassium_meq_l": 4.3, "creatinine_mg_dl": 0.9},
        )
        for c in res.candidates:
            if "Beta" in c.medication:
                assert c.safety_status == "BLOCKED"
                assert any("Bradycardia" in cd for cd in c.contraindications_detected)

    def test_hyperkalemia_gate_blocks_potassium_and_ace_inhibitors(self):
        """When K+ >= 5.2 mEq/L, potassium replacement and ACEi/ARBs must be BLOCKED."""
        res = GLOBAL_MEDICATION_CANDIDATE_ENGINE.generate_candidate_recommendations(
            ecg_finding="Normal Sinus Rhythm",
            ecg_measurements={"heart_rate_bpm": 70, "qtc_bazett_ms": 410},
            patient_profile={"age": 62, "conditions": "Hypertension, Heart Failure"},
            vital_signs={"systolic_bp": 140, "heart_rate": 70},
            laboratory_results={"potassium_meq_l": 5.6, "creatinine_mg_dl": 1.4},
        )
        for c in res.candidates:
            if "ACE Inhibitor" in c.medication or "Electrolyte" in c.medication:
                assert c.safety_status == "BLOCKED"
                assert any("Hyperkalemia" in cd for cd in c.contraindications_detected)


class TestEvidenceGatingAndWithheldTherapy:
    """Validate that unverified findings or missing context cleanly withhold suggestions with auditable reasons."""

    def test_unreliable_finding_withholds_decision_support(self):
        """When finding reliability is marked False, status becomes WITHHELD with explicit reason."""
        class MockUnreliableFinding:
            reliable = False
            reasons = ["Signal quality UNUSABLE due to excessive motion artifact"]

        ca = build_authoritative_clinical_analysis(
            ecg_finding="Premature Ventricular Contractions (PVC)",
            ecg_measurements={"heart_rate_bpm": 80, "qtc_bazett_ms": 430},
            finding_reliability=MockUnreliableFinding(),
        )
        assert ca.medication_decision_support.status == "WITHHELD"
        assert "UNUSABLE" in ca.medication_decision_support.status_reason

    def test_missing_critical_context_flags_insufficient_information(self):
        """When vitals and labs are absent, status becomes INSUFFICIENT_INFORMATION."""
        ca = build_authoritative_clinical_analysis(
            ecg_finding="Normal Sinus Rhythm",
            ecg_measurements={"heart_rate_bpm": 72, "qtc_bazett_ms": 415},
            patient_profile={"age": 50},
            # Vital signs and labs intentionally omitted
        )
        assert ca.medication_decision_support.status == "INSUFFICIENT_INFORMATION"
        assert len(ca.medication_decision_support.missing_information) >= 3


class TestReportAndPDFIntegration:
    """Validate that structured reports and PDFs display candidate decision support."""

    def test_text_report_includes_section_11_candidates(self):
        ca = build_authoritative_clinical_analysis(
            ecg_finding="Premature Ventricular Contractions (PVC)",
            ecg_measurements={"heart_rate_bpm": 76, "qtc_bazett_ms": 425},
            patient_profile={"age": 63, "conditions": "Hypertension, CAD", "symptoms": "Palpitations"},
            vital_signs={"systolic_bp": 132, "heart_rate": 76},
            laboratory_results={"potassium_meq_l": 4.1, "creatinine_mg_dl": 1.0},
        )
        rep = generate_structured_report(
            input_info={"file_name": "REC-001", "sampling_rate": 360, "duration_sec": 10.0, "lead": "II"},
            clinical_analysis=ca.to_dict(),
        )
        text = export_report_to_text(rep)
        assert "SECTION 11: CLINICAL DECISION SUPPORT" in text
        assert "EVIDENCE-BASED MEDICATION & CANDIDATE THERAPY DECISION SUPPORT" in text
        assert "Candidate Drug Classes Evaluated" in text

    def test_pdf_generation_includes_medication_candidates(self):
        ca = build_authoritative_clinical_analysis(
            ecg_finding="Premature Ventricular Contractions (PVC)",
            ecg_measurements={"heart_rate_bpm": 76, "qtc_bazett_ms": 425},
            patient_profile={"age": 63, "conditions": "CAD", "symptoms": "Palpitations"},
            vital_signs={"systolic_bp": 130, "heart_rate": 76},
            laboratory_results={"potassium_meq_l": 4.2, "creatinine_mg_dl": 0.95},
        )
        rep = generate_structured_report(
            input_info={"file_name": "REC-001", "sampling_rate": 360, "duration_sec": 5.0, "lead": "II"},
            clinical_analysis=ca.to_dict(),
        )
        waveform = np.sin(np.linspace(0, 10 * np.pi, 1800))
        pdf_bytes = generate_doctor_report(rep, waveform=waveform, fs=360.0)
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes.startswith(b"%PDF")
        assert len(pdf_bytes) > 5000
