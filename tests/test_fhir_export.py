"""
Tests for FHIR R4 export.

Beyond structure, these tests pin two honesty properties: heart rate is the only
element granted a LOINC coding, and every other interval is emitted without a
fabricated code while the bundle records that terminology validation is pending.
"""

from __future__ import annotations

import pytest

from src.report.fhir_export import (
    LOCAL_CODE_SYSTEM,
    LOINC_HEART_RATE,
    TERMINOLOGY_VALIDATION_REQUIRED,
    to_fhir_bundle,
)


def _report(**overrides) -> dict:
    report = {
        "report_title": "AI ECG SCREENING REPORT",
        "generated_at": "2026-09-25T10:00:00Z",
        "patient_info": {
            "patient_id": "PAT-001",
            "patient_name": "Test Patient",
            "hospital_mrn": "MRN-9",
            "patient_age": 62,
            "patient_sex": "M",
        },
        "signal_quality": {"category": "GOOD"},
        "cardiac_parameters": {
            "heart_rate_bpm": 72,
            "pr_interval_ms": 156,
            "qrs_duration_ms": 94,
            "qt_interval_ms": 398,
            "qtc_interval_ms": 418,
            "machine_interpretation": "Possible PVC",
        },
        "ai_analysis": {
            "primary_finding": "Ventricular Ectopy",
            "model_id": "ECG-RF-1.0.0",
            "class_probabilities": {"Normal": 0.7, "PVC": 0.3},
        },
        "machine_comparison": {"status": "AGREEMENT", "clinical_advisory": "Routine review."},
        "clinician_review": {"status": "PENDING_REVIEW"},
        "disclaimer": "Research use only.",
    }
    report.update(overrides)
    return report


def _resources(bundle, resource_type):
    return [e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == resource_type]


def test_bundle_is_a_fhir_collection():
    bundle = to_fhir_bundle(_report())
    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "collection"
    assert bundle["entry"]


def test_patient_resource_is_produced_and_referenced():
    bundle = to_fhir_bundle(_report())
    patients = _resources(bundle, "Patient")
    assert len(patients) == 1
    assert patients[0]["name"][0]["text"] == "Test Patient"
    assert patients[0]["gender"] == "male"

    for observation in _resources(bundle, "Observation"):
        assert observation["subject"]["reference"] == f"Patient/{patients[0]['id']}"


def test_anonymous_patient_emits_no_patient_resource():
    bundle = to_fhir_bundle(_report(patient_info={"patient_id": "PAT-ANON"}))
    assert _resources(bundle, "Patient") == []


def test_audit_trail_tag_is_always_present():
    bundle = to_fhir_bundle(_report())
    codes = {tag["code"] for tag in bundle["meta"]["tag"]}
    assert "ai-assisted-clinical-decision-support" in codes
    assert "requires-physician-review" in codes


def test_only_heart_rate_carries_a_loinc_coding():
    bundle = to_fhir_bundle(_report())
    loinc_coded = []
    for observation in _resources(bundle, "Observation"):
        for coding in observation.get("code", {}).get("coding", []):
            if coding.get("system") == "http://loinc.org":
                loinc_coded.append(coding["code"])

    assert loinc_coded == [LOINC_HEART_RATE]


def test_interval_observations_explain_the_missing_loinc_code():
    bundle = to_fhir_bundle(_report())
    by_text = {o["code"]["text"]: o for o in _resources(bundle, "Observation")}

    qrs = by_text["QRS duration"]
    assert all(c["system"] != "http://loinc.org" for c in qrs["code"]["coding"])
    assert "terminology server" in qrs["note"][0]["text"]

    heart_rate = by_text["Heart rate"]
    assert "note" not in heart_rate


def test_bundle_declares_pending_terminology_validation():
    bundle = to_fhir_bundle(_report())
    extension = {
        e["url"]: e["valueString"] for e in bundle.get("extension", []) if "url" in e
    }
    assert "urn:ecg-guardian:terminology-validation-required" in extension
    assert len(TERMINOLOGY_VALIDATION_REQUIRED) >= 4


def test_ai_finding_carries_probabilities_as_components():
    bundle = to_fhir_bundle(_report())
    ai_observation = next(
        o for o in _resources(bundle, "Observation") if o["id"] == "ecg-guardian-ai-finding"
    )
    assert ai_observation["valueCodeableConcept"]["text"] == "Ventricular Ectopy"
    component_texts = {c["code"]["text"] for c in ai_observation["component"]}
    assert component_texts == {"Normal", "PVC"}
    assert "not a diagnosis" in ai_observation["note"][0]["text"]


def test_signal_quality_is_recorded_as_a_coded_value():
    bundle = to_fhir_bundle(_report())
    quality = next(
        o for o in _resources(bundle, "Observation") if o["id"] == "ecg-guardian-signal-quality"
    )
    assert quality["valueCodeableConcept"]["text"] == "GOOD"


def test_diagnostic_report_conclusion_combines_ai_machine_and_comparison():
    bundle = to_fhir_bundle(_report())
    diagnostic = _resources(bundle, "DiagnosticReport")[0]
    assert "Ventricular Ectopy" in diagnostic["conclusion"]
    assert "Possible PVC" in diagnostic["conclusion"]
    assert "AGREEMENT" in diagnostic["conclusion"]
    assert diagnostic["status"] == "preliminary"
    assert diagnostic["result"]


def test_signed_review_marks_the_report_final():
    bundle = to_fhir_bundle(_report(clinician_review={"status": "SIGNED"}))
    diagnostic = _resources(bundle, "DiagnosticReport")[0]
    assert diagnostic["status"] == "final"


def test_missing_sections_are_skipped_not_filled():
    bundle = to_fhir_bundle({"report_title": "Minimal"})
    assert _resources(bundle, "Patient") == []
    observations = _resources(bundle, "Observation")
    assert observations == []
    diagnostic = _resources(bundle, "DiagnosticReport")[0]
    assert "No AI or machine finding" in diagnostic["conclusion"]


def test_unavailable_values_are_not_emitted_as_numbers():
    report = _report(
        cardiac_parameters={
            "heart_rate_bpm": "NOT PROVIDED",
            "pr_interval_ms": None,
            "machine_interpretation": None,
        }
    )
    bundle = to_fhir_bundle(report)
    # No measurement Observation may be emitted for an unmeasured value. The
    # signal-quality and AI observations are unrelated to these fields.
    measurement_ids = {
        "ecg-guardian-heart-rate",
        "ecg-guardian-pr-interval",
        "ecg-guardian-qrs-duration",
        "ecg-guardian-qt-interval",
        "ecg-guardian-qtc-interval",
        "ecg-guardian-mean-rr",
    }
    emitted = {o["id"] for o in _resources(bundle, "Observation")}
    assert emitted & measurement_ids == set()


def test_values_embedded_in_printed_strings_are_parsed():
    bundle = to_fhir_bundle(_report(cardiac_parameters={"heart_rate_bpm": "72 bpm"}))
    heart_rate = next(
        o for o in _resources(bundle, "Observation") if o["id"] == "ecg-guardian-heart-rate"
    )
    assert heart_rate["valueQuantity"]["value"] == pytest.approx(72.0)
    assert heart_rate["valueQuantity"]["code"] == "/min"


def test_result_references_are_unique():
    bundle = to_fhir_bundle(_report())
    diagnostic = _resources(bundle, "DiagnosticReport")[0]
    references = [entry["reference"] for entry in diagnostic["result"]]
    assert len(references) == len(set(references))


def test_bundle_is_json_serialisable():
    import json

    bundle = to_fhir_bundle(_report())
    assert json.loads(json.dumps(bundle))["resourceType"] == "Bundle"


def test_local_code_system_is_used_for_non_loinc_elements():
    bundle = to_fhir_bundle(_report())
    systems = {
        coding["system"]
        for observation in _resources(bundle, "Observation")
        for coding in observation.get("code", {}).get("coding", [])
    }
    assert LOCAL_CODE_SYSTEM in systems
