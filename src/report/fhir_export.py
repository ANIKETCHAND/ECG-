"""
FHIR R4 Export
==============

Converts a structured ECG Guardian report into a FHIR R4 ``Bundle`` so the
output can leave the research sandbox and enter a hospital EMR.

Shapes produced
---------------
* ``Patient`` — demographics, when present.
* ``Observation`` — one per measured cardiac parameter (heart rate, PR, QRS,
  QT/QTc, mean RR) plus signal quality and the AI classification, the latter
  carrying per-class probabilities as components.
* ``DiagnosticReport`` — the container, with ``conclusion`` carrying the AI
  finding, the ECG machine's printed interpretation, and any AI-vs-machine
  comparison status.

Terminology honesty
-------------------
Only the heart-rate code is emitted with a LOINC coding, because that is the one
mapping this project can assert confidently offline. Every other coded element is
emitted as ``text`` only, and the codes that still need terminology-server
validation are listed in ``TERMINOLOGY_VALIDATION_REQUIRED`` and surfaced in the
bundle's ``meta.tag``. Inventing LOINC codes for intervals would inject wrong
data into a downstream system, which is worse than omitting them.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any, Dict, List, Optional

#: Mapping that is safe to assert. Heart rate is unambiguous in LOINC.
LOINC_HEART_RATE = "8867-4"

#: Elements that still require a terminology-server lookup before clinical use.
#: Each entry is (report field, what it needs).
TERMINOLOGY_VALIDATION_REQUIRED = [
    ("pr_interval_ms", "no verified LOINC code asserted for PR interval"),
    ("qrs_duration_ms", "no verified LOINC code asserted for QRS duration"),
    ("qt_interval_ms", "no verified LOINC code asserted for QT interval"),
    ("qtc_interval_ms", "no verified LOINC code asserted for QTc interval"),
    ("mean_rr_ms", "no verified LOINC code asserted for mean RR interval"),
]

LOCAL_CODE_SYSTEM = "urn:ecg-guardian:local-codes"
AI_FINDING_CODE = "ai-ecg-finding"
SIGNAL_QUALITY_CODE = "ecg-signal-quality"


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.upper() in {"NOT PROVIDED", "NOT ASSIGNED", "N/A", "NONE"}:
            return None
        # Tolerate values like "4.1 mEq/L" or "72 bpm" pulled from printed reports.
        for token in text.replace(",", " ").split():
            try:
                return float(token)
            except ValueError:
                continue
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _observation(
    *,
    observation_id: str,
    patient_ref: Optional[str],
    status: str = "final",
    code_text: str,
    code: Optional[str] = None,
    value: Optional[float] = None,
    unit: Optional[str] = None,
    ucum_code: Optional[str] = None,
    components: Optional[List[Dict[str, Any]]] = None,
    note: Optional[str] = None,
    effective: Optional[str] = None,
    category: str = "procedure",
) -> Dict[str, Any]:
    coding: List[Dict[str, Any]] = []
    if code:
        coding.append({"system": "http://loinc.org", "code": code, "display": code_text})
    coding.append({"system": LOCAL_CODE_SYSTEM, "code": code_text.lower().replace(" ", "-"), "display": code_text})

    resource: Dict[str, Any] = {
        "resourceType": "Observation",
        "id": observation_id,
        "status": status,
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": category,
                    }
                ]
            }
        ],
        "code": {"coding": coding, "text": code_text},
    }

    if patient_ref:
        resource["subject"] = {"reference": patient_ref}
    if effective:
        resource["effectiveDateTime"] = effective

    if value is not None:
        quantity: Dict[str, Any] = {"value": value}
        if unit:
            quantity["unit"] = unit
        if ucum_code:
            quantity["system"] = "http://unitsofmeasure.org"
            quantity["code"] = ucum_code
        resource["valueQuantity"] = quantity
    if components:
        resource["component"] = components
    if note:
        resource["note"] = [{"text": note}]
    return resource


def to_fhir_bundle(report: Dict[str, Any], *, bundle_id: Optional[str] = None) -> Dict[str, Any]:
    """Convert a structured report dict into a FHIR R4 collection Bundle.

    Args:
        report: Output of ``generate_structured_report`` (or any dict with the
            same key shape). Missing sections are skipped rather than filled with
            placeholder values.
        bundle_id: Optional explicit bundle id.

    Returns:
        A FHIR R4 ``Bundle`` as a plain dict, ready for JSON serialisation.
    """
    bundle_id = bundle_id or str(uuid.uuid4())
    entries: List[Dict[str, Any]] = []
    tags: List[Dict[str, str]] = [
        {"system": "urn:ecg-guardian:provenance", "code": "ai-assisted-clinical-decision-support"},
        {"system": "urn:ecg-guardian:provenance", "code": "requires-physician-review"},
    ]

    patient_info = report.get("patient_info") or {}
    patient_id_value = patient_info.get("patient_id")
    patient_ref: Optional[str] = None

    if patient_id_value and str(patient_id_value).upper() not in {"PAT-ANON", "NOT PROVIDED"}:
        patient_resource_id = f"ecg-guardian-patient-{patient_id_value}"
        patient_ref = f"Patient/{patient_resource_id}"

        patient_resource: Dict[str, Any] = {
            "resourceType": "Patient",
            "id": patient_resource_id,
            "identifier": [{"system": "urn:ecg-guardian:patient-id", "value": str(patient_id_value)}],
        }
        name = patient_info.get("patient_name")
        if name and str(name).upper() not in {"NOT PROVIDED", "ANONYMOUS"}:
            patient_resource["name"] = [{"text": str(name)}]
        mrn = patient_info.get("hospital_mrn")
        if mrn and str(mrn).upper() not in {"NOT PROVIDED", "NOT ASSIGNED"}:
            patient_resource.setdefault("identifier", []).append(
                {"system": "urn:ecg-guardian:mrn", "value": str(mrn)}
            )
        sex = str(patient_info.get("patient_sex") or "").upper()
        if sex.startswith("M"):
            patient_resource["gender"] = "male"
        elif sex.startswith("F"):
            patient_resource["gender"] = "female"
        age = _as_float(patient_info.get("patient_age"))
        if age is not None:
            patient_resource["extension"] = [
                {
                    "url": "urn:ecg-guardian:age-years",
                    "valueInteger": int(age),
                    "note": "Age as recorded at reporting time; not a birthDate.",
                }
            ]
        entries.append({"fullUrl": patient_ref, "resource": patient_resource})

    effective = report.get("generated_at")

    # --- Measured cardiac parameters -----------------------------------------
    cardiac = report.get("cardiac_parameters") or {}
    measurements = [
        ("heart-rate", "Heart rate", _as_float(cardiac.get("heart_rate_bpm")), "/min", "/min", LOINC_HEART_RATE),
        ("pr-interval", "PR interval", _as_float(cardiac.get("pr_interval_ms")), "ms", "ms", None),
        ("qrs-duration", "QRS duration", _as_float(cardiac.get("qrs_duration_ms")), "ms", "ms", None),
        ("qt-interval", "QT interval", _as_float(cardiac.get("qt_interval_ms")), "ms", "ms", None),
        ("qtc-interval", "QTc interval", _as_float(cardiac.get("qtc_interval_ms")), "ms", "ms", None),
        ("mean-rr", "Mean RR interval", _as_float(cardiac.get("mean_rr_ms")), "ms", "ms", None),
    ]

    measurement_refs: List[str] = []
    for suffix, label, value, unit, ucum, code in measurements:
        if value is None:
            continue
        observation_id = f"ecg-guardian-{suffix}"
        observation_ref = f"Observation/{observation_id}"
        note = None
        if code is None:
            note = (
                "Emitted without a LOINC coding: no verified interval code is asserted by this system. "
                "Map via a terminology server before production use."
            )
        entries.append(
            {
                "fullUrl": observation_ref,
                "resource": _observation(
                    observation_id=observation_id,
                    patient_ref=patient_ref,
                    code_text=label,
                    code=code,
                    value=value,
                    unit=unit,
                    ucum_code=ucum,
                    note=note,
                    effective=effective,
                    category="procedure",
                ),
            }
        )
        measurement_refs.append(observation_ref)

    # --- Signal quality ------------------------------------------------------
    quality = report.get("signal_quality") or {}
    quality_category = quality.get("category")
    if quality_category:
        observation_id = "ecg-guardian-signal-quality"
        quality_ref = f"Observation/{observation_id}"
        entries.append(
            {
                "fullUrl": quality_ref,
                "resource": _observation(
                    observation_id=observation_id,
                    patient_ref=patient_ref,
                    code_text="12-lead ECG signal quality, category",
                    value=None,
                    components=None,
                    note=(
                        f"Quality gate outcome: {quality_category}. "
                        "Analysis proceeds only when signal quality is GOOD or ACCEPTABLE."
                    ),
                    effective=effective,
                ),
            }
        )
        quality_ref_holder = quality_ref
        # Encode the category as a valueCodeableConcept (not a Quantity).
        entries[-1]["resource"]["valueCodeableConcept"] = {
            "coding": [{"system": LOCAL_CODE_SYSTEM, "code": str(quality_category)}],
            "text": str(quality_category),
        }
    else:
        quality_ref_holder = None

    # --- AI classification ---------------------------------------------------
    ai = report.get("ai_analysis") or {}
    ai_prediction = ai.get("primary_finding") or ai.get("prediction")
    if ai_prediction:
        observation_id = "ecg-guardian-ai-finding"
        ai_ref = f"Observation/{observation_id}"
        probabilities = ai.get("class_probabilities") or ai.get("probabilities") or {}
        components = [
            {
                "code": {
                    "coding": [{"system": LOCAL_CODE_SYSTEM, "code": f"probability-{name}"}],
                    "text": str(name),
                },
                "valueQuantity": {
                    "value": _as_float(value),
                    "unit": "probability",
                    "system": "http://unitsofmeasure.org",
                    "code": "1",
                },
            }
            for name, value in (probabilities or {}).items()
            if _as_float(value) is not None
        ]

        model_id = ai.get("model_id") or ai.get("model_version")
        entries.append(
            {
                "fullUrl": ai_ref,
                "resource": _observation(
                    observation_id=observation_id,
                    patient_ref=patient_ref,
                    code_text="AI ECG finding",
                    value=None,
                    components=components or None,
                    note=(
                        "Model output, not a diagnosis. "
                        f"Model: {model_id or 'unspecified'}. "
                        f"Probability semantics: {ai.get('probability_semantics') or 'model class probability'}. "
                        "Physician review is mandatory."
                    ),
                    effective=effective,
                ),
            }
        )
        entries[-1]["resource"]["valueCodeableConcept"] = {
            "coding": [{"system": LOCAL_CODE_SYSTEM, "code": AI_FINDING_CODE}],
            "text": str(ai_prediction),
        }
        measurement_refs.append(ai_ref)
        if quality_ref_holder:
            measurement_refs.append(quality_ref_holder)
    else:
        ai_ref = None

    # --- DiagnosticReport ----------------------------------------------------
    conclusion_parts: List[str] = [f"AI finding: {ai_prediction}"] if ai_prediction else []
    machine_interpretation = cardiac.get("machine_interpretation")
    if machine_interpretation:
        conclusion_parts.append(f"ECG machine interpretation: {machine_interpretation}")

    comparison = report.get("machine_comparison") or {}
    if comparison.get("status"):
        conclusion_parts.append(f"AI vs machine comparison: {comparison['status']}")
        if comparison.get("clinical_advisory"):
            conclusion_parts.append(str(comparison["clinical_advisory"]))

    review = report.get("clinician_review") or {}
    diagnostic_id = "ecg-guardian-diagnostic-report"
    diagnostic_resource: Dict[str, Any] = {
        "resourceType": "DiagnosticReport",
        "id": diagnostic_id,
        "status": "preliminary" if review.get("status") == "PENDING_REVIEW" else "final",
        "code": {
            "coding": [{"system": LOCAL_CODE_SYSTEM, "code": "ecg-analysis-report"}],
            "text": report.get("report_title") or "AI ECG screening report",
        },
        "issued": report.get("generated_at") or _now(),
        "conclusion": " | ".join(conclusion_parts) if conclusion_parts else "No AI or machine finding was produced.",
        "note": [{"text": report.get("disclaimer", "")}],
    }
    if patient_ref:
        diagnostic_resource["subject"] = {"reference": patient_ref}
    if measurement_refs:
        diagnostic_resource["result"] = [{"reference": ref} for ref in dict.fromkeys(measurement_refs)]
    if ai_ref:
        diagnostic_resource["presentedForm"] = [
            {
                "contentType": "text/plain",
                "title": "AI ECG analysis provenance",
                "data": _b64(
                    f"model={ai.get('model_id') or ai.get('model_version')}; "
                    f"calibration_applied={ai.get('calibration_applied')}; "
                    f"abstain={ai.get('abstain')}"
                ),
            }
        ]

    entries.append({"fullUrl": f"DiagnosticReport/{diagnostic_id}", "resource": diagnostic_resource})

    tags.append({"system": "urn:ecg-guardian:terminology", "code": "interval-codes-require-validation"})

    return {
        "resourceType": "Bundle",
        "id": bundle_id,
        "type": "collection",
        "timestamp": _now(),
        "meta": {"lastUpdated": _now(), "tag": tags},
        "entry": entries,
        "extension": [
            {
                "url": "urn:ecg-guardian:terminology-validation-required",
                "valueString": "; ".join(f"{field}: {reason}" for field, reason in TERMINOLOGY_VALIDATION_REQUIRED),
            },
            {
                "url": "urn:ecg-guardian:regulatory-status",
                "valueString": "Investigational research software. Not a certified diagnostic device.",
            },
        ],
    }


def _b64(text: str) -> str:
    import base64

    return base64.b64encode(text.encode("utf-8")).decode("ascii")
