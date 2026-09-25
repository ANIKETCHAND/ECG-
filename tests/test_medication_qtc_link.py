"""
Tests for the medication x longitudinal QTc association module.

The module must be useful without overclaiming: it reports a *temporal*
association and refuses to produce a comparison when either side of a medication
start date is unobserved.
"""

from __future__ import annotations

from src.longitudinal.medication_qtc_link import (
    MedicationQTcReport,
    assess_qtc_medication_association,
)


def _history(entries):
    return [
        {"timestamp": timestamp, "qtc_ms": qtc}
        for timestamp, qtc in entries
    ]


def test_qtc_rise_after_high_risk_drug_is_flagged_for_review():
    history = _history(
        [
            ("2026-01-05", 400.0),
            ("2026-02-05", 405.0),
            ("2026-06-05", 448.0),  # after amiodarone start
        ]
    )
    medications = [{"medication_name": "Amiodarone", "start_date": "2026-03-01"}]

    report = assess_qtc_medication_association(history, medications, patient_id="PAT-1")
    assert isinstance(report, MedicationQTcReport)
    assert report.review_required is True
    assert len(report.associations) == 1

    association = report.associations[0]
    assert association.association == "SUPPORTED"
    assert association.qt_prolongation_risk in {"HIGH", "MODERATE"}
    assert association.qtc_delta_ms == 43.0
    assert association.pre_start_recordings == 2
    assert association.post_start_recordings == 1
    assert "Temporal association" in association.note or "temporal association" in association.note


def test_no_rise_after_drug_start_is_not_supported():
    history = _history([("2026-01-05", 400.0), ("2026-06-05", 402.0)])
    medications = [{"medication_name": "Amiodarone", "start_date": "2026-03-01"}]

    report = assess_qtc_medication_association(history, medications)
    association = report.associations[0]
    assert association.association == "NOT_SUPPORTED"
    assert report.review_required is False


def test_rise_after_non_prolonging_drug_points_elsewhere():
    history = _history([("2026-01-05", 400.0), ("2026-06-05", 445.0)])
    medications = [{"medication_name": "Metoprolol", "start_date": "2026-03-01"}]

    report = assess_qtc_medication_association(history, medications)
    association = report.associations[0]
    assert association.qt_prolongation_risk == "NONE"
    assert association.association == "NOT_SUPPORTED"
    assert "another explanation" in association.note
    # A rise is still worth a look, even when the drug is not the obvious cause.
    assert report.review_required is True


def test_missing_recording_on_one_side_yields_insufficient_data():
    history = _history([("2026-06-05", 448.0)])  # nothing before the start
    medications = [{"medication_name": "Amiodarone", "start_date": "2026-03-01"}]

    report = assess_qtc_medication_association(history, medications)
    association = report.associations[0]
    assert association.association == "INSUFFICIENT_DATA"
    assert association.qtc_delta_ms is None
    assert "before" in association.note and "after" in association.note


def test_medication_without_start_date_cannot_be_paired():
    history = _history([("2026-01-05", 400.0), ("2026-06-05", 448.0)])
    medications = [{"medication_name": "Amiodarone"}]

    report = assess_qtc_medication_association(history, medications)
    association = report.associations[0]
    assert association.association == "INSUFFICIENT_DATA"
    assert association.medication_start is None
    assert "start date" in association.note


def test_undated_qtc_values_are_dropped_rather_than_guessed():
    history = [
        {"timestamp": "2026-01-01", "qtc_ms": 400.0},
        {"timestamp": "not-a-date", "qtc_ms": 500.0},
        {"timestamp": "2026-06-01", "qtc_ms": 450.0},
    ]
    medications = [{"medication_name": "Amiodarone", "start_date": "2026-03-01"}]

    report = assess_qtc_medication_association(history, medications)
    # The unparseable row must not contribute to either side.
    assert report.associations[0].pre_start_recordings == 1


def test_nested_measurement_shape_is_accepted():
    history = [
        {"timestamp": "2026-01-05", "ecg_measurements": {"qtc_interval": 400.0}},
        {"timestamp": "2026-06-05", "ecg_measurements": {"qtc_interval": 450.0}},
    ]
    medications = [{"medication_name": "Amiodarone", "start_date": "2026-03-01"}]

    report = assess_qtc_medication_association(history, medications)
    assert report.associations[0].qtc_delta_ms == 50.0


def test_empty_inputs_report_insufficient_data_without_raising():
    report = assess_qtc_medication_association([], [])
    assert report.associations == []
    assert "Insufficient data" in report.summary

    report = assess_qtc_medication_association([], [{"medication_name": "Amiodarone", "start_date": "2026-01-01"}])
    assert "Insufficient data" in report.summary


def test_report_always_carries_the_non_causation_note():
    report = assess_qtc_medication_association(
        _history([("2026-01-05", 400.0), ("2026-06-05", 450.0)]),
        [{"medication_name": "Amiodarone", "start_date": "2026-03-01"}],
    )
    payload = report.to_dict()
    assert "not evidence of causation" in payload["clinical_note"]
    assert payload["review_required"] is True
    assert "Clinician review advised" in payload["summary"]


def test_multiple_medications_are_each_assessed():
    history = _history([("2026-01-05", 400.0), ("2026-06-05", 450.0)])
    medications = [
        {"medication_name": "Amiodarone", "start_date": "2026-03-01"},
        {"medication_name": "Metoprolol", "start_date": "2026-02-01"},
    ]
    report = assess_qtc_medication_association(history, medications)
    assert len(report.associations) == 2
    assert {a.medication for a in report.associations} == {"Amiodarone", "Metoprolol"}


def test_confidence_reflects_recording_support():
    sparse = assess_qtc_medication_association(
        _history([("2026-01-05", 400.0), ("2026-06-05", 450.0)]),
        [{"medication_name": "Amiodarone", "start_date": "2026-03-01"}],
    )
    dense = assess_qtc_medication_association(
        _history(
            [
                ("2026-01-05", 400.0),
                ("2026-02-05", 402.0),
                ("2026-04-05", 430.0),
                ("2026-06-05", 450.0),
            ]
        ),
        [{"medication_name": "Amiodarone", "start_date": "2026-03-01"}],
    )
    assert sparse.associations[0].confidence == "LOW"
    assert dense.associations[0].confidence == "MODERATE"
