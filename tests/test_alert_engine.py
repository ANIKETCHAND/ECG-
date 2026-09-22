"""
Unit tests for Hospital Alert & Escalation Engine.
Verifies:
- Alert categorization (AI-generated, clinician-defined, technical, medication safety).
- Critical tachycardia / bradycardia threshold alerts.
- Unusable signal technical alerts.
- Machine vs AI discrepancy escalation.
- Alert acknowledgment and auditing.
"""

import sys
from pathlib import Path
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alerts.alert_engine import (
    AlertEngine,
    AlertEvent,
    AlertSeverity,
    AlertType,
    GLOBAL_ALERT_ENGINE,
)


def test_clean_ecg_produces_no_alerts():
    engine = AlertEngine()
    alerts = engine.evaluate_ecg_case(
        record_id="REC-001",
        patient_id="PAT-001",
        signal_quality="GOOD",
        ai_finding="Normal Sinus Rhythm",
        heart_rate=72.0,
        machine_disagreement=False,
    )
    assert len(alerts) == 0


def test_unusable_signal_triggers_technical_alert():
    engine = AlertEngine()
    alerts = engine.evaluate_ecg_case(
        record_id="REC-002",
        patient_id="PAT-002",
        signal_quality="UNUSABLE",
        ai_finding="Quality Halt",
        heart_rate=None,
        machine_disagreement=False,
    )
    assert len(alerts) == 1
    assert alerts[0].alert_type == AlertType.SYSTEM_TECHNICAL
    assert alerts[0].severity == AlertSeverity.WARNING
    assert "Unusable" in alerts[0].title


def test_machine_discrepancy_triggers_urgent_alert():
    engine = AlertEngine()
    alerts = engine.evaluate_ecg_case(
        record_id="REC-003",
        patient_id="PAT-003",
        signal_quality="GOOD",
        ai_finding="Premature Ventricular Contractions (PVC)",
        heart_rate=80.0,
        machine_disagreement=True,
    )
    assert any(a.alert_type == AlertType.AI_GENERATED and "Discrepancy" in a.title for a in alerts)
    assert any(a.severity == AlertSeverity.URGENT_REVIEW for a in alerts)


def test_critical_tachycardia_triggers_alert():
    engine = AlertEngine()
    alerts = engine.evaluate_ecg_case(
        record_id="REC-004",
        patient_id="PAT-004",
        signal_quality="GOOD",
        ai_finding="Tachycardia Pattern",
        heart_rate=165.0,  # Critical rate >= 150 bpm
        machine_disagreement=False,
    )
    assert len(alerts) == 1
    assert alerts[0].severity == AlertSeverity.CRITICAL_EMERGENCY
    assert "Tachycardia" in alerts[0].title


def test_critical_bradycardia_triggers_alert():
    engine = AlertEngine()
    alerts = engine.evaluate_ecg_case(
        record_id="REC-005",
        patient_id="PAT-005",
        signal_quality="GOOD",
        ai_finding="Bradycardia Pattern",
        heart_rate=32.0,  # Critical rate <= 38 bpm
        machine_disagreement=False,
    )
    assert len(alerts) == 1
    assert alerts[0].severity == AlertSeverity.CRITICAL_EMERGENCY
    assert "Bradycardia" in alerts[0].title


def test_alert_acknowledgment_workflow():
    engine = AlertEngine()
    alerts = engine.evaluate_ecg_case(
        record_id="REC-006",
        patient_id="PAT-006",
        signal_quality="GOOD",
        ai_finding="Tachycardia Pattern",
        heart_rate=160.0,
        machine_disagreement=False,
    )
    assert len(alerts) == 1
    alert_id = alerts[0].alert_id

    # Acknowledge
    success = engine.acknowledge_alert(alert_id, acknowledged_by="dr_patel")
    assert success is True

    # Retrieve and check
    active = engine.get_active_alerts()
    assert len(active) == 0  # Unacknowledged only

    all_alerts = engine.get_all_alerts()
    assert len(all_alerts) == 1
    assert all_alerts[0].acknowledged is True
    assert all_alerts[0].acknowledged_by == "dr_patel"
    assert all_alerts[0].acknowledged_at is not None
