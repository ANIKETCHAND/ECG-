"""
Hospital Alert & Escalation Engine.
Phase 23:
- Distinguishes AI-generated alerts, clinician-defined alerts, and technical system alerts.
- Enforces clinical safety: never declares an emergency without rule-based justification.
- Supports poor signal quality, significant machine disagreements, and critical rate thresholds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class AlertType(str, Enum):
    AI_GENERATED = "AI_GENERATED"
    CLINICIAN_DEFINED = "CLINICIAN_DEFINED"
    SYSTEM_TECHNICAL = "SYSTEM_TECHNICAL"
    MEDICATION_SAFETY = "MEDICATION_SAFETY"


class AlertSeverity(str, Enum):
    CRITICAL_EMERGENCY = "CRITICAL_EMERGENCY"
    URGENT_REVIEW = "URGENT_REVIEW"
    WARNING = "WARNING"
    INFORMATIONAL = "INFORMATIONAL"


@dataclass
class AlertEvent:
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    patient_id: Optional[str]
    record_id: Optional[str]
    triggered_at: str
    requires_acknowledgement: bool
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["alert_type"] = self.alert_type.value
        d["severity"] = self.severity.value
        return d


class AlertEngine:
    """Hospital alert dispatcher tracking active clinical and technical alerts."""

    def __init__(self):
        self._alerts: List[AlertEvent] = []

    def evaluate_ecg_case(
        self,
        record_id: str,
        patient_id: Optional[str],
        signal_quality: str,
        ai_finding: str,
        heart_rate: Optional[float],
        machine_disagreement: bool = False,
    ) -> List[AlertEvent]:
        events: List[AlertEvent] = []
        now = datetime.now().isoformat()

        # 1. Technical Unusable Signal Alert
        if signal_quality == "UNUSABLE":
            events.append(
                AlertEvent(
                    alert_id=f"ALT-TECH-{record_id[:8]}",
                    alert_type=AlertType.SYSTEM_TECHNICAL,
                    severity=AlertSeverity.WARNING,
                    title="Technical Quality Failure: Signal Unusable",
                    message="Pre-inference quality gate halted analysis. Check electrode leads and repeat acquisition.",
                    patient_id=patient_id,
                    record_id=record_id,
                    triggered_at=now,
                    requires_acknowledgement=False,
                )
            )

        # 2. Significant AI vs Machine Discrepancy Alert
        if machine_disagreement:
            events.append(
                AlertEvent(
                    alert_id=f"ALT-DISC-{record_id[:8]}",
                    alert_type=AlertType.AI_GENERATED,
                    severity=AlertSeverity.URGENT_REVIEW,
                    title="Machine vs. AI Clinical Discrepancy",
                    message="AI and printed machine interpretation conflict. Neither is declared correct; doctor review required.",
                    patient_id=patient_id,
                    record_id=record_id,
                    triggered_at=now,
                    requires_acknowledgement=True,
                )
            )

        # 3. Severe Hemodynamic Rate Thresholds
        if heart_rate is not None:
            if heart_rate >= 150.0:
                events.append(
                    AlertEvent(
                        alert_id=f"ALT-TACHY-{record_id[:8]}",
                        alert_type=AlertType.CLINICIAN_DEFINED,
                        severity=AlertSeverity.CRITICAL_EMERGENCY,
                        title="Severe Tachycardia Alert (HR >= 150 bpm)",
                        message=f"Measured ventricular rate of {heart_rate:.0f} bpm meets critical hospital tachycardia criteria.",
                        patient_id=patient_id,
                        record_id=record_id,
                        triggered_at=now,
                        requires_acknowledgement=True,
                    )
                )
            elif heart_rate <= 38.0:
                events.append(
                    AlertEvent(
                        alert_id=f"ALT-BRADY-{record_id[:8]}",
                        alert_type=AlertType.CLINICIAN_DEFINED,
                        severity=AlertSeverity.CRITICAL_EMERGENCY,
                        title="Severe Bradycardia Alert (HR <= 38 bpm)",
                        message=f"Measured rate of {heart_rate:.0f} bpm meets critical hospital bradycardia criteria.",
                        patient_id=patient_id,
                        record_id=record_id,
                        triggered_at=now,
                        requires_acknowledgement=True,
                    )
                )

        self._alerts.extend(events)
        return events

    def list_active_alerts(self) -> List[AlertEvent]:
        return [a for a in self._alerts if not a.acknowledged]

    def get_active_alerts(self) -> List[AlertEvent]:
        return self.list_active_alerts()

    def get_all_alerts(self) -> List[AlertEvent]:
        return list(self._alerts)

    def acknowledge_alert(
        self,
        alert_id: str,
        clinician_id: Optional[str] = None,
        acknowledged_by: Optional[str] = None,
    ) -> bool:
        actor = acknowledged_by or clinician_id or "clinician"
        for a in self._alerts:
            if a.alert_id == alert_id:
                a.acknowledged = True
                a.acknowledged_by = actor
                a.acknowledged_at = datetime.now().isoformat()
                return True
        return False


GLOBAL_ALERT_ENGINE = AlertEngine()

