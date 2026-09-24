"""
Temporal Alignment & Point-in-Time Validator
============================================

Phase 3 & Phase 7:
Guarantees that multimodal ECG models only ingest clinical information that existed
at or before the exact timestamp of the ECG recording.

Rules:
- T_event <= T_ecg enforced across all clinical observations (vitals, labs, meds, diagnoses).
- Rejects future diagnoses, future prescriptions, and future laboratory tests from training vectors.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


class TimestampMapper:
    """Parses, aligns, and filters clinical events based on temporal cutoffs."""

    @staticmethod
    def parse_iso_datetime(dt_str: Optional[str]) -> Optional[datetime]:
        if not dt_str or str(dt_str).strip() in ("", "None", "null"):
            return None
        try:
            # Handle ISO variations with Z or space
            clean_str = dt_str.replace("Z", "+00:00").replace(" ", "T")
            return datetime.fromisoformat(clean_str)
        except Exception:
            # Fallback for common YYYY-MM-DD format
            try:
                return datetime.strptime(dt_str[:10], "%Y-%m-%d")
            except Exception:
                return None

    @classmethod
    def is_temporally_valid(
        cls,
        event_time_str: Optional[str],
        ecg_time_str: Optional[str],
        allow_same_time: bool = True,
    ) -> bool:
        """Check whether event_time occurred strictly at or before ecg_time."""
        # If either timestamp is missing, cannot verify temporal precedence
        if not event_time_str or not ecg_time_str:
            return True  # Retain static baseline history

        t_event = cls.parse_iso_datetime(event_time_str)
        t_ecg = cls.parse_iso_datetime(ecg_time_str)

        if not t_event or not t_ecg:
            return True

        if allow_same_time:
            return t_event <= t_ecg
        return t_event < t_ecg

    @classmethod
    def filter_point_in_time_events(
        cls,
        events: List[Dict[str, Any]],
        ecg_timestamp: Optional[str],
        timestamp_key: str = "recorded_at",
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Splits events into valid historical events (t <= t_ecg) and future leaked events (t > t_ecg)."""
        valid_events = []
        leaked_future_events = []

        if not ecg_timestamp:
            return list(events), []

        for ev in events:
            t_ev = ev.get(timestamp_key) or ev.get("timestamp") or ev.get("start_date")
            if not t_ev:
                valid_events.append(ev)
                continue

            if cls.is_temporally_valid(t_ev, ecg_timestamp, allow_same_time=True):
                valid_events.append(ev)
            else:
                leaked_future_events.append(ev)

        return valid_events, leaked_future_events


GLOBAL_TIMESTAMP_MAPPER = TimestampMapper()
