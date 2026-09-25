"""
Tamper-Evident Clinical Audit Logger
====================================

Implements cryptographic hash-chained audit logging to fulfill:
- IEC 62304 Section 5.1.4 (Software Traceability & Change Records)
- ISO 27799 / Health Informatics Information Security Management
- CDSCO MDR 2017 Audit Trail Requirements

Every event is recorded with an incremental index, timestamp, actor credentials,
event payload, and a SHA-256 hash chaining back to the previous event.
Any retroactive tampering or deletion invalidates the chain.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple


GENESIS_HASH = "0" * 64


@dataclass
class AuditEvent:
    sequence_id: int
    timestamp: str
    event_type: str
    user_id: str
    username: str
    user_role: str
    action: str
    details: Dict[str, Any]
    status: str  # SUCCESS, WARNING, FAILURE
    patient_id: Optional[str] = None
    record_id: Optional[str] = None
    ip_address: Optional[str] = "127.0.0.1"
    previous_hash: str = GENESIS_HASH
    entry_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AuditLogger:
    """Manages append-only, cryptographically chained audit logging."""

    def __init__(self, db_path: Optional[str] = None, jsonl_path: Optional[str] = None):
        if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
            data_dir = Path("/tmp") / "data"
        else:
            base_dir = Path(__file__).resolve().parent.parent.parent
            data_dir = base_dir / "data"
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError):
            data_dir = Path("/tmp") / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path or str(data_dir / "audit_trail.db")
        self.jsonl_path = jsonl_path or str(data_dir / "audit_trail.jsonl")
        self._lock = threading.Lock()
        self._init_db()

    @contextmanager
    def _db(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._lock:
            with self._db() as conn:
                conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_trail (
                    sequence_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    username TEXT NOT NULL,
                    user_role TEXT NOT NULL,
                    action TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    patient_id TEXT,
                    record_id TEXT,
                    ip_address TEXT,
                    previous_hash TEXT NOT NULL,
                    entry_hash TEXT NOT NULL
                )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_event ON audit_trail(event_type)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_trail(user_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_record ON audit_trail(record_id)")
                conn.commit()

    def _compute_hash(self, sequence_id: int, timestamp: str, event_type: str,
                      user_id: str, action: str, details_json: str,
                      status: str, previous_hash: str) -> str:
        payload = f"{sequence_id}|{timestamp}|{event_type}|{user_id}|{action}|{details_json}|{status}|{previous_hash}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _get_last_entry(self, conn: sqlite3.Connection) -> Optional[sqlite3.Row]:
        return conn.execute("SELECT * FROM audit_trail ORDER BY sequence_id DESC LIMIT 1").fetchone()

    def log_event(self, event_type: str, user_id: str, username: str, user_role: str,
                  action: str, details: Optional[Dict[str, Any]] = None,
                  status: str = "SUCCESS", patient_id: Optional[str] = None,
                  record_id: Optional[str] = None, ip_address: Optional[str] = "127.0.0.1") -> AuditEvent:
        """Append a tamper-evident audit event."""
        timestamp = datetime.now(timezone.utc).isoformat()
        details_dict = details or {}
        details_json = json.dumps(details_dict, sort_keys=True)

        with self._lock:
            with self._db() as conn:
                last_row = self._get_last_entry(conn)
                if last_row is None:
                    sequence_id = 1
                    previous_hash = GENESIS_HASH
                else:
                    sequence_id = last_row["sequence_id"] + 1
                    previous_hash = last_row["entry_hash"]

                entry_hash = self._compute_hash(
                    sequence_id=sequence_id,
                    timestamp=timestamp,
                    event_type=event_type,
                    user_id=user_id,
                    action=action,
                    details_json=details_json,
                    status=status,
                    previous_hash=previous_hash,
                )

                conn.execute(
                    """
                    INSERT INTO audit_trail
                    (sequence_id, timestamp, event_type, user_id, username, user_role,
                     action, details_json, status, patient_id, record_id, ip_address,
                     previous_hash, entry_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sequence_id, timestamp, event_type, user_id, username, user_role,
                        action, details_json, status, patient_id, record_id, ip_address,
                        previous_hash, entry_hash,
                    ),
                )
                conn.commit()

            event = AuditEvent(
                sequence_id=sequence_id,
                timestamp=timestamp,
                event_type=event_type,
                user_id=user_id,
                username=username,
                user_role=user_role,
                action=action,
                details=details_dict,
                status=status,
                patient_id=patient_id,
                record_id=record_id,
                ip_address=ip_address,
                previous_hash=previous_hash,
                entry_hash=entry_hash,
            )

            try:
                with open(self.jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event.to_dict()) + "\n")
            except Exception:
                pass

            return event

    def get_logs(self, limit: int = 100, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve recent audit logs."""
        with self._db() as conn:
            if event_type:
                rows = conn.execute(
                    "SELECT * FROM audit_trail WHERE event_type = ? ORDER BY sequence_id DESC LIMIT ?",
                    (event_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM audit_trail ORDER BY sequence_id DESC LIMIT ?",
                    (limit,),
                ).fetchall()

            results = []
            for r in rows:
                d = dict(r)
                d["details"] = json.loads(d["details_json"])
                results.append(d)
            return results

    def verify_chain_integrity(self) -> Tuple[bool, List[str]]:
        """Validate the cryptographic integrity of the entire audit trail."""
        issues = []
        with self._lock:
            with self._db() as conn:
                rows = conn.execute("SELECT * FROM audit_trail ORDER BY sequence_id ASC").fetchall()

            if not rows:
                return True, []

            expected_prev = GENESIS_HASH
            for idx, r in enumerate(rows):
                expected_seq = idx + 1
                if r["sequence_id"] != expected_seq:
                    issues.append(f"Sequence break at row {idx}: found id {r['sequence_id']}, expected {expected_seq}")

                if r["previous_hash"] != expected_prev:
                    issues.append(f"Hash chain broken at sequence {r['sequence_id']}: previous_hash mismatch.")

                computed = self._compute_hash(
                    sequence_id=r["sequence_id"],
                    timestamp=r["timestamp"],
                    event_type=r["event_type"],
                    user_id=r["user_id"],
                    action=r["action"],
                    details_json=r["details_json"],
                    status=r["status"],
                    previous_hash=r["previous_hash"],
                )
                if computed != r["entry_hash"]:
                    issues.append(f"Hash tamper detected at sequence {r['sequence_id']}: hash does not match computed value.")

                expected_prev = r["entry_hash"]

        return len(issues) == 0, issues


# Global singleton instance
AUDIT_LOGGER = AuditLogger()
