"""
System Health & Integrity Diagnostic Script
============================================

Performs pre-flight diagnostic validation of clinical subsystems:
1. Production model artifact presence and validity
2. Database connectivity and schema initialization
3. Cryptographic audit trail chain integrity
4. Signal quality safety gate responsiveness

Run as part of container health checks or CI/CD deployment gates.
"""

import sys
from pathlib import Path

# Add src to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    # 1. Model Artifact Verification
    from inference.inference_engine import ACTIVE_MODEL_ID, get_model_artifacts
    clf, scaler, meta = get_model_artifacts(BASE_DIR / "models")
    print(f"[PASS] Model Artifacts OK: ID={ACTIVE_MODEL_ID}, Classes={meta.get('classes', [])}")

    # 2. Clinical Database Verification
    from database.db_manager import DB_MANAGER
    patients = DB_MANAGER.list_patients(limit=1)
    print("[PASS] Clinical Database OK: Connected and responsive.")

    # 3. Cryptographic Audit Trail Verification
    from audit.audit_logger import AUDIT_LOGGER
    is_valid, issues = AUDIT_LOGGER.verify_chain_integrity()
    if not is_valid:
        print(f"[FAIL] Audit Trail Integrity FAILED: {issues}", file=sys.stderr)
        sys.exit(1)
    print("[PASS] Audit Trail OK: Cryptographic hash chain verified.")

    # 4. Signal Quality Gate Verification
    from safety.signal_quality_gate import QualityCategory, evaluate_signal_quality_gate
    import numpy as np
    dummy_signal = np.sin(np.linspace(0, 10, 1000))
    res = evaluate_signal_quality_gate(dummy_signal, 360.0)
    print(f"[PASS] Signal Quality Gate OK: Gatekeeper evaluated ({res.category.value}).")

    print("\n[SUCCESS] ALL CLINICAL & REGULATORY SUBSYSTEMS OPERATIONAL (PASS).")
    sys.exit(0)

except Exception as exc:
    print(f"[FAIL] Subsystem Healthcheck Failure: {exc}", file=sys.stderr)
    sys.exit(1)
