"""
Unit Tests for Authentication and RBAC Manager
"""

import time
import pytest
from src.auth.auth_manager import (
    AUTH_MANAGER,
    AuthManager,
    UserRole,
    hash_password,
    verify_password,
)


def test_password_hashing_and_verification():
    pwd = "SecureDoctorPass#2026"
    p_hash, salt = hash_password(pwd)
    assert len(p_hash) == 64
    assert len(salt) == 32
    assert verify_password(pwd, p_hash, salt)
    assert not verify_password("WrongPassword", p_hash, salt)


def test_auth_manager_default_users():
    mgr = AuthManager()
    doc = mgr.authenticate("doctor", "Doctor@Hospital2026")
    assert doc is not None
    assert doc.role == UserRole.DOCTOR
    assert "MCI-DOC" in doc.registration_number

    cardio = mgr.authenticate("cardiologist", "Cardio@Hospital2026")
    assert cardio is not None
    assert cardio.role == UserRole.CARDIOLOGIST

    bad = mgr.authenticate("doctor", "BadPassword")
    assert bad is None


def test_auth_sessions():
    mgr = AuthManager()
    user = mgr.get_user_by_username("cardiologist")
    assert user is not None

    token = mgr.create_session(user, duration_sec=2)
    assert token.startswith("sess_")
    validated_user = mgr.validate_session(token)
    assert validated_user is not None
    assert validated_user.username == "cardiologist"

    # Expired session
    time.sleep(2.1)
    expired_user = mgr.validate_session(token)
    assert expired_user is None


def test_rbac_permissions():
    mgr = AuthManager()
    tech = mgr.get_user_by_username("technician")
    doctor = mgr.get_user_by_username("doctor")
    researcher = mgr.get_user_by_username("researcher")
    admin = mgr.get_user_by_username("admin")

    # Technician can upload but cannot sign off reports
    assert mgr.has_permission(tech, "ecg:upload")
    assert not mgr.has_permission(tech, "report:sign_off")

    # Doctor can view waveform and sign off
    assert mgr.has_permission(doctor, "ecg:view_waveform")
    assert mgr.has_permission(doctor, "report:sign_off")

    # Researcher can view de-identified data but cannot sign off
    assert mgr.has_permission(researcher, "research:view_deidentified")
    assert not mgr.has_permission(researcher, "report:sign_off")

    # Admin can manage users and view audit logs
    assert mgr.has_permission(admin, "user:manage")
    assert mgr.has_permission(admin, "audit:view_logs")
