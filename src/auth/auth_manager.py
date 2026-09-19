"""
Clinical Authentication & Role-Based Access Control (RBAC)
==========================================================

Implements security and access control following hospital software guidelines:
- Roles: ADMIN, DOCTOR, CARDIOLOGIST, TECHNICIAN, RESEARCHER
- Secure password hashing using PBKDF2-HMAC-SHA256 (100,000 rounds + cryptographically random salt)
- Strict RBAC permission verification
- Session token generation and validation

References:
- CDSCO Medical Devices Rules, 2017 (Data Security)
- ISO 27799 / HIPAA Security Rule (Access Control)
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class UserRole(str, Enum):
    ADMIN = "ADMIN"
    DOCTOR = "DOCTOR"
    CARDIOLOGIST = "CARDIOLOGIST"
    TECHNICIAN = "TECHNICIAN"
    RESEARCHER = "RESEARCHER"


@dataclass
class User:
    user_id: str
    username: str
    password_hash: str
    salt: str
    role: UserRole
    full_name: str
    email: str
    registration_number: Optional[str] = None  # Medical Council Registration
    is_active: bool = True
    created_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["role"] = self.role.value
        # Exclude secrets from export
        d.pop("password_hash", None)
        d.pop("salt", None)
        return d


# Permissions matrix
ROLE_PERMISSIONS: Dict[UserRole, Set[str]] = {
    UserRole.TECHNICIAN: {
        "ecg:upload",
        "ecg:view_status",
        "patient:create",
        "patient:view_basic",
    },
    UserRole.DOCTOR: {
        "ecg:upload",
        "ecg:view_status",
        "ecg:view_waveform",
        "ai:view_analysis",
        "report:view",
        "report:review",
        "report:sign_off",
        "patient:create",
        "patient:view_full",
    },
    UserRole.CARDIOLOGIST: {
        "ecg:upload",
        "ecg:view_status",
        "ecg:view_waveform",
        "ai:view_analysis",
        "report:view",
        "report:review",
        "report:sign_off",
        "patient:create",
        "patient:view_full",
        "validation:view_metrics",
    },
    UserRole.RESEARCHER: {
        "ecg:view_waveform",
        "ai:view_analysis",
        "research:view_deidentified",
        "model:evaluate",
        "validation:view_metrics",
    },
    UserRole.ADMIN: {
        "user:manage",
        "system:configure",
        "audit:view_logs",
        "patient:manage",
        "report:view_all",
        "ecg:upload",
        "ai:view_analysis",
    },
}


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """Hash password using PBKDF2-HMAC-SHA256 with 100,000 iterations."""
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations=100_000,
    )
    return hashed.hex(), salt


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    """Safely verify password using constant-time comparison."""
    computed_hash, _ = hash_password(password, salt)
    return hmac.compare_digest(computed_hash, password_hash)


class AuthManager:
    """Manages hospital users, sessions, and authorization checks."""

    def __init__(self):
        self._users: Dict[str, User] = {}
        self._sessions: Dict[str, Tuple[str, float]] = {}  # token -> (user_id, expiry)
        self._init_default_clinical_users()

    def _init_default_clinical_users(self):
        """Seed initial hospital user accounts with strong hashed credentials."""
        defaults = [
            ("admin", "Admin@Hospital2026", UserRole.ADMIN, "Dr. Rajesh Verma (Chief Medical IT)", "admin@hospital.org", "MCI-ADM-01"),
            ("cardiologist", "Cardio@Hospital2026", UserRole.CARDIOLOGIST, "Dr. Ananya Roy, MD DM (Cardiology)", "ananya.roy@hospital.org", "MCI-CARD-4421"),
            ("doctor", "Doctor@Hospital2026", UserRole.DOCTOR, "Dr. Vikram Patel, MD (Internal Med)", "vikram.patel@hospital.org", "MCI-DOC-8921"),
            ("technician", "Tech@Hospital2026", UserRole.TECHNICIAN, "Aniket Chand (Lead ECG Technologist)", "aniket.chand@hospital.org", "ECG-TECH-104"),
            ("researcher", "Research@Hospital2026", UserRole.RESEARCHER, "Dr. Elena Rostova (Biostatistician)", "elena.r@cardioresearch.org", "RES-BIO-09"),
        ]
        for uname, pwd, role, fname, email, reg_no in defaults:
            p_hash, salt = hash_password(pwd)
            u = User(
                user_id=f"USR-{secrets.token_hex(4).upper()}",
                username=uname,
                password_hash=p_hash,
                salt=salt,
                role=role,
                full_name=fname,
                email=email,
                registration_number=reg_no,
                is_active=True,
                created_at=time.time(),
            )
            self._users[uname.lower()] = u

    def authenticate(self, username: str, password: str) -> Optional[User]:
        """Authenticate user against credential store."""
        user = self._users.get(username.strip().lower())
        if user is None or not user.is_active:
            return None
        if verify_password(password, user.password_hash, user.salt):
            return user
        return None

    def create_session(self, user: User, duration_sec: int = 3600) -> str:
        """Create a secure session token."""
        token = f"sess_{secrets.token_urlsafe(32)}"
        expiry = time.time() + duration_sec
        self._sessions[token] = (user.user_id, expiry)
        return token

    def validate_session(self, token: str) -> Optional[User]:
        """Validate token and return active user."""
        if token not in self._sessions:
            return None
        user_id, expiry = self._sessions[token]
        if time.time() > expiry:
            del self._sessions[token]
            return None
        # Find user by ID
        for user in self._users.values():
            if user.user_id == user_id and user.is_active:
                return user
        return None

    def has_permission(self, user: User, permission: str) -> bool:
        """Check whether a user role has the required permission."""
        perms = ROLE_PERMISSIONS.get(user.role, set())
        return permission in perms

    def get_user_by_username(self, username: str) -> Optional[User]:
        return self._users.get(username.strip().lower())

    def list_users(self) -> List[User]:
        return list(self._users.values())


# Global singleton instance
AUTH_MANAGER = AuthManager()
