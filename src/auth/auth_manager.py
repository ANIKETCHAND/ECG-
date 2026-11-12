"""
Clinical Authentication & Role-Based Access Control (RBAC)
==========================================================

Implements security and access control following hospital software guidelines:
- Multi-Tenant Isolation with hospital_id linking
- Roles: HOSPITAL_ADMIN, CARDIOLOGIST, DOCTOR, ECG_TECHNICIAN, NURSE, RESEARCHER, PATIENT
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
    HOSPITAL_ADMIN = "HOSPITAL_ADMIN"
    CARDIOLOGIST = "CARDIOLOGIST"
    DOCTOR = "DOCTOR"
    ECG_TECHNICIAN = "ECG_TECHNICIAN"
    NURSE = "NURSE"
    RESEARCHER = "RESEARCHER"
    PATIENT = "PATIENT"

    # Backward compatibility aliases
    ADMIN = "ADMIN"
    TECHNICIAN = "TECHNICIAN"


@dataclass
class User:
    user_id: str
    username: str
    password_hash: str
    salt: str
    role: UserRole
    full_name: str
    email: str
    hospital_id: str = "HOSP-APEX"
    registration_number: Optional[str] = None  # Medical Council Registration
    picture: Optional[str] = None
    is_active: bool = True
    created_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["role"] = self.role.value
        d.pop("password_hash", None)
        d.pop("salt", None)
        return d


# Granular RBAC Permissions matrix
ROLE_PERMISSIONS: Dict[UserRole, Set[str]] = {
    UserRole.HOSPITAL_ADMIN: {
        "user:manage",
        "hospital:manage",
        "doctor:manage",
        "staff:manage",
        "system:configure",
        "audit:view_logs",
        "patient:manage",
        "patient:create",
        "patient:view_full",
        "report:view_all",
        "ecg:upload",
        "ai:view_analysis",
        "ecg:view_worklist",
    },
    UserRole.ADMIN: {
        "user:manage",
        "hospital:manage",
        "doctor:manage",
        "staff:manage",
        "system:configure",
        "audit:view_logs",
        "patient:manage",
        "patient:create",
        "patient:view_full",
        "report:view_all",
        "ecg:upload",
        "ai:view_analysis",
        "ecg:view_worklist",
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
        "treatment:authorize",
        "ecg:view_worklist",
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
        "treatment:authorize",
        "ecg:view_worklist",
    },
    UserRole.ECG_TECHNICIAN: {
        "ecg:upload",
        "ecg:view_status",
        "ecg:view_waveform",
        "patient:create",
        "patient:view_basic",
        "ecg:view_worklist",
    },
    UserRole.TECHNICIAN: {
        "ecg:upload",
        "ecg:view_status",
        "ecg:view_waveform",
        "patient:create",
        "patient:view_basic",
        "ecg:view_worklist",
    },
    UserRole.NURSE: {
        "patient:create",
        "patient:view_full",
        "ecg:upload",
        "ecg:view_status",
        "report:view",
        "ecg:view_worklist",
    },
    UserRole.RESEARCHER: {
        "ecg:view_waveform",
        "ai:view_analysis",
        "research:view_deidentified",
        "model:evaluate",
        "validation:view_metrics",
    },
    UserRole.PATIENT: {
        "patient:view_self",
        "report:view_self",
        "ecg:upload_self",
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
    """Manages hospital users, sessions, and multi-tenant authorization checks."""

    def __init__(self):
        self._users: Dict[str, User] = {}
        self._sessions: Dict[str, Tuple[str, float]] = {}  # token -> (user_id, expiry)
        self._init_default_clinical_users()

    def _init_default_clinical_users(self):
        """Seed initial hospital user accounts with strong hashed credentials."""
        defaults = [
            ("aniket", "Doctor@Hospital2026", UserRole.DOCTOR, "Dr. Aniket Chandra, MD", "dr.aniket@hospital.org", "MD/34667", "HOSP-APEX"),
            ("raj", "Patient@Hospital2026", UserRole.PATIENT, "Raj Verma", "rajverma@gmail.com", "MRN-5F2298", "HOSP-APEX"),
            ("admin", "Admin@Hospital2026", UserRole.ADMIN, "Dr. Rajesh Verma (Chief Medical IT)", "admin@hospital.org", "MCI-ADM-01", "HOSP-APEX"),
            ("cardiologist", "Cardio@Hospital2026", UserRole.CARDIOLOGIST, "Dr. Ananya Roy, MD DM (Cardiology)", "ananya.roy@hospital.org", "MCI-CARD-4421", "HOSP-APEX"),
            ("doctor", "Doctor@Hospital2026", UserRole.DOCTOR, "Dr. Vikram Patel, MD (Internal Med)", "vikram.patel@hospital.org", "MCI-DOC-8921", "HOSP-APEX"),
            ("technician", "Tech@Hospital2026", UserRole.TECHNICIAN, "Aniket Chand (Lead ECG Technologist)", "aniket.chand@hospital.org", "ECG-TECH-104", "HOSP-APEX"),
            ("nurse", "Nurse@Hospital2026", UserRole.NURSE, "Nurse Priya Nair (CCU Triage)", "priya.nair@hospital.org", "NURSE-IND-502", "HOSP-APEX"),
            ("researcher", "Research@Hospital2026", UserRole.RESEARCHER, "Dr. Elena Rostova (Biostatistician)", "elena.r@cardioresearch.org", "RES-BIO-09", "HOSP-APEX"),
            ("patient", "Patient@Hospital2026", UserRole.PATIENT, "Ramesh Sharma (Patient Portal)", "ramesh.sharma@example.com", None, "HOSP-APEX"),
        ]
        for uname, pwd, role, fname, email, reg_no, hosp_id in defaults:
            p_hash, salt = hash_password(pwd)
            u = User(
                user_id=f"USR-{secrets.token_hex(4).upper()}",
                username=uname,
                password_hash=p_hash,
                salt=salt,
                role=role,
                full_name=fname,
                email=email,
                hospital_id=hosp_id,
                registration_number=reg_no,
                is_active=True,
                created_at=time.time(),
            )
            self._users[uname.lower()] = u
            self._persist_user_safe(u)

    def _persist_user_safe(self, u: User) -> None:
        try:
            from src.database.db_manager import DB_MANAGER
            DB_MANAGER.save_user(
                user_id=u.user_id,
                username=u.username,
                password_hash=u.password_hash,
                salt=u.salt,
                role=u.role.value if isinstance(u.role, UserRole) else str(u.role),
                full_name=u.full_name,
                email=u.email,
                hospital_id=u.hospital_id,
                registration_number=u.registration_number,
                picture=u.picture,
                is_active=u.is_active,
                created_at=u.created_at,
            )
        except Exception:
            pass

    def _persist_session_safe(self, token: str, user_id: str, expiry: float) -> None:
        try:
            from src.database.db_manager import DB_MANAGER
            DB_MANAGER.save_session(token, user_id, expiry)
        except Exception:
            pass

    def register_user(
        self,
        username: str,
        password: str,
        role: UserRole,
        full_name: str,
        email: str,
        hospital_id: str = "HOSP-APEX",
        registration_number: Optional[str] = None,
    ) -> User:
        p_hash, salt = hash_password(password)
        u = User(
            user_id=f"USR-{secrets.token_hex(4).upper()}",
            username=username.strip().lower(),
            password_hash=p_hash,
            salt=salt,
            role=role,
            full_name=full_name,
            email=email,
            hospital_id=hospital_id,
            registration_number=registration_number,
            is_active=True,
            created_at=time.time(),
        )
        self._users[username.strip().lower()] = u
        self._persist_user_safe(u)
        return u

    def authenticate(self, username: str, password: str) -> Optional[User]:
        """Authenticate user against credential store by username or email."""
        ident = username.strip().lower()
        user = self._users.get(ident)
        if user is None:
            for u in self._users.values():
                if u.email and u.email.strip().lower() == ident:
                    user = u
                    break
        if user is None or not user.is_active:
            return None
        if verify_password(password, user.password_hash, user.salt):
            return user
        return None

    def create_session(self, user: User, duration_sec: int = 3600) -> str:
        """Create a secure session token with SQLite persistence."""
        token = f"sess_{secrets.token_urlsafe(32)}"
        expiry = time.time() + duration_sec
        self._sessions[token] = (user.user_id, expiry)
        self._persist_session_safe(token, user.user_id, expiry)
        return token

    def validate_session(self, token: str) -> Optional[User]:
        """Validate token and return active user, checking memory and SQLite."""
        if token in self._sessions:
            user_id, expiry = self._sessions[token]
            if time.time() > expiry:
                del self._sessions[token]
                self.invalidate_session(token)
                return None
            for user in self._users.values():
                if user.user_id == user_id and user.is_active:
                    return user

        # Check SQLite persistence
        try:
            from src.database.db_manager import DB_MANAGER
            persisted = DB_MANAGER.get_session(token)
            if persisted:
                user_id, expiry = persisted
                if time.time() > expiry:
                    DB_MANAGER.delete_session(token)
                    return None
                for user in self._users.values():
                    if user.user_id == user_id and user.is_active:
                        self._sessions[token] = (user_id, expiry)
                        return user
                p_users = DB_MANAGER.list_persisted_users()
                for pu in p_users:
                    if pu["user_id"] == user_id and pu.get("is_active", 1):
                        role_key = pu["role"]
                        role_enum = UserRole.__members__.get(role_key, UserRole.CARDIOLOGIST)
                        user = User(
                            user_id=pu["user_id"],
                            username=pu["username"],
                            password_hash=pu["password_hash"],
                            salt=pu["salt"],
                            role=role_enum,
                            full_name=pu["full_name"],
                            email=pu["email"],
                            hospital_id=pu.get("hospital_id", "HOSP-APEX"),
                            registration_number=pu.get("registration_number"),
                            picture=pu.get("picture"),
                            is_active=bool(pu.get("is_active", 1)),
                            created_at=float(pu.get("created_at", time.time())),
                        )
                        self._users[user.username] = user
                        self._sessions[token] = (user_id, expiry)
                        return user
        except Exception:
            pass

        return None

    def invalidate_session(self, token: str) -> None:
        """Invalidate session in memory and SQLite."""
        if token in self._sessions:
            del self._sessions[token]
        try:
            from src.database.db_manager import DB_MANAGER
            DB_MANAGER.delete_session(token)
        except Exception:
            pass

    def has_permission(self, user: User, permission: str) -> bool:
        """Check whether a user role has the required permission."""
        perms = ROLE_PERMISSIONS.get(user.role, set())
        return permission in perms

    def get_user_by_username(self, username: str) -> Optional[User]:
        return self._users.get(username.strip().lower())

    def get_user(self, user_id: str) -> Optional[User]:
        """Find a user by user_id or username."""
        clean_id = (user_id or "").strip().lower()
        for u in self._users.values():
            if u.user_id.lower() == clean_id or u.username.lower() == clean_id:
                return u
        return None

    def authenticate_google(
        self,
        email: str,
        full_name: str,
        google_id: Optional[str] = None,
        picture: Optional[str] = None,
        role: UserRole = UserRole.CARDIOLOGIST,
        hospital_id: str = "HOSP-APEX",
    ) -> Tuple[User, str]:
        """Authenticate or auto-provision a clinical user logging in with Google."""
        clean_email = email.strip().lower()
        existing_user = None
        for u in self._users.values():
            if u.email.strip().lower() == clean_email:
                existing_user = u
                break

        if existing_user:
            if full_name and full_name.strip():
                existing_user.full_name = full_name.strip()
            if picture:
                existing_user.picture = picture
            user = existing_user
        else:
            username = clean_email.split("@")[0]
            base_username = username
            counter = 1
            while username in self._users:
                username = f"{base_username}_{counter}"
                counter += 1

            dummy_pwd = secrets.token_hex(16)
            p_hash, salt = hash_password(dummy_pwd)
            user = User(
                user_id=f"USR-G-{secrets.token_hex(4).upper()}",
                username=username,
                password_hash=p_hash,
                salt=salt,
                role=role,
                full_name=full_name or username,
                email=clean_email,
                hospital_id=hospital_id,
                registration_number=f"MCI-G-{secrets.token_hex(3).upper()}",
                picture=picture,
                is_active=True,
                created_at=time.time(),
            )
            self._users[username] = user

        self._persist_user_safe(user)
        token = self.create_session(user, duration_sec=86400)
        return user, token

    def list_users(self, hospital_id: Optional[str] = None) -> List[User]:
        users = list(self._users.values())
        if hospital_id:
            users = [u for u in users if u.hospital_id == hospital_id]
        return users


# Global singleton instance
AUTH_MANAGER = AuthManager()

# Ensure module unification between 'auth.auth_manager' and 'src.auth.auth_manager'
import sys as _sys
if "auth.auth_manager" not in _sys.modules:
    _sys.modules["auth.auth_manager"] = _sys.modules[__name__]
if "src.auth.auth_manager" not in _sys.modules:
    _sys.modules["src.auth.auth_manager"] = _sys.modules[__name__]
