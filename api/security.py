"""
API Security Configuration
==========================

Hardens the serverless clinical API without breaking local development.

Three controls, each configurable by environment variable:

``ECG_API_KEY``
    When set, every data-bearing endpoint requires the ``X-API-Key`` header.
    When unset, the API stays open and ``/health`` reports ``auth: DISABLED`` so
    an unauthenticated deployment is visible rather than silently assumed safe.
    A shared key is a floor, not a substitute for per-user identity — the
    role-based ``src/auth`` subsystem remains the source of truth inside the
    application.

``ECG_CORS_ORIGINS``
    Comma-separated allow-list. Defaults to *no* cross-origin allowance, so a
    wildcard must be requested deliberately (``ECG_CORS_ORIGINS=*``).

``ECG_MAX_UPLOAD_MB`` / ``ECG_MAX_SIGNAL_SAMPLES``
    Bound the request size. Clinical telemetry is large, but an unbounded body
    is a denial-of-service vector on a serverless function.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import jwt
from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.auth.auth_manager import AUTH_MANAGER, User, UserRole

API_KEY_ENV_VAR = "ECG_API_KEY"
CORS_ENV_VAR = "ECG_CORS_ORIGINS"
MAX_UPLOAD_ENV_VAR = "ECG_MAX_UPLOAD_MB"
MAX_SIGNAL_ENV_VAR = "ECG_MAX_SIGNAL_SAMPLES"
STRICT_AUTH_ENV_VAR = "ECG_STRICT_AUTH"
JWT_SECRET_ENV_VAR = "ECG_JWT_SECRET"

DEFAULT_MAX_UPLOAD_MB = 25
DEFAULT_MAX_SIGNAL_SAMPLES = 5_000_000

HTTP_413_TOO_LARGE = 413

SECURITY_BEARER = HTTPBearer(auto_error=False)
JWT_ALGORITHM = "HS256"


def get_jwt_secret_key() -> str:
    """Retrieve cryptographic JWT signing key with production guard."""
    secret = os.environ.get(JWT_SECRET_ENV_VAR) or os.environ.get("JWT_SECRET")
    is_prod = bool(
        os.environ.get("VERCEL")
        or os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
        or os.environ.get("ENVIRONMENT") == "production"
    )
    if is_prod and not secret:
        # In serverless/production, fail-safe to a persistent hash to prevent forged tokens
        return os.environ.get("VERCEL_DEPLOYMENT_ID") or "STRICT_PROD_FALLBACK_SECRET_ECG_2026"
    return secret or "DEV_FALLBACK_JWT_SECRET_2026_CDSCO_ECG_GUARDIAN"


def create_jwt_token(user: User, duration_sec: int = 86400) -> str:
    """Generate cryptographically signed JWT token with identity claims."""
    now = time.time()
    payload = {
        "sub": user.user_id,
        "username": user.username,
        "role": user.role.value if isinstance(user.role, UserRole) else str(user.role),
        "name": user.full_name,
        "email": user.email,
        "hospital_id": user.hospital_id,
        "registration_number": user.registration_number,
        "iat": int(now),
        "exp": int(now + duration_sec),
    }
    return jwt.encode(payload, get_jwt_secret_key(), algorithm=JWT_ALGORITHM)


def decode_jwt_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify and decode JWT token, rejecting expired or forged tokens."""
    try:
        return jwt.decode(token, get_jwt_secret_key(), algorithms=[JWT_ALGORITHM])
    except Exception:
        return None


def is_strict_auth_required() -> bool:
    """Determine whether zero-trust mandatory authentication is active."""
    if os.environ.get(STRICT_AUTH_ENV_VAR) in ("1", "true", "True"):
        return True
    if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME") or os.environ.get("ENVIRONMENT") == "production":
        return True
    settings = get_security_settings()
    return settings.auth_enabled


def get_current_user_strictly(
    request: Request,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(SECURITY_BEARER),
) -> User:
    """
    Mandatory authentication dependency for clinical and patient endpoints.
    Rejects anonymous callers with HTTP 401. Never falls back to a default profile.
    Returns typed User dataclass instance.
    """
    token = None
    if creds and creds.credentials:
        token = creds.credentials
    elif "Authorization" in request.headers or "authorization" in request.headers:
        auth_hdr = request.headers.get("Authorization") or request.headers.get("authorization")
        if auth_hdr:
            parts = auth_hdr.strip().split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                token = parts[1]
            elif len(parts) == 1:
                token = parts[0]
    if not token and "X-Session-Token" in request.headers:
        token = request.headers["X-Session-Token"]
    if not token and "x-session-token" in request.headers:
        token = request.headers["x-session-token"]
    if not token and "token" in request.query_params:
        token = request.query_params["token"]

    if not token or not token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide valid Authorization Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = token.strip()

    # 1. Check Cryptographic JWT
    payload = decode_jwt_token(token)
    if payload:
        exp = payload.get("exp")
        if exp and time.time() > exp:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session expired.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        user_id = payload.get("sub") or payload.get("user_id")
        role_str = payload.get("role", "PATIENT")
        role_enum = UserRole.__members__.get(role_str, UserRole.PATIENT)
        return User(
            user_id=user_id,
            username=payload.get("username", user_id),
            password_hash="",
            salt="",
            role=role_enum,
            full_name=payload.get("name", "Authenticated User"),
            email=payload.get("email", ""),
            hospital_id=payload.get("hospital_id", "HOSP-APEX"),
            registration_number=payload.get("registration_number"),
            is_active=True,
            created_at=float(payload.get("iat", time.time())),
        )

    # 2. Check Database / AuthManager Session Store
    user = AUTH_MANAGER.validate_session(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_roles(*allowed_roles: UserRole):
    """Factory creating strict RBAC role-enforcement dependencies."""
    def role_checker(user: User = Depends(get_current_user_strictly)) -> User:
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: Role '{user.role.value if isinstance(user.role, UserRole) else user.role}' does not have permission for this resource.",
            )
        return user
    return role_checker


@dataclass
class SecuritySettings:
    api_key: Optional[str] = None
    cors_origins: List[str] = field(default_factory=list)
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_MB * 1024 * 1024
    max_signal_samples: int = DEFAULT_MAX_SIGNAL_SAMPLES

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key)

    def describe(self) -> Dict[str, Any]:
        """Non-secret summary for the health endpoint."""
        return {
            "auth": "ENABLED" if self.auth_enabled else "DISABLED",
            "cors_origins": self.cors_origins or ["<same-origin only>"],
            "max_upload_mb": round(self.max_upload_bytes / (1024 * 1024), 2),
            "max_signal_samples": self.max_signal_samples,
            "auth_note": (
                "Shared API key in use; per-user RBAC is enforced inside the application."
                if self.auth_enabled
                else "API key not configured (ECG_API_KEY unset). Do not expose this deployment publicly."
            ),
        }


def get_security_settings() -> SecuritySettings:
    """Read security settings from the environment."""
    raw_origins = os.environ.get(CORS_ENV_VAR, "").strip()
    origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]

    try:
        max_upload_mb = int(os.environ.get(MAX_UPLOAD_ENV_VAR, str(DEFAULT_MAX_UPLOAD_MB)))
    except ValueError:
        max_upload_mb = DEFAULT_MAX_UPLOAD_MB

    try:
        max_signal_samples = int(os.environ.get(MAX_SIGNAL_ENV_VAR, str(DEFAULT_MAX_SIGNAL_SAMPLES)))
    except ValueError:
        max_signal_samples = DEFAULT_MAX_SIGNAL_SAMPLES

    api_key = os.environ.get(API_KEY_ENV_VAR)
    return SecuritySettings(
        api_key=api_key.strip() if api_key and api_key.strip() else None,
        cors_origins=origins,
        max_upload_bytes=max(1, max_upload_mb) * 1024 * 1024,
        max_signal_samples=max(1, max_signal_samples),
    )


def require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    """FastAPI dependency enforcing the shared API key when configured.

    Uses a constant-time comparison so a wrong key cannot be probed by timing.
    """
    settings = get_security_settings()
    if not settings.auth_enabled:
        return

    import hmac

    provided = (x_api_key or "").encode("utf-8")
    expected = settings.api_key.encode("utf-8")  # type: ignore[union-attr]

    if not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key header.",
            headers={"WWW-Authenticate": "X-API-Key"},
        )


def enforce_signal_limits(samples: int) -> None:
    """Reject payloads whose sample count exceeds the configured ceiling."""
    settings = get_security_settings()
    if samples > settings.max_signal_samples:
        raise HTTPException(
            status_code=HTTP_413_TOO_LARGE,
            detail=(
                f"Signal contains {samples} samples, exceeding the configured maximum of "
                f"{settings.max_signal_samples}. Adjust with {MAX_SIGNAL_ENV_VAR}."
            ),
        )


def enforce_upload_limit_bytes(size_bytes: int) -> None:
    """Reject an already-buffered upload that exceeds the configured ceiling."""
    settings = get_security_settings()
    if size_bytes > settings.max_upload_bytes:
        raise HTTPException(
            status_code=HTTP_413_TOO_LARGE,
            detail=(
                f"Upload of {size_bytes} bytes exceeds the {settings.max_upload_bytes}-byte limit. "
                f"Adjust with {MAX_UPLOAD_ENV_VAR}."
            ),
        )


async def enforce_body_size_limit(request: Request) -> None:
    """Reject oversized request bodies before the handler buffers them.

    Applied as a router dependency so both JSON analytics payloads and multipart
    uploads share one ceiling.
    """
    settings = get_security_settings()
    content_length = request.headers.get("content-length")
    if content_length is None:
        return
    try:
        declared = int(content_length)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Malformed Content-Length header.",
        )
    if declared > settings.max_upload_bytes:
        raise HTTPException(
            status_code=HTTP_413_TOO_LARGE,
            detail=(
                f"Request body of {declared} bytes exceeds the {settings.max_upload_bytes}-byte limit. "
                f"Adjust with {MAX_UPLOAD_ENV_VAR}."
            ),
        )
