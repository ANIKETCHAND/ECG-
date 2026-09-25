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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import Header, HTTPException, Request, status

API_KEY_ENV_VAR = "ECG_API_KEY"
CORS_ENV_VAR = "ECG_CORS_ORIGINS"
MAX_UPLOAD_ENV_VAR = "ECG_MAX_UPLOAD_MB"
MAX_SIGNAL_ENV_VAR = "ECG_MAX_SIGNAL_SAMPLES"

DEFAULT_MAX_UPLOAD_MB = 25
DEFAULT_MAX_SIGNAL_SAMPLES = 5_000_000

# 413 carries different constant names across Starlette versions; the numeric
# value is stable and avoids depending on a deprecated alias.
HTTP_413_TOO_LARGE = 413


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
