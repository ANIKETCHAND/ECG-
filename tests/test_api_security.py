"""
Tests for API security controls: key auth, CORS allow-list, and size limits.

The default posture is "open but visibly so": with no key configured the API
serves requests and ``/health`` reports ``auth: DISABLED``, so an unauthenticated
deployment is discoverable rather than silently assumed to be protected.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from api import security as security_module
from api import index as api_index


@pytest.fixture(autouse=True)
def _clear_security_env(monkeypatch):
    for variable in (
        security_module.API_KEY_ENV_VAR,
        security_module.CORS_ENV_VAR,
        security_module.MAX_UPLOAD_ENV_VAR,
        security_module.MAX_SIGNAL_ENV_VAR,
    ):
        monkeypatch.delenv(variable, raising=False)


# ------------------------------------------------------------------- settings
def test_defaults_are_open_but_documented():
    settings = security_module.get_security_settings()
    assert settings.auth_enabled is False
    assert settings.cors_origins == []

    described = settings.describe()
    assert described["auth"] == "DISABLED"
    assert "ECG_API_KEY" in described["auth_note"]
    assert described["cors_origins"] == ["<same-origin only>"]


def test_api_key_enables_auth(monkeypatch):
    monkeypatch.setenv(security_module.API_KEY_ENV_VAR, "s3cret")
    settings = security_module.get_security_settings()
    assert settings.auth_enabled is True
    assert settings.describe()["auth"] == "ENABLED"


def test_blank_api_key_is_treated_as_unset(monkeypatch):
    monkeypatch.setenv(security_module.API_KEY_ENV_VAR, "   ")
    assert security_module.get_security_settings().auth_enabled is False


def test_cors_origins_are_parsed_from_a_list(monkeypatch):
    monkeypatch.setenv(security_module.CORS_ENV_VAR, "https://a.example, https://b.example ,")
    assert security_module.get_security_settings().cors_origins == [
        "https://a.example",
        "https://b.example",
    ]


def test_malformed_numeric_settings_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv(security_module.MAX_UPLOAD_ENV_VAR, "not-a-number")
    monkeypatch.setenv(security_module.MAX_SIGNAL_ENV_VAR, "")
    settings = security_module.get_security_settings()
    assert settings.max_upload_bytes == security_module.DEFAULT_MAX_UPLOAD_MB * 1024 * 1024
    assert settings.max_signal_samples == security_module.DEFAULT_MAX_SIGNAL_SAMPLES


# ------------------------------------------------------------------- key guard
def test_require_api_key_passes_when_auth_disabled():
    assert security_module.require_api_key(None) is None


def test_require_api_key_rejects_missing_header(monkeypatch):
    monkeypatch.setenv(security_module.API_KEY_ENV_VAR, "s3cret")
    with pytest.raises(HTTPException) as exc:
        security_module.require_api_key(None)
    assert exc.value.status_code == 401


def test_require_api_key_rejects_wrong_key(monkeypatch):
    monkeypatch.setenv(security_module.API_KEY_ENV_VAR, "s3cret")
    with pytest.raises(HTTPException) as exc:
        security_module.require_api_key("wrong")
    assert exc.value.status_code == 401


def test_require_api_key_accepts_correct_key(monkeypatch):
    monkeypatch.setenv(security_module.API_KEY_ENV_VAR, "s3cret")
    assert security_module.require_api_key("s3cret") is None


# ------------------------------------------------------------------- limits
def test_signal_limit_accepts_within_bound(monkeypatch):
    monkeypatch.setenv(security_module.MAX_SIGNAL_ENV_VAR, "1000")
    assert security_module.enforce_signal_limits(999) is None


def test_signal_limit_rejects_oversized_payload(monkeypatch):
    monkeypatch.setenv(security_module.MAX_SIGNAL_ENV_VAR, "1000")
    with pytest.raises(HTTPException) as exc:
        security_module.enforce_signal_limits(1001)
    assert exc.value.status_code == 413
    assert security_module.MAX_SIGNAL_ENV_VAR in exc.value.detail


def test_upload_limit_rejects_oversized_body(monkeypatch):
    monkeypatch.setenv(security_module.MAX_UPLOAD_ENV_VAR, "1")
    with pytest.raises(HTTPException) as exc:
        security_module.enforce_upload_limit_bytes(2 * 1024 * 1024)
    assert exc.value.status_code == 413


# --------------------------------------------------------------- live API tests
def test_health_reports_security_posture():
    client = TestClient(api_index.app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["security"]["auth"] in {"ENABLED", "DISABLED"}


def test_protected_endpoint_is_open_when_no_key_configured():
    client = TestClient(api_index.app)
    response = client.post(
        "/api/analyze",
        json={"signal": [0.0] * 500, "fs": 360.0, "lead": "II"},
    )
    # 200 with the safety-gate verdict, i.e. the request was processed.
    assert response.status_code == 200


def test_protected_endpoint_requires_key_when_configured(monkeypatch):
    monkeypatch.setenv(security_module.API_KEY_ENV_VAR, "s3cret")

    # The dependency reads settings at request time, so no reload is needed.
    client = TestClient(api_index.app)
    payload = {"signal": [0.0] * 500, "fs": 360.0, "lead": "II"}

    denied = client.post("/api/analyze", json=payload)
    assert denied.status_code == 401

    allowed = client.post("/api/analyze", json=payload, headers={"X-API-Key": "s3cret"})
    assert allowed.status_code == 200
