"""
Pytest configuration and common fixtures for ECG Guardian test suite.
Provides authenticated clinician and patient tokens to ensure zero-trust security compliance.
"""

import pytest
from src.auth.auth_manager import User, UserRole
from api.security import create_jwt_token


@pytest.fixture
def clinician_user():
    return User(
        user_id="doc_default_apex",
        username="dr_aniket",
        password_hash="",
        salt="",
        role=UserRole.CARDIOLOGIST,
        full_name="Dr. Aniket Chandra, MD",
        email="dr.aniket@apexheart.org",
        hospital_id="HOSP-APEX",
        registration_number="MCI-DOC-99482",
    )


@pytest.fixture
def clinician_token(clinician_user):
    return create_jwt_token(clinician_user)


@pytest.fixture
def auth_headers(clinician_token):
    return {"Authorization": f"Bearer {clinician_token}"}


@pytest.fixture
def patient_user():
    return User(
        user_id="pat_default_test",
        username="raj_verma",
        password_hash="",
        salt="",
        role=UserRole.PATIENT,
        full_name="Raj Verma",
        email="rajverma@gmail.com",
        hospital_id="HOSP-APEX",
        registration_number="MRN-5F2298",
    )


@pytest.fixture
def patient_token(patient_user):
    return create_jwt_token(patient_user)


@pytest.fixture
def patient_auth_headers(patient_token):
    return {"Authorization": f"Bearer {patient_token}"}
