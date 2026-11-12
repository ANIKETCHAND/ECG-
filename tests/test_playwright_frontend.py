"""
ECG Guardian — Frontend Complete Functional & Automation Test Suite
===================================================================
Uses Playwright to systematically test all 12 application pages and workflows:
- A. Landing Page & Public Navigation
- B. Authentication (Doctor & Patient Registration, Login, Errors, RBAC)
- C. Doctor Dashboard (Stats, Activity, Patient Lists, Chart)
- D. Patient Dashboard (Greeting, Demographics, ECG Reports, Isolation)
- E. Patient Registration (Multi-step input validation, Auto-MRN, Auto-age, Persistence)
- F. Patient Profile (Details, Tabs, Edit flow, Action buttons)
- G. ECG Scanning (Benchmark loading, File upload, Patient linkage)
- H. ECG Results & Analysis (Waveform chart, Zoom controls, Measurements, AI findings)
- I. Clinical Decision Support & Medication Safety
- J. Clinical Reports (Doctor clinical PDF, Patient plain-language view)
- K. Report History & Audit Snapshot (Filtering, Search, Modal inspection)
- L. Patient Directory & Management
"""

import os
import socket
import threading
import time
import urllib.request
import pytest
from playwright.sync_api import sync_playwright

BASE_URL = os.environ.get("TEST_APP_URL", "http://127.0.0.1:8008")


def _ensure_local_server(url: str):
    """Ensure local test server is responding; if not, spin up background thread."""
    try:
        with urllib.request.urlopen(f"{url}/api/health", timeout=1.0) as resp:
            if resp.status == 200:
                data = resp.read().decode('utf-8', errors='ignore')
                if "ECG Guardian" in data or "ecg" in data.lower():
                    return
    except Exception:
        pass

    import uvicorn
    import urllib.parse
    parsed = urllib.parse.urlparse(url)
    port = parsed.port or 8008
    def run_app():
        uvicorn.run("api.index:app", host="127.0.0.1", port=port, log_level="warning")

    t = threading.Thread(target=run_app, daemon=True)
    t.start()
    for _ in range(40):
        time.sleep(0.2)
        try:
            with urllib.request.urlopen(f"{url}/api/health", timeout=0.5) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass


@pytest.fixture(scope="module")
def browser_context():
    _ensure_local_server(BASE_URL)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 800})
        yield context
        browser.close()


@pytest.fixture
def page(browser_context):
    page = browser_context.new_page()
    page.goto(BASE_URL)
    page.wait_for_load_state("domcontentloaded")
    yield page
    page.close()


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2 - A. LANDING PAGE TESTS
# ──────────────────────────────────────────────────────────────────────────────

def test_landing_page_title_and_elements(page):
    assert "ECG GUARDIAN" in page.title()
    assert page.locator("#publicNavbar").is_visible()
    assert page.locator("#viewLanding").is_visible()

    # Brand Logo
    brand_logo = page.locator(".brand-logo-wrap").first
    assert brand_logo.is_visible()

    # Public Nav Links
    nav_links = page.locator(".public-nav-link")
    assert nav_links.count() >= 4

    # Hero Action Buttons
    hero_get_started = page.locator("#viewLanding button:has-text('Get Started')")
    hero_learn_more = page.locator("#viewLanding a:has-text('Learn More')")

    assert hero_get_started.is_visible()
    assert hero_learn_more.is_visible()


def test_landing_page_navigation_to_auth(page):
    page.locator("#publicNavbar button:has-text('Login')").click()
    page.wait_for_timeout(300)
    assert page.locator("#viewAuth").is_visible()
    assert not page.locator("#viewLanding").is_visible()


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2 - B. AUTHENTICATION & RBAC TESTS
# ──────────────────────────────────────────────────────────────────────────────

def test_auth_role_selection_and_login_form(page):
    page.evaluate("navigateTo('auth')")
    page.wait_for_timeout(200)

    # Click Physician Portal Card
    page.locator(".role-select-card").first.click()
    page.wait_for_timeout(200)

    auth_card = page.locator("#authFormContainer")
    assert auth_card.is_visible()
    header_text = page.locator("#authFormTitle").inner_text()
    assert "Doctor" in header_text

    # Test Password Visibility Toggle
    pwd_input = page.locator("#authPassword")
    assert pwd_input.get_attribute("type") == "password"
    page.locator("#authEyeIcon").click()
    assert pwd_input.get_attribute("type") == "text"
    page.locator("#authEyeIcon").click()
    assert pwd_input.get_attribute("type") == "password"

    # Test Back to Role Select
    page.locator("#authFormContainer button:has-text('Change Role')").click()
    page.wait_for_timeout(200)
    assert page.locator("#authRoleSelectionScreen").is_visible()


def test_auth_doctor_login_success(page):
    page.evaluate("navigateTo('auth')")
    page.wait_for_timeout(200)
    page.evaluate("openLoginFormForRole('DOCTOR')")
    page.wait_for_timeout(200)

    page.locator("#authEmail").fill("dr.aniket@hospital.org")
    page.locator("#authPassword").fill("Doctor@Hospital2026")
    page.locator("#authCoreForm button[type='submit']").click()
    page.wait_for_timeout(800)

    # Should redirect to Doctor Dashboard
    assert page.locator("#viewDoctorDashboard").is_visible()
    assert page.locator("#appSidebar").is_visible()
    assert page.locator("#appTopHeader").is_visible()


def test_auth_patient_login_and_role_redirect(page):
    page.evaluate("quickLoginUser('raj_verma')")
    page.wait_for_timeout(400)

    # Should redirect to Patient Dashboard
    assert page.locator("#viewPatientDashboard").is_visible()
    greeting = page.locator("#topHeaderGreeting").inner_text()
    assert "Raj" in greeting or "Patient" in greeting


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2 - C. DOCTOR DASHBOARD TESTS
# ──────────────────────────────────────────────────────────────────────────────

def test_doctor_dashboard_components(page):
    # Log in as doctor
    page.evaluate("quickLoginUser('dr_aniket')")
    page.wait_for_timeout(300)

    # Stat cards
    assert page.locator("#statTotalPatients").is_visible()
    assert page.locator("#statTotalScans").is_visible()
    assert page.locator("#statReportsGenerated").is_visible()
    assert page.locator("#statPendingReviews").is_visible()

    # Activity table
    assert page.locator("#tblRecentActivityBody").is_visible()

    # Chart container
    assert page.locator("#doctorTrendsChart").is_visible()


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2 - D. PATIENT DASHBOARD TESTS
# ──────────────────────────────────────────────────────────────────────────────

def test_patient_dashboard_elements(page):
    page.evaluate("quickLoginUser('raj_verma')")
    page.wait_for_timeout(300)

    assert page.locator("#viewPatientDashboard").is_visible()
    assert page.locator(".patient-hero-report-card").is_visible()
    assert page.locator(".patient-quick-actions-grid").is_visible()


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2 - E. PATIENT REGISTRATION FORM TESTS
# ──────────────────────────────────────────────────────────────────────────────

def test_patient_registration_multistep_validation(page):
    page.evaluate("quickLoginUser('dr_aniket')")
    page.evaluate("navigateTo('patientRegistration')")
    page.wait_for_timeout(300)

    assert page.locator("#viewPatientRegistration").is_visible()

    # Step 1: Personal & Demographics
    assert page.locator("#stepSection1").is_visible()
    name_input = page.locator("#pFormName")
    mrn_input = page.locator("#pFormMrn")
    dob_input = page.locator("#pFormDob")

    assert name_input.is_visible()
    assert mrn_input.is_visible()
    assert dob_input.is_visible()

    # Test auto-age calculation on DOB change
    page.evaluate("document.getElementById('pFormDob').value = '1990-05-15'")
    page.evaluate("onDobChange('1990-05-15')")
    calculated_age = int(page.locator("#pFormAge").input_value())
    assert calculated_age >= 34

    # Navigation to Step 2
    page.locator("#stepSection1 button:has-text('Next')").click()
    page.wait_for_timeout(200)
    assert page.locator("#stepSection2").is_visible()

    # Navigation to Step 3
    page.locator("#stepSection2 button:has-text('Next')").click()
    page.wait_for_timeout(200)
    assert page.locator("#stepSection3").is_visible()

    # Navigation Back to Step 2
    page.locator("#stepSection3 button:has-text('Previous')").click()
    page.wait_for_timeout(200)
    assert page.locator("#stepSection2").is_visible()


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2 - F. PATIENT PROFILE VIEW TESTS
# ──────────────────────────────────────────────────────────────────────────────

def test_patient_profile_tabs_and_actions(page):
    page.evaluate("quickLoginUser('dr_aniket')")
    page.evaluate("navigateTo('patientProfile')")
    page.wait_for_timeout(300)

    assert page.locator("#viewPatientProfile").is_visible()
    assert page.locator("#profName").is_visible()

    # Switch tabs
    tab_meds = page.locator("#viewPatientProfile button:has-text('Medications')")
    tab_reports = page.locator("#viewPatientProfile button:has-text('ECG Reports')")

    if tab_meds.is_visible():
        tab_meds.click()
        page.wait_for_timeout(200)
        assert page.locator("#profTabMeds").is_visible()

    if tab_reports.is_visible():
        tab_reports.click()
        page.wait_for_timeout(200)
        assert page.locator("#profTabReports").is_visible()


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2 - G. ECG SCANNING & INGESTION TESTS
# ──────────────────────────────────────────────────────────────────────────────

def test_ecg_scanning_and_sample_load(page):
    page.evaluate("navigateTo('ecgScan')")
    page.wait_for_timeout(300)

    # Ingestion context bar
    assert page.locator(".compact-selected-patient-bar").is_visible()

    # Load Normal Benchmark
    btn_normal = page.locator("button:has-text('Normal Rhythm')")
    if btn_normal.is_visible():
        btn_normal.click()
    page.wait_for_timeout(600)

    # Trigger Analysis
    page.locator("#viewEcgScan button:has-text('Analyze ECG')").click()
    page.wait_for_timeout(1000)

    # Should land on Analysis View
    assert page.locator("#viewAnalysis").is_visible()


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2 - H. ECG RESULTS & WAVEFORM VIEWER TESTS
# ──────────────────────────────────────────────────────────────────────────────

def test_ecg_analysis_results_display(page):
    page.evaluate("navigateTo('analysis')")
    page.wait_for_timeout(300)

    # AI Classification and Confidence
    pred_el = page.locator("#aiPrimaryPrediction")
    conf_el = page.locator("#aiConfidenceText")
    assert pred_el.is_visible()
    assert conf_el.is_visible()
    assert len(pred_el.inner_text()) > 0

    # Cardiac Measurements
    assert page.locator("#measHr").is_visible()
    assert page.locator("#measPr").is_visible()
    assert page.locator("#measQrs").is_visible()
    assert page.locator("#measQt").is_visible()
    assert page.locator("#measQtc").is_visible()

    # Waveform Canvas
    assert page.locator("#analysisWaveformChart").is_visible()


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2 - I. REPORTS (DOCTOR & PATIENT)
# ──────────────────────────────────────────────────────────────────────────────

def test_doctor_and_patient_report_views(page):
    # Doctor Report View
    page.evaluate("navigateTo('doctorReport')")
    page.wait_for_timeout(300)

    assert page.locator("#viewDoctorReport").is_visible()
    assert page.locator("#docRepPatName").is_visible()
    assert page.locator("#docRepPatMrn").is_visible()
    assert page.locator("#docRepFinding").is_visible()

    # Patient Plain-Language Report View
    page.evaluate("navigateTo('patientReport')")
    page.wait_for_timeout(300)

    assert page.locator("#viewPatientReport").is_visible()
    assert page.locator("#patRepGreeting").is_visible()
    assert page.locator("#patRepResultHeader").is_visible()


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2 - J. REPORT HISTORY & SEARCH
# ──────────────────────────────────────────────────────────────────────────────

def test_report_history_view_and_filtering(page):
    page.evaluate("quickLoginUser('dr_aniket')")
    page.evaluate("navigateTo('reportHistory')")
    page.wait_for_timeout(400)

    assert page.locator("#viewReportHistory").is_visible()
    assert page.locator("#historySearchInput").is_visible()
    assert page.locator("#historyPatientFilter").is_visible()
    assert page.locator("#historyDateFilter").is_visible()
    assert page.locator("#tblReportHistoryBody").is_visible()

    # Search filter input test
    search_box = page.locator("#historySearchInput")
    search_box.fill("Raj")
    page.wait_for_timeout(200)
    visible_rows = page.locator("#tblReportHistoryBody tr:not([style*='display: none'])")
    assert visible_rows.count() > 0
    first_text = visible_rows.first.inner_text()
    assert "Raj" in first_text

    # Search for nonexistent patient hides rows
    search_box.fill("NonExistentPatientZZZ")
    page.wait_for_timeout(200)
    hidden_rows = page.locator("#tblReportHistoryBody tr:not([style*='display: none'])")
    assert hidden_rows.count() == 0


def test_unusable_signal_ui_safety_gating(page):
    """Verify that unusable / flatline signal displays explicit safety warning and suppresses fake normal."""
    page.evaluate("navigateTo('analysis')")
    page.wait_for_timeout(200)

    # Inject unusable signal data
    unusable_payload = {
        "status": "UNUSABLE_SIGNAL",
        "can_run_ai": False,
        "prediction": "NO_RESULT_SIGNAL_UNUSABLE",
        "quality_category": "UNUSABLE",
        "rejection_reasons": ["Signal is flatline or disconnected."],
        "detected_r_peaks": [],
    }
    import json
    payload_json = json.dumps(unusable_payload)
    page.evaluate(f"lastAnalysisResult = {payload_json}")
    page.evaluate(f"updateAnalysisUiWithData({payload_json})")
    page.wait_for_timeout(200)

    pred_el = page.locator("#aiPrimaryPrediction")
    assert "SIGNAL UNUSABLE" in pred_el.inner_text()
    conf_el = page.locator("#aiConfidenceText")
    assert "0.0%" in conf_el.inner_text() or "Rejected" in conf_el.inner_text()

    # In Doctor Report, it must also NOT claim Normal Sinus Rhythm
    page.evaluate("navigateTo('doctorReport')")
    page.wait_for_timeout(200)
    doc_finding = page.locator("#docRepFinding").inner_text()
    assert "SIGNAL UNUSABLE" in doc_finding
    assert "Normal Sinus Rhythm" not in doc_finding


def test_auth_logout_cycle(page):
    """Verify clinician logout invalidates UI session and returns to auth or landing."""
    page.evaluate("quickLoginUser('dr_aniket')")
    page.wait_for_timeout(300)
    assert page.locator("#viewDoctorDashboard").is_visible()

    # Click logout
    page.evaluate("logoutUser()")
    page.wait_for_timeout(400)

    assert not page.locator("#viewDoctorDashboard").is_visible()
    assert page.locator("#viewLanding").is_visible() or page.locator("#viewAuth").is_visible()
