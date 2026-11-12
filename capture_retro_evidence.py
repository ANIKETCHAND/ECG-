# capture_retro_evidence.py
import os
import time
import threading
import urllib.request
from playwright.sync_api import sync_playwright

OUTPUT_DIR = os.path.abspath("evidence_screenshots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
BASE_URL = os.environ.get("TEST_APP_URL", "http://127.0.0.1:8000")

def _ensure_local_server(url: str):
    try:
        with urllib.request.urlopen(f"{url}/api/health", timeout=1.0) as resp:
            if resp.status == 200:
                return
    except Exception:
        pass

    import uvicorn
    def run_app():
        uvicorn.run("api.index:app", host="127.0.0.1", port=8000, log_level="warning")

    t = threading.Thread(target=run_app, daemon=True)
    t.start()
    for _ in range(30):
        time.sleep(0.2)
        try:
            with urllib.request.urlopen(f"{url}/api/health", timeout=0.5) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass

def capture_all():
    _ensure_local_server(BASE_URL)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        
        # Desktop context
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
        page.goto(BASE_URL)
        page.wait_for_load_state("domcontentloaded")
        time.sleep(1)

        # Login as doctor first to have all clinical state ready
        page.evaluate("quickLoginUser('dr_aniket')")
        time.sleep(1)

        # 1. Doctor Dashboard
        page.evaluate("navigateTo('doctorDashboard')")
        time.sleep(1.5) # allow telemetry canvas animation to render trace
        view1_path = os.path.join(OUTPUT_DIR, "01_doctor_dashboard.png")
        page.screenshot(path=view1_path, full_page=False)
        print(f"Captured: {view1_path}")

        # 2. Patient Record
        page.evaluate("navigateTo('patientProfile')")
        time.sleep(1)
        view2_path = os.path.join(OUTPUT_DIR, "02_patient_record.png")
        page.screenshot(path=view2_path, full_page=False)
        print(f"Captured: {view2_path}")

        # 3. ECG Analysis
        page.evaluate("navigateTo('analysis')")
        time.sleep(1)
        view3_path = os.path.join(OUTPUT_DIR, "03_ecg_analysis.png")
        page.screenshot(path=view3_path, full_page=False)
        print(f"Captured: {view3_path}")

        # 4. Reports View
        page.evaluate("navigateTo('reportHistory')")
        time.sleep(1)
        view4_path = os.path.join(OUTPUT_DIR, "04_reports_view.png")
        page.screenshot(path=view4_path, full_page=False)
        print(f"Captured: {view4_path}")

        # 5. Clinical ECG Report (Doctor)
        page.evaluate("navigateTo('doctorReport')")
        time.sleep(1)
        view5_path = os.path.join(OUTPUT_DIR, "05_clinical_ecg_report.png")
        page.screenshot(path=view5_path, full_page=True)
        print(f"Captured: {view5_path}")

        # 6. Patient Portal (Rajesh Kumar view)
        page.evaluate("quickLoginUser('raj_verma')")
        time.sleep(0.5)
        page.evaluate("navigateTo('patientDashboard')")
        time.sleep(1)
        view6_path = os.path.join(OUTPUT_DIR, "06_patient_portal.png")
        page.screenshot(path=view6_path, full_page=False)
        print(f"Captured: {view6_path}")

        # 7. Mobile View (Dedicated Handheld Monitor)
        page.evaluate("navigateTo('mobileMonitor')")
        time.sleep(1.5)
        view7_path = os.path.join(OUTPUT_DIR, "07_mobile_monitor_view.png")
        page.screenshot(path=view7_path, full_page=False)
        print(f"Captured: {view7_path}")

        # Also capture in responsive mobile device viewport (iPhone 14 / standard 390x844)
        mobile_context = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True)
        mobile_page = mobile_context.new_page()
        mobile_page.goto(BASE_URL)
        mobile_page.wait_for_load_state("domcontentloaded")
        mobile_page.evaluate("quickLoginUser('raj_verma')")
        mobile_page.evaluate("navigateTo('mobileMonitor')")
        time.sleep(1.5)
        view7_mobile_path = os.path.join(OUTPUT_DIR, "07_mobile_device_viewport.png")
        mobile_page.screenshot(path=view7_mobile_path, full_page=False)
        print(f"Captured: {view7_mobile_path}")

        browser.close()
        print("All screenshots successfully captured!")

if __name__ == "__main__":
    capture_all()
