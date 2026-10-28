"""
Fast, Robust MIT-BIH Arrhythmia Dataset Downloader
==================================================
Direct HTTP streaming downloader using `requests`.
Bypasses wfdb's internal multiprocessing dummy pool to avoid Windows deadlocks.
Downloads all 48 standard MIT-BIH records and validates them with `wfdb.rdsamp`.
Syncs all files to both data/datasets/mit_bih_arrhythmia/raw/ and data/raw/.
"""

import sys
import shutil
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import wfdb

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
from data_loader import RECORDS, record_is_available

BASE_URL = "https://physionet.org/files/mitdb/1.0.0"
RAW_DATASET_DIR = ROOT_DIR / "data" / "datasets" / "mit_bih_arrhythmia" / "raw"
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"

RAW_DATASET_DIR.mkdir(parents=True, exist_ok=True)
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = "ECG-Guardian-Research/1.0 (Clinical Decision Support; Python/requests)"


def sync_file_between_dirs(filename: str):
    f1 = RAW_DATASET_DIR / filename
    f2 = DATA_RAW_DIR / filename
    if f1.exists() and f1.stat().st_size > 0:
        if not f2.exists() or f2.stat().st_size != f1.stat().st_size:
            shutil.copy2(f1, f2)
    elif f2.exists() and f2.stat().st_size > 0:
        if not f1.exists() or f1.stat().st_size != f2.stat().st_size:
            shutil.copy2(f2, f1)


def download_single_file(filename: str, session: requests.Session) -> bool:
    target_path = RAW_DATASET_DIR / filename
    raw_path = DATA_RAW_DIR / filename

    # Check if either already exists with non-zero size
    if target_path.exists() and target_path.stat().st_size > 0:
        sync_file_between_dirs(filename)
        return True
    if raw_path.exists() and raw_path.stat().st_size > 0:
        sync_file_between_dirs(filename)
        return True

    url = f"{BASE_URL}/{filename}"
    temp_path = target_path.with_suffix(f"{target_path.suffix}.tmp")

    for attempt in range(1, 4):
        try:
            resp = session.get(url, stream=True, timeout=25, headers={"User-Agent": USER_AGENT})
            if resp.status_code == 200:
                with open(temp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                temp_path.replace(target_path)
                sync_file_between_dirs(filename)
                return True
            else:
                time.sleep(1.0 * attempt)
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            time.sleep(1.5 * attempt)

    return False


def download_record_files(record_id: str) -> tuple[str, bool]:
    session = requests.Session()
    success = True
    for ext in [".hea", ".dat", ".atr"]:
        fn = f"{record_id}{ext}"
        ok = download_single_file(fn, session)
        if not ok:
            success = False
            break

    # Verify record readability with wfdb if files exist
    if success:
        try:
            sig, meta = wfdb.rdsamp(str(RAW_DATASET_DIR / record_id), sampto=100)
            ann = wfdb.rdann(str(RAW_DATASET_DIR / record_id), "atr", sampto=100)
            return record_id, True
        except Exception as e:
            print(f"    [!] Corrupt record {record_id}: {e}, will need redownload")
            for ext in [".hea", ".dat", ".atr"]:
                (RAW_DATASET_DIR / f"{record_id}{ext}").unlink(missing_ok=True)
                (DATA_RAW_DIR / f"{record_id}{ext}").unlink(missing_ok=True)
            return record_id, False

    return record_id, False


def main():
    print("=" * 80)
    print("MIT-BIH Arrhythmia Database Ingestion Engine")
    print(f"Target catalog: {len(RECORDS)} standard recordings")
    print("=" * 80)

    # Initial cross-sync
    for rec in RECORDS:
        for ext in [".hea", ".dat", ".atr"]:
            sync_file_between_dirs(f"{rec}{ext}")

    missing_records = [r for r in RECORDS if not record_is_available(RAW_DATASET_DIR, r)]
    already_available = len(RECORDS) - len(missing_records)
    print(f"Verified complete records on disk: {already_available}/{len(RECORDS)}")

    if not missing_records:
        print("[OK] All 48 MIT-BIH recordings are already present and verified on disk!")
        return

    print(f"Records to download: {len(missing_records)} -> {missing_records}")
    print("[*] Starting concurrent download with 5 worker threads...")

    completed = 0
    failed = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_rec = {executor.submit(download_record_files, rec): rec for rec in missing_records}
        for future in as_completed(future_to_rec):
            rec = future_to_rec[future]
            try:
                rec_id, ok = future.result()
                if ok:
                    completed += 1
                    print(f"  [OK] ({completed}/{len(missing_records)}) Successfully ingested Record {rec_id}")
                else:
                    failed.append(rec)
                    print(f"  [FAIL] Failed to ingest Record {rec}")
            except Exception as e:
                failed.append(rec)
                print(f"  [FAIL] Error on Record {rec}: {e}")

    # Final tally
    final_available = [r for r in RECORDS if record_is_available(RAW_DATASET_DIR, r)]
    print("=" * 80)
    print(f"Ingestion finished! Succeeded: {completed}, Failed: {len(failed)}")
    print(f"Total verified MIT-BIH records on disk: {len(final_available)}/{len(RECORDS)}")
    print("=" * 80)

    if len(final_available) == len(RECORDS):
        print("[SUCCESS] All 48 records downloaded and verified!")
    else:
        print(f"[PARTIAL] {len(final_available)}/48 records available.")


if __name__ == "__main__":
    main()
