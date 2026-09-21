"""
Unit Tests for Phase 2: ECG Input & Ingestion Engine
====================================================
Tests loaders: CSV, TXT, NPY, JSON, EDF, XML, DICOM, WFDB, Metadata Extractor, Validation Gates.
"""

import io
import json
import struct
import numpy as np
import pytest

from src.ecg_core.models import ECGRecording
from src.ecg_input.csv_loader import load_csv_ecg
from src.ecg_input.dicom_loader import is_dicom_file, load_dicom_ecg
from src.ecg_input.edf_loader import load_edf_ecg
from src.ecg_input.extraction_validator import validate_extracted_signal
from src.ecg_input.input_detector import InputModality, detect_input_modality
from src.ecg_input.json_loader import load_json_ecg
from src.ecg_input.metadata_extractor import (
    extract_machine_statement,
    extract_patient_demographics,
    extract_technical_calibrations,
)
from src.ecg_input.npy_loader import load_npy_ecg
from src.ecg_input.signal_loader import load_any_ecg
from src.ecg_input.txt_loader import load_txt_ecg
from src.ecg_input.xml_loader import load_xml_ecg



def test_csv_loader_valid_single_column():
    csv_content = "\n".join([str(0.1 * i) for i in range(500)])
    rec, err = load_csv_ecg(csv_content.encode("utf-8"), fs=250.0, lead_name="II")
    assert err is None
    assert isinstance(rec, ECGRecording)
    assert rec.sampling_rate == 250.0
    assert rec.number_of_leads == 1
    assert rec.lead_names == ["II"]
    assert rec.duration == 2.0


def test_csv_loader_with_time_column():
    # 2 columns: Time, Voltage
    lines = ["Time,Lead_II"]
    for i in range(500):
        t = i * 0.002  # 500 Hz
        v = np.sin(2 * np.pi * 1.2 * t)
        lines.append(f"{t:.4f},{v:.4f}")
    csv_content = "\n".join(lines)
    rec, err = load_csv_ecg(csv_content.encode("utf-8"), fs=None)  # Infer fs from time
    assert err is None
    assert rec is not None
    assert abs(rec.sampling_rate - 500.0) < 5.0


def test_csv_loader_missing_sampling_rate_rejected():
    # Rule 4: Must never silently assume 360 Hz
    csv_content = "\n".join([str(0.1 * i) for i in range(200)])
    rec, err = load_csv_ecg(csv_content.encode("utf-8"), fs=None)
    assert rec is None
    assert "Missing sampling rate" in err


def test_csv_loader_empty_file_rejected():
    rec, err = load_csv_ecg(b"", fs=360.0)
    assert rec is None
    assert "empty" in err.lower()


def test_txt_loader():
    txt_content = "\n".join([f"{0.05 * i}\t{0.02 * i}" for i in range(300)])
    rec, err = load_txt_ecg(txt_content.encode("utf-8"), fs=360.0)
    assert err is None
    assert rec is not None
    assert rec.source_format == "TXT"


def test_npy_loader_1d():
    arr = np.linspace(-1.0, 1.0, 400)
    buf = io.BytesIO()
    np.save(buf, arr)
    rec, err = load_npy_ecg(buf.getvalue(), fs=200.0, lead_names=["V1"])
    assert err is None
    assert rec is not None
    assert rec.lead_names == ["V1"]
    assert rec.duration == 2.0


def test_json_loader_array_and_dict():
    # Plain list
    data_list = [0.12, 0.15, -0.05, 0.88] * 100
    rec1, err1 = load_json_ecg(json.dumps(data_list).encode("utf-8"), fs=250.0)
    assert err1 is None
    assert rec1 is not None

    # Structured dict
    data_dict = {
        "record_id": "REC-JSON-TEST",
        "sampling_rate": 500.0,
        "lead_names": ["II"],
        "signal": [0.1] * 500,
        "patient_id": "PT-9988",
    }
    rec2, err2 = load_json_ecg(json.dumps(data_dict).encode("utf-8"))
    assert err2 is None
    assert rec2 is not None
    assert rec2.patient_id == "PT-9988"
    assert rec2.sampling_rate == 500.0


def test_edf_loader_mock_binary():
    # Construct a minimal valid EDF header + 1 record
    n_signals = 1
    record_duration = 1.0
    samples_per_record = 200
    header = bytearray(256 + 256 * n_signals)
    header[:8] = b"0       "  # Version
    header[8:88] = b"PATIENT_MOCK".ljust(80)
    header[88:168] = b"RECORD_MOCK".ljust(80)
    header[168:176] = b"01.01.26"
    header[176:184] = b"12.00.00"
    header[184:192] = f"{len(header)}".encode("latin-1").ljust(8)
    header[236:244] = b"1       "  # 1 record
    header[244:252] = b"1       "  # 1 second duration
    header[252:256] = b"1   "  # 1 signal

    # Signal header offsets
    header[256 : 256 + 16] = b"ECG Lead II     "
    header[256 + 16 : 256 + 96] = b"Transducer      ".ljust(80)
    header[256 + 96 : 256 + 104] = b"mV      "
    header[256 + 104 : 256 + 112] = b"-5.0    "  # phys min
    header[256 + 112 : 256 + 120] = b"5.0     "  # phys max
    header[256 + 120 : 256 + 128] = b"-32768  "  # dig min
    header[256 + 128 : 256 + 136] = b"32767   "  # dig max
    header[256 + 136 : 256 + 216] = b"None    ".ljust(80)
    header[256 + 216 : 256 + 224] = b"200     "  # 200 samples/sec
    header[256 + 224 : 256 + 256] = b"Reserved".ljust(32)

    # Signal samples (sine wave in 16-bit integers)
    samples = [int(10000 * np.sin(2 * np.pi * 2.0 * i / 200)) for i in range(200)]
    data_bytes = struct.pack(f"<{len(samples)}h", *samples)
    full_edf = bytes(header) + data_bytes

    rec, err = load_edf_ecg(full_edf)
    assert err is None
    assert rec is not None
    assert rec.sampling_rate == 200.0
    assert rec.number_of_leads == 1
    assert "Lead II" in rec.lead_names[0]


def test_xml_loader():
    seq_vals = ", ".join(["0.05, 0.12, 0.18, 0.95, -0.22"] * 60)  # 300 samples at 500 Hz = 0.6s
    xml_data = f"""<?xml version="1.0" encoding="UTF-8"?>
    <AnnotatedECG>
        <component>
            <series rate="500.0">
                <sequence lead="II">
                    {seq_vals}
                </sequence>
            </series>
        </component>
    </AnnotatedECG>"""
    rec, err = load_xml_ecg(xml_data.encode("utf-8"))
    assert err is None
    assert rec is not None
    assert rec.sampling_rate == 500.0
    assert rec.lead_names == ["II"]
    assert rec.duration >= 0.5



def test_dicom_preamble_and_failsafe():
    # Check invalid bytes reject DICOM
    assert is_dicom_file(b"NOT A DICOM FILE") is False

    # Check synthetic 132 bytes with DICM marker
    dcm_bytes = b"\x00" * 128 + b"DICM"
    assert is_dicom_file(dcm_bytes) is True

    # Attempt load returns clean NOT IMPLEMENTED / error without pydicom
    rec, err = load_dicom_ecg(dcm_bytes)
    assert rec is None
    assert "pydicom" in err.lower() or "no result" in err.lower()


def test_metadata_extractor_demographics_and_calibrations():
    report_text = """
    Patient Name: John Doe    ID: MRN-88492-A    Age: 48 yr    Sex: Male
    25 mm/s    10 mm/mV    fs: 500 Hz
    INTERPRETATION: Normal Sinus Rhythm. No acute ST-T elevation.
    """
    demo = extract_patient_demographics(report_text)
    assert demo["patient_name"] == "John Doe"
    assert demo["patient_id"] == "MRN-88492-A"
    assert demo["age"] == "48"
    assert demo["sex"] == "MALE"

    cal = extract_technical_calibrations(report_text)
    assert cal["paper_speed_mm_s"] == 25.0
    assert cal["voltage_gain_mm_mv"] == 10.0
    assert cal["sampling_rate_hz"] == 500.0

    stmt = extract_machine_statement(report_text)
    assert stmt is not None
    assert "Normal Sinus Rhythm" in stmt


def test_universal_load_any_ecg_dispatcher():
    csv_bytes = "\n".join([str(0.2 * i) for i in range(400)]).encode("utf-8")
    rec, err = load_any_ecg(csv_bytes, "patient_trace.csv", fs=250.0)
    assert err is None
    assert rec is not None
    assert rec.sampling_rate == 250.0
    assert rec.source_format == "CSV"


def test_extraction_validator_rules():
    # Empty signal
    ok, msg = validate_extracted_signal(np.array([]), fs=360.0, confidence_score=0.9)
    assert not ok

    # Low confidence (< 0.60)
    sig = np.random.randn(500)
    ok, msg = validate_extracted_signal(sig, fs=360.0, confidence_score=0.45)
    assert not ok
    assert "confidence is too low" in msg

    # Too short (< 1.5s)
    short_sig = np.random.randn(200)
    ok, msg = validate_extracted_signal(short_sig, fs=360.0, confidence_score=0.85, min_duration_sec=1.5)
    assert not ok
    assert "below the minimum" in msg

    # Valid signal
    t = np.linspace(0, 3, 3 * 360)
    valid_sig = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 2.4 * t)
    ok, msg = validate_extracted_signal(valid_sig, fs=360.0, confidence_score=0.85)
    assert ok
