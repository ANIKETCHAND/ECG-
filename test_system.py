import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import load_record
from prediction import predict_ecg

records = ["100", "101", "106", "119", "200", "208", "213"]
print("=" * 70)
print(f"{'Record':6s} | {'Predicted Pattern':45s} | {'Quality':10s} | {'Peaks':5s} | {'HR (BPM)':8s}")
print("=" * 70)
for r in records:
    sig, fs, _ = load_record(r, "data/raw")
    res = predict_ecg(sig[:int(15 * fs)], fs)
    pred_short = res["predicted_class"].split("(")[0].strip()
    print(f"{r:6s} | {pred_short:45s} | {res['signal_quality']:10s} | {len(res['detected_peaks']):5d} | {res['heart_rate_bpm']:8.1f}")
print("=" * 70)
