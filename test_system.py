import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import load_record
from prediction import predict_ecg

def run_benchmark():
    records = ["100", "101", "106", "119", "200", "208", "213"]
    data_dir = Path(__file__).parent / "data" / "raw"
    if not data_dir.exists() or not any(data_dir.glob("*.hea")):
        print(f"Benchmark skipped: Raw data not found in {data_dir}")
        return

    print("=" * 70)
    print(f"{'Record':6s} | {'Predicted Pattern':45s} | {'Quality':10s} | {'Peaks':5s} | {'HR (BPM)':8s}")
    print("=" * 70)
    for r in records:
        rec_file = data_dir / f"{r}.hea"
        if not rec_file.exists():
            continue
        sig, fs, _ = load_record(r, str(data_dir))
        res = predict_ecg(sig[:int(15 * fs)], fs)
        pred_short = res["predicted_class"].split("(")[0].strip()
        print(f"{r:6s} | {pred_short:45s} | {res['signal_quality']:10s} | {len(res['detected_peaks']):5d} | {res['heart_rate_bpm']:8.1f}")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
