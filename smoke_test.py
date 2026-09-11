"""
Echo Shield — Pipeline smoke test
Runs run_full_pipeline on all 4 WAV files and prints a summary.
"""
import os
import sys

# Ensure we import from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils

MODEL_PATH = os.path.join("outputs", "noise_classifier.joblib")
WAV_FILES  = [
    os.path.join("data", "tank.wav"),
    os.path.join("data", "heli.wav"),
    os.path.join("data", "gun_fight.wav"),
    os.path.join("data", "test.wav"),
]

print("=" * 70)
print("  Echo Shield — Pipeline Smoke Test")
print("=" * 70)

# Load model
model, err = utils.load_model(MODEL_PATH)
if err:
    print(f"[FAIL] Model load error: {err}")
    sys.exit(1)
print(f"[OK]   Model loaded: {type(model).__name__}  classes={list(model.classes_)}")
print("-" * 70)

all_ok = True
for wav in WAV_FILES:
    if not os.path.isfile(wav):
        print(f"[SKIP] Missing file: {wav}")
        continue
    try:
        r = utils.run_full_pipeline(wav, model, max_anc_samples=60_000)
        print(
            f"[OK]   {os.path.basename(wav):15s} "
            f"class={r['majority_class']:12s} "
            f"LMS={r['lms_db']:6.2f} dB  "
            f"NLMS={r['nlms_db']:6.2f} dB  "
            f"FxLMS={r['fx_db']:6.2f} dB  "
            f"wavelet_level={r['wv_level']}  "
            f"anc_capped={r['anc_capped']}"
        )
    except Exception as exc:
        print(f"[FAIL] {wav}: {exc}")
        import traceback
        traceback.print_exc()
        all_ok = False

print("=" * 70)
if all_ok:
    print("All files processed successfully.")
else:
    print("One or more files failed — see errors above.")
    sys.exit(1)
