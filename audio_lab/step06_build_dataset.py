# ==============================================================================
# Project: AI/ML-enabled Adaptive Noise Cancellation (ANC) for Defence Vehicles
# PS ID: SIH26052 | Team: Echo Shield | Theme: Smart Vehicles
# 
# Step 06: ML Dataset Builder (Feature Aggregation & Labelling)
# ==============================================================================

import os
import sys
import csv

# --- 1. Define File Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Ensure outputs directory exists
os.makedirs(OUTPUTS_DIR, exist_ok=True)

OUTPUT_DATASET_PATH = os.path.join(OUTPUTS_DIR, "noise_dataset.csv")

# Input files mapping: (filename, class_label)
# NOTE: test_window_features.csv is intentionally excluded as it is not a training class.
DATASET_SOURCES = [
    ("tank_window_features.csv", "tank"),
    ("heli_window_features.csv", "helicopter"),
    ("gun_fight_window_features.csv", "gun_fight")
]

EXPECTED_FEATURE_COLUMNS = [
    "window_index",
    "start_time_sec",
    "end_time_sec",
    "rms_energy",
    "zero_crossing_rate",
    "spectral_centroid_hz",
    "spectral_rolloff_hz",
    "band_energy_0_500",
    "band_energy_500_2000",
    "band_energy_2000_5000",
    "band_energy_5000_10000"
]

OUTPUT_HEADER = EXPECTED_FEATURE_COLUMNS + ["label"]

print("=" * 60)
print("ECHO SHIELD - Step 06: ML Dataset Builder")
print("=" * 60)
print("\nInput files:")
for filename, label in DATASET_SOURCES:
    print(f"- {filename} -> label={label}")

# --- 2. Check for Missing Input CSV Files ---
missing_files = []
for filename, _ in DATASET_SOURCES:
    filepath = os.path.join(OUTPUTS_DIR, filename)
    if not os.path.isfile(filepath):
        missing_files.append(filename)

if missing_files:
    print("\n[ERROR] Missing required window feature CSV file(s):")
    for mf in missing_files:
        print(f"  - {os.path.join(OUTPUTS_DIR, mf)}")
    print("\nPlease generate missing window features first by running Step 05:")
    if "tank_window_features.csv" in missing_files:
        print("  python audio_lab/step05_window_features.py data/tank.wav")
    if "heli_window_features.csv" in missing_files:
        print("  python audio_lab/step05_window_features.py data/heli.wav")
    if "gun_fight_window_features.csv" in missing_files:
        print("  python audio_lab/step05_window_features.py data/gun_fight.wav")
    print()
    sys.exit(1)

# --- 3. Read and Aggregate Data with Validation ---
combined_rows = []
rows_per_class = {}

for filename, label in DATASET_SOURCES:
    filepath = os.path.join(OUTPUTS_DIR, filename)
    class_row_count = 0
    
    with open(filepath, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        # Validate that all expected columns exist in the CSV header
        if reader.fieldnames is None:
            print(f"\n[ERROR] File {filename} is empty or has no header.")
            sys.exit(1)
            
        missing_columns = [col for col in EXPECTED_FEATURE_COLUMNS if col not in reader.fieldnames]
        if missing_columns:
            print(f"\n[ERROR] File {filename} is missing required column(s): {missing_columns}")
            sys.exit(1)
        
        for row in reader:
            # Build new row preserving exact values and appending label
            dataset_row = {col: row[col] for col in EXPECTED_FEATURE_COLUMNS}
            dataset_row["label"] = label
            combined_rows.append(dataset_row)
            class_row_count += 1
            
    rows_per_class[label] = class_row_count

total_rows = len(combined_rows)

# --- 4. Save Combined Dataset to CSV ---
with open(OUTPUT_DATASET_PATH, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=OUTPUT_HEADER)
    writer.writeheader()
    writer.writerows(combined_rows)

# --- 5. Print Summary Report ---
print("\nRows per class:")
for label, count in rows_per_class.items():
    print(f"{label:<12}: {count:,}")

print(f"\nTotal rows  : {total_rows:,}")
print("-" * 60)

print("First 5 dataset rows:")
print(f"{'Idx':<4} {'Start(s)':<9} {'End(s)':<8} {'RMS':<10} {'ZCR':<9} {'Centroid(Hz)':<13} {'Rolloff(Hz)':<12} {'Label':<12}")
for r in combined_rows[:5]:
    rms_val = float(r['rms_energy']) if r['rms_energy'] else 0.0
    zcr_val = float(r['zero_crossing_rate']) if r['zero_crossing_rate'] else 0.0
    cent_val = float(r['spectral_centroid_hz']) if r['spectral_centroid_hz'] else 0.0
    roll_val = float(r['spectral_rolloff_hz']) if r['spectral_rolloff_hz'] else 0.0
    print(f"{r['window_index']:<4} {r['start_time_sec']:<9} {r['end_time_sec']:<8} {rms_val:<10.2f} {zcr_val:<9.4f} {cent_val:<13.1f} {roll_val:<12.1f} {r['label']:<12}")

print("=" * 60)
print(f"\n[ SUCCESS ] Dataset saved to:")
print(f"  --> {OUTPUT_DATASET_PATH}")
print("=" * 60)
