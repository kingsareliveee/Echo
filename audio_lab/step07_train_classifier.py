# ==============================================================================
# Project: AI/ML-enabled Adaptive Noise Cancellation (ANC) for Defence Vehicles
# PS ID: SIH26052 | Team: Echo Shield | Theme: Smart Vehicles
# 
# Step 07: Lightweight Noise Classifier (Baseline Decision Tree Model)
# ==============================================================================

import os
import sys
import csv
import numpy as np
# pyrefly: ignore [missing-import]
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- 1. Define File Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

DATASET_PATH = os.path.join(OUTPUTS_DIR, "noise_dataset.csv")
MODEL_OUTPUT_PATH = os.path.join(OUTPUTS_DIR, "noise_classifier.joblib")

# Ensure outputs directory exists
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# 8 Acoustic Feature Columns used for Machine Learning
# (window_index, start_time_sec, end_time_sec are metadata and excluded from features)
FEATURE_COLUMNS = [
    "rms_energy",
    "zero_crossing_rate",
    "spectral_centroid_hz",
    "spectral_rolloff_hz",
    "band_energy_0_500",
    "band_energy_500_2000",
    "band_energy_2000_5000",
    "band_energy_5000_10000"
]

TARGET_COLUMN = "label"

print("=" * 60)
print("ECHO SHIELD - Step 07: Noise Classifier")
print("=" * 60)

# --- 2. Input Validation: Check Dataset File ---
if not os.path.isfile(DATASET_PATH):
    print(f"\n[ERROR] Dataset not found at: {DATASET_PATH}")
    print("\nPlease build the training dataset first by running Step 06:")
    print("  python audio_lab/step06_build_dataset.py\n")
    sys.exit(1)

# --- 3. Load and Parse Dataset ---
feature_rows = []
labels = []

with open(DATASET_PATH, mode="r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    
    # Verify expected columns exist
    if reader.fieldnames is None:
        print("\n[ERROR] Dataset CSV is empty.")
        sys.exit(1)
        
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        if col not in reader.fieldnames:
            print(f"\n[ERROR] Missing required column in dataset: '{col}'")
            sys.exit(1)
            
    for row in reader:
        try:
            # Parse 8 numerical features
            row_features = [float(row[col]) for col in FEATURE_COLUMNS]
            row_label = row[TARGET_COLUMN].strip()
            if not row_label:
                continue
            feature_rows.append(row_features)
            labels.append(row_label)
        except ValueError as e:
            print(f"\n[WARNING] Skipping malformed row: {e}")
            continue

X = np.array(feature_rows, dtype=np.float64)
y = np.array(labels, dtype=object)

total_rows = len(X)
if total_rows == 0:
    print("\n[ERROR] No valid data rows found in dataset.")
    sys.exit(1)

unique_classes = sorted(list(set(y)))

print(f"\nDataset rows : {total_rows:,}")
print(f"Features     : {len(FEATURE_COLUMNS)}")
print(f"Classes      : {', '.join(unique_classes)}")

# --- 4. Train-Test Split (80% Train, 20% Test with Stratification) ---
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining rows: {len(X_train):,}")
print(f"Testing rows : {len(X_test):,}")
print("-" * 60)

# --- 5. Train Baseline Decision Tree Classifier ---
# max_depth=5 prevents overfitting on small acoustic feature distributions
clf = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)

clf.fit(X_train, y_train)

# --- 6. Evaluate Model on Test Split ---
y_pred = clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred) * 100.0

print(f"\nTest Accuracy: {test_accuracy:.2f}%")
print("-" * 60)

print("\n[Classification Report]")
print(classification_report(y_test, y_pred, target_names=unique_classes))

print("[Confusion Matrix]")
cm = confusion_matrix(y_test, y_pred, labels=unique_classes)

# Clean formatted confusion matrix table
header_str = "Actual \\ Predicted | " + " | ".join([f"{c:>11}" for c in unique_classes])
print(header_str)
print("-" * len(header_str))
for idx, actual_cls in enumerate(unique_classes):
    row_counts = " | ".join([f"{cm[idx, j]:>11}" for j in range(len(unique_classes))])
    print(f"{actual_cls:<18} | {row_counts}")

print("-" * 60)
print(f"Class names: {unique_classes}")
print("-" * 60)

# --- 7. Save Model to Disk ---
joblib.dump(clf, MODEL_OUTPUT_PATH)
print(f"\n[ SUCCESS ] Model trained and saved to:")
print(f"  --> {MODEL_OUTPUT_PATH}")

# --- 8. Reload Model and Verify Predictions ---
reloaded_clf = joblib.load(MODEL_OUTPUT_PATH)
reloaded_pred = reloaded_clf.predict(X_test)

if np.array_equal(y_pred, reloaded_pred):
    print("\n[ SUCCESS ] Reloaded model verified.")
else:
    print("\n[WARNING] Reloaded model predictions differed from initial model.")

print("=" * 60)
