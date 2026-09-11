# ==============================================================================
# Project: AI/ML-enabled Adaptive Noise Cancellation (ANC) for Defence Vehicles
# PS ID: SIH26052 | Team: Echo Shield | Theme: Smart Vehicles
#
# Dataset Aggregation with Anti-Data-Leakage Splitting
# Generates two datasets:
#   Dataset A: ALL RETAINED DATA (CORE + PROXY + ENVIRONMENTAL)
#   Dataset B: CORE + PROXY ONLY (Excluding environmental ambient noise)
# All splits strictly enforced at the RECORDING level.
# ==============================================================================

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset_sources import BEHAVIORAL_CLASSES, RELEVANCE_TIERS, DOMAIN_GAP_STATEMENT

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
MANIFEST_PATH = os.path.join(OUTPUTS_DIR, "dataset_manifest.csv")
SPLIT_MANIFEST_PATH = os.path.join(OUTPUTS_DIR, "split_manifest.csv")
SUMMARY_CSV_PATH = os.path.join(OUTPUTS_DIR, "dataset_summary.csv")


def build_splits(seed: int = 42):
    print("=" * 70)
    print("  ECHO SHIELD (SIH 2026) — Dataset Assembly & Anti-Leakage Split")
    print("=" * 70)
    print(f"\n{DOMAIN_GAP_STATEMENT}\n")

    if not os.path.isfile(MANIFEST_PATH):
        print(f"[ERROR] Manifest not found at: {MANIFEST_PATH}")
        print("Please run audio_lab/preprocess_dataset.py first.")
        sys.exit(1)

    df_manifest = pd.read_csv(MANIFEST_PATH)
    if len(df_manifest) == 0:
        print("[ERROR] Manifest is empty.")
        sys.exit(1)

    print(f"Total actual recordings in manifest: {len(df_manifest)}")

    # Anti-Leakage Recording-Level Split:
    # Stratified by behavioral class at the source clip level
    recordings = np.array(df_manifest["clip_id"].tolist())
    categories = np.array(df_manifest["behavior_class"].tolist())

    # Step 1: Split into Train (70%) and Temp (30%)
    train_clips, temp_clips, _, y_temp = train_test_split(
        recordings,
        categories,
        test_size=0.30,
        random_state=seed,
        stratify=categories
    )

    # Step 2: Split Temp into Validation (15%) and Test (15%)
    val_clips, test_clips, _, _ = train_test_split(
        temp_clips,
        y_temp,
        test_size=0.50,
        random_state=seed,
        stratify=y_temp
    )

    train_set = set(train_clips)
    val_set = set(val_clips)
    test_set = set(test_clips)

    # Anti-leakage mathematical verification
    assert len(train_set.intersection(val_set)) == 0, "DATA LEAKAGE: Overlap between Train and Validation!"
    assert len(train_set.intersection(test_set)) == 0, "DATA LEAKAGE: Overlap between Train and Test!"
    assert len(val_set.intersection(test_set)) == 0, "DATA LEAKAGE: Overlap between Validation and Test!"
    print("  [VERIFIED] Anti-Leakage Check Passed: 0% recording overlap across splits.")

    def assign_split(clip_id):
        if clip_id in train_set:
            return "train"
        elif clip_id in val_set:
            return "val"
        else:
            return "test"

    df_manifest["split"] = df_manifest["clip_id"].apply(assign_split)
    df_manifest.to_csv(SPLIT_MANIFEST_PATH, index=False)
    print(f"  [SAVED] Split manifest written to: {SPLIT_MANIFEST_PATH}")

    # Containers for Dataset A (All Retained)
    X_train_all, y_train_all, tier_train_all = [], [], []
    X_val_all, y_val_all, tier_val_all = [], [], []
    X_test_all, y_test_all, tier_test_all = [], [], []

    # Containers for Dataset B (Core + Proxy Only)
    X_train_cp, y_train_cp = [], []
    X_val_cp, y_val_cp = [], []
    X_test_cp, y_test_cp = [], []

    split_stats = {
        "train": {
            "recordings_all": 0, "recordings_cp": 0,
            "windows_all": 0, "windows_cp": 0,
            "classes_all": {c: 0 for c in BEHAVIORAL_CLASSES},
            "classes_cp": {c: 0 for c in BEHAVIORAL_CLASSES},
            "tiers": {t: 0 for t in RELEVANCE_TIERS}
        },
        "val": {
            "recordings_all": 0, "recordings_cp": 0,
            "windows_all": 0, "windows_cp": 0,
            "classes_all": {c: 0 for c in BEHAVIORAL_CLASSES},
            "classes_cp": {c: 0 for c in BEHAVIORAL_CLASSES},
            "tiers": {t: 0 for t in RELEVANCE_TIERS}
        },
        "test": {
            "recordings_all": 0, "recordings_cp": 0,
            "windows_all": 0, "windows_cp": 0,
            "classes_all": {c: 0 for c in BEHAVIORAL_CLASSES},
            "classes_cp": {c: 0 for c in BEHAVIORAL_CLASSES},
            "tiers": {t: 0 for t in RELEVANCE_TIERS}
        }
    }

    for _, row in df_manifest.iterrows():
        split = row["split"]
        beh = row["behavior_class"]
        tier = row["relevance_tier"]
        feat_file = os.path.join(PROCESSED_DIR, row["feature_file"])

        if not os.path.isfile(feat_file):
            continue

        feats = np.load(feat_file)
        n_windows = len(feats)
        labels = [beh] * n_windows
        tiers = [tier] * n_windows

        is_core_or_proxy = (tier in ["CORE", "PROXY"])

        # Update stats
        split_stats[split]["recordings_all"] += 1
        split_stats[split]["windows_all"] += n_windows
        split_stats[split]["classes_all"][beh] += n_windows
        split_stats[split]["tiers"][tier] += n_windows

        if is_core_or_proxy:
            split_stats[split]["recordings_cp"] += 1
            split_stats[split]["windows_cp"] += n_windows
            split_stats[split]["classes_cp"][beh] += n_windows

        # Append to Dataset A
        if split == "train":
            X_train_all.append(feats)
            y_train_all.extend(labels)
            tier_train_all.extend(tiers)
            if is_core_or_proxy:
                X_train_cp.append(feats)
                y_train_cp.extend(labels)
        elif split == "val":
            X_val_all.append(feats)
            y_val_all.extend(labels)
            tier_val_all.extend(tiers)
            if is_core_or_proxy:
                X_val_cp.append(feats)
                y_val_cp.extend(labels)
        else:
            X_test_all.append(feats)
            y_test_all.extend(labels)
            tier_test_all.extend(tiers)
            if is_core_or_proxy:
                X_test_cp.append(feats)
                y_test_cp.extend(labels)

    # Convert to NumPy
    X_train_all = np.vstack(X_train_all).astype(np.float64) if X_train_all else np.empty((0, 11))
    y_train_all = np.array(y_train_all)
    tier_train_all = np.array(tier_train_all)

    X_val_all = np.vstack(X_val_all).astype(np.float64) if X_val_all else np.empty((0, 11))
    y_val_all = np.array(y_val_all)
    tier_val_all = np.array(tier_val_all)

    X_test_all = np.vstack(X_test_all).astype(np.float64) if X_test_all else np.empty((0, 11))
    y_test_all = np.array(y_test_all)
    tier_test_all = np.array(tier_test_all)

    X_train_cp = np.vstack(X_train_cp).astype(np.float64) if X_train_cp else np.empty((0, 11))
    y_train_cp = np.array(y_train_cp)

    X_val_cp = np.vstack(X_val_cp).astype(np.float64) if X_val_cp else np.empty((0, 11))
    y_val_cp = np.array(y_val_cp)

    X_test_cp = np.vstack(X_test_cp).astype(np.float64) if X_test_cp else np.empty((0, 11))
    y_test_cp = np.array(y_test_cp)

    # Save arrays for Dataset A (All Retained)
    np.save(os.path.join(DATA_DIR, "X_train_all.npy"), X_train_all)
    np.save(os.path.join(DATA_DIR, "y_train_all.npy"), y_train_all)
    np.save(os.path.join(DATA_DIR, "tier_train_all.npy"), tier_train_all)

    np.save(os.path.join(DATA_DIR, "X_val_all.npy"), X_val_all)
    np.save(os.path.join(DATA_DIR, "y_val_all.npy"), y_val_all)
    np.save(os.path.join(DATA_DIR, "tier_val_all.npy"), tier_val_all)

    np.save(os.path.join(DATA_DIR, "X_test_all.npy"), X_test_all)
    np.save(os.path.join(DATA_DIR, "y_test_all.npy"), y_test_all)
    np.save(os.path.join(DATA_DIR, "tier_test_all.npy"), tier_test_all)

    # Save arrays for Dataset B (Core + Proxy)
    np.save(os.path.join(DATA_DIR, "X_train_core_proxy.npy"), X_train_cp)
    np.save(os.path.join(DATA_DIR, "y_train_core_proxy.npy"), y_train_cp)

    np.save(os.path.join(DATA_DIR, "X_val_core_proxy.npy"), X_val_cp)
    np.save(os.path.join(DATA_DIR, "y_val_core_proxy.npy"), y_val_cp)

    np.save(os.path.join(DATA_DIR, "X_test_core_proxy.npy"), X_test_cp)
    np.save(os.path.join(DATA_DIR, "y_test_core_proxy.npy"), y_test_cp)

    # Default aliases point to All Retained
    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train_all)
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train_all)
    np.save(os.path.join(DATA_DIR, "X_val.npy"), X_val_all)
    np.save(os.path.join(DATA_DIR, "y_val.npy"), y_val_all)
    np.save(os.path.join(DATA_DIR, "X_test.npy"), X_test_all)
    np.save(os.path.join(DATA_DIR, "y_test.npy"), y_test_all)

    # Summary table
    print("\n" + "=" * 70)
    print("ACTUAL SPLIT STATISTICS (Counted, Anti-Leakage Verified):")
    print("=" * 70)
    summary_rows = []
    for split_name in ["train", "val", "test"]:
        st = split_stats[split_name]
        print(f"\n--- Split: {split_name.upper()} ---")
        print(f"  Version A (ALL RETAINED): {st['recordings_all']} recordings | {st['windows_all']:,} windows")
        for c in BEHAVIORAL_CLASSES:
            w = st["classes_all"][c]
            pct = (w / st["windows_all"] * 100.0) if st["windows_all"] > 0 else 0.0
            print(f"     • {c:<16}: {w:>6,} windows ({pct:>5.1f}%)")
        print(f"  Version B (CORE + PROXY): {st['recordings_cp']} recordings | {st['windows_cp']:,} windows")
        for c in BEHAVIORAL_CLASSES:
            w = st["classes_cp"][c]
            pct = (w / st["windows_cp"] * 100.0) if st["windows_cp"] > 0 else 0.0
            print(f"     • {c:<16}: {w:>6,} windows ({pct:>5.1f}%)")
        print(f"  Relevance Tiers in {split_name.upper()}:")
        for t in RELEVANCE_TIERS:
            w = st["tiers"][t]
            print(f"     * {t:<14}: {w:>6,} windows")

        summary_rows.append({
            "split": split_name,
            "recordings_all": st["recordings_all"],
            "windows_all": st["windows_all"],
            "stationary_all": st["classes_all"]["STATIONARY"],
            "non_stationary_all": st["classes_all"]["NON_STATIONARY"],
            "impulsive_all": st["classes_all"]["IMPULSIVE"],
            "recordings_core_proxy": st["recordings_cp"],
            "windows_core_proxy": st["windows_cp"],
            "stationary_core_proxy": st["classes_cp"]["STATIONARY"],
            "non_stationary_core_proxy": st["classes_cp"]["NON_STATIONARY"],
            "impulsive_core_proxy": st["classes_cp"]["IMPULSIVE"],
            "windows_core": st["tiers"]["CORE"],
            "windows_proxy": st["tiers"]["PROXY"],
            "windows_environmental": st["tiers"]["ENVIRONMENTAL"],
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(SUMMARY_CSV_PATH, index=False)
    print(f"\n[SAVED] Actual dataset summary saved to: {SUMMARY_CSV_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    build_splits()
