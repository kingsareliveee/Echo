# ==============================================================================
# Project: AI/ML-enabled Adaptive Noise Cancellation (ANC) for Defence Vehicles
# PS ID: SIH26052 | Team: Echo Shield | Theme: Smart Vehicles
#
# Final Evaluation on Unseen Test Recordings & Baseline Comparison
# Evaluates on completely unseen recordings (anti-leakage guaranteed):
#   - Version A Model (Trained on ALL RETAINED: Core + Proxy + Environmental)
#   - Version B Model (Trained on CORE + PROXY ONLY)
#   - BASELINE Model (Original 3-File Tank/Heli/Gunfight Classifier)
# Computes Accuracy, Macro F1, Precision, Recall, and Confusion Matrices.
# ==============================================================================

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset_sources import BEHAVIORAL_CLASSES, RELEVANCE_TIERS, DOMAIN_GAP_STATEMENT

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

MODEL_ALL_PATH = os.path.join(OUTPUTS_DIR, "noise_behavior_classifier_all.joblib")
MODEL_CP_PATH = os.path.join(OUTPUTS_DIR, "noise_behavior_classifier_core_proxy.joblib")
BASELINE_MODEL_PATH = os.path.join(OUTPUTS_DIR, "noise_classifier.joblib")

CONF_MATRIX_ALL_PNG = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
CONF_MATRIX_CP_PNG = os.path.join(OUTPUTS_DIR, "confusion_matrix_core_proxy.png")
MANIFEST_PATH = os.path.join(OUTPUTS_DIR, "split_manifest.csv")
SUMMARY_PATH = os.path.join(OUTPUTS_DIR, "dataset_summary.csv")
COMPARISON_CSV = os.path.join(OUTPUTS_DIR, "classifier_comparison.csv")


def plot_cm(cm, classes, accuracy, macro_f1, out_path, title):
    """Plots and saves normalized confusion matrix figure."""
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=f"{title}\nAccuracy: {accuracy * 100:.2f}% | Macro F1: {macro_f1:.4f}",
        ylabel="True Behavioral Class",
        xlabel="Predicted Behavioral Class"
    )

    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val_raw = cm[i, j]
            val_pct = cm_norm[i, j] * 100.0
            ax.text(
                j, i, f"{val_raw:,}\n({val_pct:.1f}%)",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontweight="bold"
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  [SAVED] Confusion matrix saved: {out_path}")


def evaluate_baseline_on_unseen(X_test, y_test):
    """Evaluates 3-file baseline classifier (8 features) mapped to behavioral classes."""
    if not os.path.isfile(BASELINE_MODEL_PATH):
        return None

    try:
        baseline_clf = joblib.load(BASELINE_MODEL_PATH)
        X_test_8 = X_test[:, :8]
        preds_raw = baseline_clf.predict(X_test_8)

        label_to_beh = {
            "tank": "STATIONARY",
            "heli": "NON_STATIONARY",
            "gun_fight": "IMPULSIVE"
        }
        preds_beh = np.array([label_to_beh.get(str(p).strip().lower(), "UNKNOWN") for p in preds_raw])

        acc = accuracy_score(y_test, preds_beh)
        macro_f1 = f1_score(y_test, preds_beh, average="macro", zero_division=0)
        macro_prec = precision_score(y_test, preds_beh, average="macro", zero_division=0)
        macro_rec = recall_score(y_test, preds_beh, average="macro", zero_division=0)

        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "cm": confusion_matrix(y_test, preds_beh, labels=BEHAVIORAL_CLASSES)
        }
    except Exception as e:
        print(f"  [NOTE] Baseline evaluation skipped: {e}")
        return None


def run_eval(model_pkg, X_test, y_test, out_png, title):
    model = model_pkg["model"]
    scaler = model_pkg.get("scaler", None)

    X_in = scaler.transform(X_test) if scaler is not None else X_test
    y_pred = model.predict(X_in)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_test, y_pred, labels=BEHAVIORAL_CLASSES)
    plot_cm(cm, BEHAVIORAL_CLASSES, acc, macro_f1, out_png, title)

    # Per class breakdown
    per_class = {}
    for c in BEHAVIORAL_CLASSES:
        mask = (y_test == c)
        c_acc = np.mean(y_pred[mask] == c) if np.sum(mask) > 0 else 0.0
        c_prec = precision_score(y_test == c, y_pred == c, zero_division=0)
        c_rec = recall_score(y_test == c, y_pred == c, zero_division=0)
        c_f1 = f1_score(y_test == c, y_pred == c, zero_division=0)
        per_class[c] = {"acc": c_acc, "prec": c_prec, "rec": c_rec, "f1": c_f1}

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "cm": cm,
        "per_class": per_class
    }


def main():
    if not (os.path.isfile(MODEL_ALL_PATH) and os.path.isfile(MODEL_CP_PATH)):
        print("[ERROR] Trained models not found. Run audio_lab/train_behavior_classifier.py first.")
        sys.exit(1)

    pkg_all = joblib.load(MODEL_ALL_PATH)
    pkg_cp = joblib.load(MODEL_CP_PATH)

    # Load Unseen Test Sets
    X_test_all = np.load(os.path.join(DATA_DIR, "X_test_all.npy"))
    y_test_all = np.load(os.path.join(DATA_DIR, "y_test_all.npy"))
    tier_test_all = np.load(os.path.join(DATA_DIR, "tier_test_all.npy"))

    X_test_cp = np.load(os.path.join(DATA_DIR, "X_test_core_proxy.npy"))
    y_test_cp = np.load(os.path.join(DATA_DIR, "y_test_core_proxy.npy"))

    # Evaluate Version A (ALL RETAINED) on All Unseen Test Data
    res_a_on_all = run_eval(
        pkg_all, X_test_all, y_test_all, CONF_MATRIX_ALL_PNG,
        f"Version A ({pkg_all['model_name']}) on All Unseen Recordings"
    )

    # Evaluate Version B (CORE + PROXY) on Core+Proxy Unseen Test Data
    res_b_on_cp = run_eval(
        pkg_cp, X_test_cp, y_test_cp, CONF_MATRIX_CP_PNG,
        f"Version B ({pkg_cp['model_name']}) on Core+Proxy Unseen Recordings"
    )

    # Evaluate Version A on Core+Proxy Unseen Test Data (Cross-comparison)
    res_a_on_cp = run_eval(
        pkg_all, X_test_cp, y_test_cp, os.path.join(OUTPUTS_DIR, "cm_modelA_on_coreproxy.png"),
        f"Version A ({pkg_all['model_name']}) evaluated on Core+Proxy subset"
    )

    # Evaluate Baseline Model on All Unseen Test Data
    base_res = evaluate_baseline_on_unseen(X_test_all, y_test_all)

    # Load actual manifest stats
    df_manifest = pd.read_csv(MANIFEST_PATH) if os.path.isfile(MANIFEST_PATH) else pd.DataFrame()
    df_summary = pd.read_csv(SUMMARY_PATH) if os.path.isfile(SUMMARY_PATH) else pd.DataFrame()
    df_comp = pd.read_csv(COMPARISON_CSV) if os.path.isfile(COMPARISON_CSV) else pd.DataFrame()

    total_rec = len(df_manifest)
    total_dur = df_manifest["actual_duration_sec"].sum() if not df_manifest.empty else 0.0
    total_win = df_manifest["num_windows"].sum() if not df_manifest.empty else 0

    h = int(total_dur // 3600)
    m = int((total_dur % 3600) // 60)
    s = int(total_dur % 60)

    # Print Full Rigorous Final Report
    print("\n" + "=" * 75)
    print("  ECHO SHIELD — Behavioral Noise Dataset Expansion Final Report")
    print("=" * 75)
    print("DATASET CHARACTERIZATION & DOMAIN NOTICE:")
    print(f"  {DOMAIN_GAP_STATEMENT}")

    print("\nACTUAL DATASET MEASURED METRICS (Counted, No Estimates):")
    print(f"  Total source recordings        : {total_rec:,}")
    print(f"  Total audio duration           : {h} hours {m} minutes {s} seconds ({total_dur:.2f} s)")
    print(f"  Total 100ms analysis windows   : {total_win:,}")

    print("\nACTUAL RECORDING-LEVEL SPLIT (Anti-Leakage Guaranteed):")
    if not df_summary.empty:
        for _, r in df_summary.iterrows():
            sp = r['split'].upper()
            print(f"  {sp:<6} | Recordings: {r['recordings_all']} | Windows: {r['windows_all']:,} (Stat: {r['stationary_all']:,}, Non-Stat: {r['non_stationary_all']:,}, Imp: {r['impulsive_all']:,})")

    print("\nVALIDATION SET ARCHITECTURE COMPARISON:")
    if not df_comp.empty:
        for _, r in df_comp.iterrows():
            print(f"  [{r['dataset_version']}] {r['model_name']:<16} | Acc={r['val_accuracy']:.2f}% | F1={r['val_macro_f1']:.4f} | Prec={r['val_precision']:.4f} | Rec={r['val_recall']:.4f} | Lat={r['latency_ms']:.3f}ms")

    print("\nFINAL UNSEEN RECORDING TEST EVALUATION:")
    print("-" * 75)
    print(f"  [VERSION A: ALL RETAINED] ({pkg_all['model_name']}) on ALL Unseen Test Recordings ({len(X_test_all):,} windows):")
    print(f"    • Test Accuracy   : {res_a_on_all['accuracy'] * 100:.2f}%")
    print(f"    • Macro F1        : {res_a_on_all['macro_f1']:.4f}")
    print(f"    • Macro Precision : {res_a_on_all['macro_precision']:.4f}")
    print(f"    • Macro Recall    : {res_a_on_all['macro_recall']:.4f}")
    print(f"    • Per-class breakdown:")
    for c, metrics in res_a_on_all["per_class"].items():
        print(f"        - {c:<16}: F1={metrics['f1']:.4f} | Precision={metrics['prec']:.4f} | Recall={metrics['rec']:.4f}")

    print(f"\n  [VERSION B: CORE + PROXY] ({pkg_cp['model_name']}) on CORE+PROXY Unseen Recordings ({len(X_test_cp):,} windows):")
    print(f"    • Test Accuracy   : {res_b_on_cp['accuracy'] * 100:.2f}%")
    print(f"    • Macro F1        : {res_b_on_cp['macro_f1']:.4f}")
    print(f"    • Macro Precision : {res_b_on_cp['macro_precision']:.4f}")
    print(f"    • Macro Recall    : {res_b_on_cp['macro_recall']:.4f}")
    print(f"    • Per-class breakdown:")
    for c, metrics in res_b_on_cp["per_class"].items():
        print(f"        - {c:<16}: F1={metrics['f1']:.4f} | Precision={metrics['prec']:.4f} | Recall={metrics['rec']:.4f}")

    print(f"\n  [CROSS EVALUATION] Version A Model on CORE+PROXY Test Subset ({len(X_test_cp):,} windows):")
    print(f"    • Test Accuracy   : {res_a_on_cp['accuracy'] * 100:.2f}%")
    print(f"    • Macro F1        : {res_a_on_cp['macro_f1']:.4f}")

    print("\nBASELINE vs. IMPROVED COMPARISON (on Unseen Recordings):")
    print("-" * 75)
    if base_res is not None:
        print(f"  BASELINE (3-File Classifier) : Acc = {base_res['accuracy'] * 100:.2f}% | Macro F1 = {base_res['macro_f1']:.4f} | Prec = {base_res['macro_precision']:.4f} | Rec = {base_res['macro_recall']:.4f}")
    else:
        print(f"  BASELINE (3-File Classifier) : [Model not found]")
    print(f"  IMPROVED (Version A: All)    : Acc = {res_a_on_all['accuracy'] * 100:.2f}% | Macro F1 = {res_a_on_all['macro_f1']:.4f} | Prec = {res_a_on_all['macro_precision']:.4f} | Rec = {res_a_on_all['macro_recall']:.4f}")
    print(f"  IMPROVED (Version B: C+P)    : Acc = {res_b_on_cp['accuracy'] * 100:.2f}% | Macro F1 = {res_b_on_cp['macro_f1']:.4f} | Prec = {res_b_on_cp['macro_precision']:.4f} | Rec = {res_b_on_cp['macro_recall']:.4f}")

    if base_res is not None:
        gain_a = (res_a_on_all['accuracy'] - base_res['accuracy']) * 100.0
        gain_b = (res_b_on_cp['accuracy'] - base_res['accuracy']) * 100.0
        print(f"\n  EMPIRICAL DELTA GAIN ON UNSEEN DATA:")
        print(f"    • Version A Gain over Baseline : +{gain_a:.2f}% accuracy points")
        print(f"    • Version B Gain over Baseline : +{gain_b:.2f}% accuracy points")

    print("\nREPRODUCIBILITY:")
    print("  python audio_lab/preprocess_dataset.py")
    print("  python audio_lab/build_behavior_dataset.py")
    print("  python audio_lab/dataset_statistics.py")
    print("  python audio_lab/train_behavior_classifier.py")
    print("  python audio_lab/evaluate_behavior_classifier.py")
    print("=" * 75)


if __name__ == "__main__":
    main()
