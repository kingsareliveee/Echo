# ==============================================================================
# Project: AI/ML-enabled Adaptive Noise Cancellation (ANC) for Defence Vehicles
# PS ID: SIH26052 | Team: Echo Shield | Theme: Smart Vehicles
#
# Behavioral Noise Classifier Training & Model Selection
# Evaluates DT vs. RF vs. MLP on both:
#   Version A: ALL RETAINED DATA (Core + Proxy + Environmental)
#   Version B: CORE + PROXY ONLY
# ==============================================================================

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset_sources import BEHAVIORAL_CLASSES, DOMAIN_GAP_STATEMENT

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

MODEL_ALL_PATH = os.path.join(OUTPUTS_DIR, "noise_behavior_classifier_all.joblib")
MODEL_CP_PATH = os.path.join(OUTPUTS_DIR, "noise_behavior_classifier_core_proxy.joblib")
DEFAULT_MODEL_PATH = os.path.join(OUTPUTS_DIR, "noise_behavior_classifier.joblib")
COMPARISON_CSV = os.path.join(OUTPUTS_DIR, "classifier_comparison.csv")
COMPARISON_PNG = os.path.join(OUTPUTS_DIR, "classifier_comparison.png")

FEATURE_NAMES = [
    "rms_energy",
    "zero_crossing_rate",
    "spectral_centroid_hz",
    "spectral_rolloff_hz",
    "band_energy_0_500",
    "band_energy_500_2000",
    "band_energy_2000_5000",
    "band_energy_5000_10000",
    "spectral_bandwidth_hz",
    "crest_factor",
    "band_energy_ratio"
]


def measure_latency_ms(model, X_sample, n_iters=1000, scaler=None):
    """Measures single-window inference latency in milliseconds."""
    sample = X_sample[:1]
    if scaler is not None:
        sample = scaler.transform(sample)

    for _ in range(50):
        _ = model.predict(sample)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = model.predict(sample)
    t1 = time.perf_counter()

    return ((t1 - t0) / n_iters) * 1000.0


def train_and_eval_suite(X_tr, y_tr, X_val, y_val, suite_label: str):
    print(f"\n" + "=" * 70)
    print(f"  TRAINING SUITE: {suite_label.upper()}")
    print(f"  Train windows: {len(X_tr):,} | Val windows: {len(X_val):,}")
    print("=" * 70)

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)

    candidate_models = {
        "Decision Tree": {
            "model": DecisionTreeClassifier(max_depth=10, criterion="gini", random_state=42),
            "use_scaled": False,
            "desc": "Interpretable tree, ultra-low latency on embedded ARM"
        },
        "Random Forest": {
            "model": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
            "use_scaled": False,
            "desc": "100-tree ensemble, high variance reduction"
        },
        "MLP (Neural Net)": {
            "model": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, early_stopping=True, random_state=42),
            "use_scaled": True,
            "desc": "Feedforward neural network (64-32 layers)"
        }
    }

    suite_results = []
    trained_artifacts = {}

    for name, cfg in candidate_models.items():
        print(f"--> Training {name} on {suite_label}...")
        clf = cfg["model"]
        use_scaled = cfg["use_scaled"]

        X_train_in = X_tr_scaled if use_scaled else X_tr
        X_val_in = X_val_scaled if use_scaled else X_val

        t0 = time.time()
        clf.fit(X_train_in, y_tr)
        train_sec = time.time() - t0

        y_pred = clf.predict(X_val_in)

        acc = accuracy_score(y_val, y_pred)
        macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        macro_prec = precision_score(y_val, y_pred, average="macro", zero_division=0)
        macro_rec = recall_score(y_val, y_pred, average="macro", zero_division=0)
        latency = measure_latency_ms(clf, X_val, scaler=scaler if use_scaled else None)

        print(f"    Val Accuracy : {acc * 100:.2f}%")
        print(f"    Macro F1     : {macro_f1:.4f}")
        print(f"    Precision    : {macro_prec:.4f} | Recall: {macro_rec:.4f}")
        print(f"    Latency      : {latency:.3f} ms/window (Train time: {train_sec:.2f}s)")

        res_entry = {
            "dataset_version": suite_label,
            "model_name": name,
            "val_accuracy": round(acc * 100.0, 2),
            "val_macro_f1": round(macro_f1, 4),
            "val_precision": round(macro_prec, 4),
            "val_recall": round(macro_rec, 4),
            "latency_ms": round(latency, 3),
            "train_time_sec": round(train_sec, 2)
        }
        suite_results.append(res_entry)
        trained_artifacts[name] = {
            "model": clf,
            "scaler": scaler if use_scaled else None,
            "metrics": res_entry
        }

    best_name = max(suite_results, key=lambda x: x["val_macro_f1"])["model_name"]
    best_art = trained_artifacts[best_name]

    return suite_results, best_name, best_art


def main():
    print("=" * 70)
    print("  ECHO SHIELD (SIH 2026) — Dual-Version Behavioral Model Selection")
    print("=" * 70)
    print(f"\n{DOMAIN_GAP_STATEMENT}\n")

    # Load Version A (All Retained)
    X_train_all = np.load(os.path.join(DATA_DIR, "X_train_all.npy"))
    y_train_all = np.load(os.path.join(DATA_DIR, "y_train_all.npy"))
    X_val_all = np.load(os.path.join(DATA_DIR, "X_val_all.npy"))
    y_val_all = np.load(os.path.join(DATA_DIR, "y_val_all.npy"))

    # Load Version B (Core + Proxy Only)
    X_train_cp = np.load(os.path.join(DATA_DIR, "X_train_core_proxy.npy"))
    y_train_cp = np.load(os.path.join(DATA_DIR, "y_train_core_proxy.npy"))
    X_val_cp = np.load(os.path.join(DATA_DIR, "X_val_core_proxy.npy"))
    y_val_cp = np.load(os.path.join(DATA_DIR, "y_val_core_proxy.npy"))

    # Suite A: ALL RETAINED
    res_a, best_name_a, best_art_a = train_and_eval_suite(
        X_train_all, y_train_all, X_val_all, y_val_all, "Version A: ALL RETAINED"
    )

    # Suite B: CORE + PROXY ONLY
    res_b, best_name_b, best_art_b = train_and_eval_suite(
        X_train_cp, y_train_cp, X_val_cp, y_val_cp, "Version B: CORE + PROXY ONLY"
    )

    # Save Model Packages
    pkg_a = {
        "model": best_art_a["model"],
        "model_name": best_name_a,
        "scaler": best_art_a["scaler"],
        "dataset_version": "ALL_RETAINED",
        "feature_names": FEATURE_NAMES,
        "classes": BEHAVIORAL_CLASSES,
        "val_metrics": best_art_a["metrics"]
    }
    joblib.dump(pkg_a, MODEL_ALL_PATH)
    joblib.dump(pkg_a, DEFAULT_MODEL_PATH)
    print(f"\n[SAVED] Version A Best Model saved to: {MODEL_ALL_PATH}")

    pkg_b = {
        "model": best_art_b["model"],
        "model_name": best_name_b,
        "scaler": best_art_b["scaler"],
        "dataset_version": "CORE_PROXY_ONLY",
        "feature_names": FEATURE_NAMES,
        "classes": BEHAVIORAL_CLASSES,
        "val_metrics": best_art_b["metrics"]
    }
    joblib.dump(pkg_b, MODEL_CP_PATH)
    print(f"[SAVED] Version B Best Model saved to: {MODEL_CP_PATH}")

    # Combine results
    all_results = res_a + res_b
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(COMPARISON_CSV, index=False)
    print(f"[SAVED] Comprehensive comparison CSV: {COMPARISON_CSV}")

    # Plot Model Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Echo Shield — Model Architecture & Dataset Comparison (Validation Set)", fontsize=14, fontweight="bold")

    models = ["Decision Tree", "Random Forest", "MLP (Neural Net)"]
    x = np.arange(len(models))
    width = 0.35

    acc_a = [r["val_accuracy"] for r in res_a]
    acc_b = [r["val_accuracy"] for r in res_b]
    axes[0].bar(x - width/2, acc_a, width, label="All Retained", color="#2b5c8f", edgecolor="black")
    axes[0].bar(x + width/2, acc_b, width, label="Core + Proxy", color="#e6550d", edgecolor="black")
    axes[0].set_title("Validation Accuracy (%)", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=9)
    axes[0].set_ylim([0, 105])
    axes[0].legend(loc="lower right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    f1_a = [r["val_macro_f1"] for r in res_a]
    f1_b = [r["val_macro_f1"] for r in res_b]
    axes[1].bar(x - width/2, f1_a, width, label="All Retained", color="#2b5c8f", edgecolor="black")
    axes[1].bar(x + width/2, f1_b, width, label="Core + Proxy", color="#e6550d", edgecolor="black")
    axes[1].set_title("Validation Macro F1", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, fontsize=9)
    axes[1].set_ylim([0, 1.1])
    axes[1].legend(loc="lower right")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    lat_a = [r["latency_ms"] for r in res_a]
    axes[2].bar(x, lat_a, width=0.5, color="#2ca02c", edgecolor="black")
    axes[2].axhline(100.0, color="red", linestyle="--", label="100ms Real-Time Limit")
    axes[2].set_title("Inference Latency (ms / 100ms window)", fontsize=12, fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, fontsize=9)
    axes[2].legend(loc="upper right")
    axes[2].grid(axis="y", linestyle="--", alpha=0.4)
    for i, v in enumerate(lat_a):
        axes[2].text(i, v + 0.02, f"{v:.3f} ms", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(COMPARISON_PNG, dpi=200)
    plt.close()
    print(f"[SAVED] Comparison plot saved to: {COMPARISON_PNG}")
    print("=" * 70)


if __name__ == "__main__":
    main()
