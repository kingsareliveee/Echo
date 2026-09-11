# ==============================================================================
# Project: AI/ML-enabled Adaptive Noise Cancellation (ANC) for Defence Vehicles
# PS ID: SIH26052 | Team: Echo Shield | Theme: Smart Vehicles
#
# Dataset Statistics & Behavioral Distribution Visualization
# Calculates and visualizes actual measured metrics (no nominal estimates):
# - outputs/class_distribution.png
# - outputs/feature_distribution.png
# ==============================================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset_sources import BEHAVIORAL_CLASSES, RELEVANCE_TIERS, DOMAIN_GAP_STATEMENT

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
MANIFEST_PATH = os.path.join(OUTPUTS_DIR, "dataset_manifest.csv")
SPLIT_MANIFEST_PATH = os.path.join(OUTPUTS_DIR, "split_manifest.csv")
SUMMARY_PATH = os.path.join(OUTPUTS_DIR, "dataset_summary.csv")

CLASS_DIST_IMG = os.path.join(OUTPUTS_DIR, "class_distribution.png")
FEAT_DIST_IMG = os.path.join(OUTPUTS_DIR, "feature_distribution.png")

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

PALETTE = {
    "STATIONARY": "#1f77b4",       # Solid blue
    "NON_STATIONARY": "#ff7f0e",   # Dynamic amber/orange
    "IMPULSIVE": "#d62728"         # Sharp crimson
}

TIER_PALETTE = {
    "CORE": "#2ca02c",             # Emerald green
    "PROXY": "#1f77b4",            # Steel blue
    "ENVIRONMENTAL": "#9467bd"     # Muted purple
}


def plot_distributions(df_manifest: pd.DataFrame, df_summary: pd.DataFrame):
    """Generates a 4-panel publication-ready overview of the expanded dataset."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Echo Shield — Acoustic Behavioral Dataset Distributions (SIH 2026)", fontsize=15, fontweight="bold")

    # 1. Total Windows by Behavioral Class (Version A vs Version B)
    ax1 = axes[0, 0]
    class_all = df_manifest.groupby("behavior_class")["num_windows"].sum().reindex(BEHAVIORAL_CLASSES).fillna(0)
    df_cp = df_manifest[df_manifest["relevance_tier"].isin(["CORE", "PROXY"])]
    class_cp = df_cp.groupby("behavior_class")["num_windows"].sum().reindex(BEHAVIORAL_CLASSES).fillna(0)

    x = np.arange(len(BEHAVIORAL_CLASSES))
    width = 0.35
    b1 = ax1.bar(x - width/2, class_all, width, label="All Retained", color="#2b5c8f", edgecolor="black", alpha=0.85)
    b2 = ax1.bar(x + width/2, class_cp, width, label="Core + Proxy Only", color="#e6550d", edgecolor="black", alpha=0.85)
    ax1.set_title("Actual Windows per Behavioral Class", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(BEHAVIORAL_CLASSES)
    ax1.set_ylabel("Number of 100ms Windows", fontsize=10)
    ax1.legend(loc="upper right")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 100, f"{int(h):,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 100, f"{int(h):,}", ha="center", va="bottom", fontsize=9)

    # 2. Split Breakdown by Version (Train / Val / Test)
    ax2 = axes[0, 1]
    if not df_summary.empty:
        splits = [s.upper() for s in df_summary["split"]]
        w_all = df_summary["windows_all"]
        w_cp = df_summary["windows_core_proxy"]

        xs = np.arange(len(splits))
        ax2.bar(xs - width/2, w_all, width, label="All Retained Windows", color="#2b5c8f", edgecolor="black", alpha=0.85)
        ax2.bar(xs + width/2, w_cp, width, label="Core + Proxy Windows", color="#e6550d", edgecolor="black", alpha=0.85)
        ax2.set_title("Anti-Leakage Split Counts (Actual Windows)", fontsize=12, fontweight="bold")
        ax2.set_xticks(xs)
        ax2.set_xticklabels(splits)
        ax2.set_ylabel("Windows", fontsize=10)
        ax2.legend(loc="upper right")
        ax2.grid(axis="y", linestyle="--", alpha=0.4)

    # 3. Windows by Relevance Tier
    ax3 = axes[1, 0]
    tier_counts = df_manifest.groupby("relevance_tier")["num_windows"].sum().reindex(RELEVANCE_TIERS).fillna(0)
    bars_tier = ax3.bar(RELEVANCE_TIERS, tier_counts, color=[TIER_PALETTE[t] for t in RELEVANCE_TIERS], edgecolor="black", alpha=0.85)
    ax3.set_title("Actual Windows by Relevance Tier (Metadata)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Windows", fontsize=10)
    ax3.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars_tier:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., h + 100, f"{int(h):,}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 4. Source Dataset Distribution (Clips)
    ax4 = axes[1, 1]
    src_counts = df_manifest["dataset"].value_counts()
    ax4.pie(
        src_counts,
        labels=[f"{k}\n({v} files)" for k, v in src_counts.items()],
        autopct="%1.1f%%",
        startangle=140,
        colors=["#3182bd", "#31a354"][:len(src_counts)],
        wedgeprops={"edgecolor": "black", "linewidth": 1}
    )
    ax4.set_title("Actual Source Recordings by Dataset", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(CLASS_DIST_IMG, dpi=200)
    plt.close()
    print(f"  [SAVED] Class distribution figure: {CLASS_DIST_IMG}")


def plot_feature_boxplots():
    """Plots feature boxplots comparing distributions across target behavioral classes."""
    X_train_path = os.path.join(DATA_DIR, "X_train_all.npy")
    y_train_path = os.path.join(DATA_DIR, "y_train_all.npy")

    if not (os.path.isfile(X_train_path) and os.path.isfile(y_train_path)):
        return

    X = np.load(X_train_path)
    y = np.load(y_train_path)

    features_to_show = [
        ("rms_energy", "RMS Energy (Signal Power)", 0),
        ("zero_crossing_rate", "Zero Crossing Rate (Friction / High-Freq)", 1),
        ("spectral_centroid_hz", "Spectral Centroid (Hz) (Pitch Center)", 2),
        ("spectral_bandwidth_hz", "Spectral Bandwidth (Hz) (Spread)", 8),
        ("crest_factor", "Crest Factor (Peak / RMS) [Impulse Metric]", 9),
        ("band_energy_ratio", "Band Energy Ratio (0-500Hz / 2000-5000Hz)", 10),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle("Echo Shield — Acoustic Behavioral Feature Separation (Train Split)", fontsize=14, fontweight="bold")

    for idx, (f_key, f_title, f_col) in enumerate(features_to_show):
        r, c = idx // 3, idx % 3
        ax = axes[r, c]

        data_per_class = []
        for cls_name in BEHAVIORAL_CLASSES:
            vals = X[y == cls_name, f_col]
            if f_key in ["crest_factor", "band_energy_ratio"]:
                p98 = np.percentile(vals, 98) if len(vals) > 0 else 1.0
                vals = np.clip(vals, 0, p98)
            data_per_class.append(vals)

        bp = ax.boxplot(
            data_per_class,
            patch_artist=True,
            tick_labels=["STAT", "NON-STAT", "IMPULSE"],
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.5}
        )

        for patch, cls_name in zip(bp["boxes"], BEHAVIORAL_CLASSES):
            patch.set_facecolor(PALETTE[cls_name])
            patch.set_alpha(0.7)

        ax.set_title(f_title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(FEAT_DIST_IMG, dpi=200)
    plt.close()
    print(f"  [SAVED] Feature distribution figure: {FEAT_DIST_IMG}")


def main():
    print("=" * 70)
    print("  ECHO SHIELD (SIH 2026) — Dataset Statistics & Visualization")
    print("=" * 70)
    print(f"\n{DOMAIN_GAP_STATEMENT}\n")

    manifest_file = SPLIT_MANIFEST_PATH if os.path.isfile(SPLIT_MANIFEST_PATH) else MANIFEST_PATH
    if not os.path.isfile(manifest_file):
        print(f"[ERROR] Manifest not found at: {manifest_file}")
        sys.exit(1)

    df_manifest = pd.read_csv(manifest_file)
    df_summary = pd.read_csv(SUMMARY_PATH) if os.path.isfile(SUMMARY_PATH) else pd.DataFrame()

    total_rec = len(df_manifest)
    total_win = df_manifest["num_windows"].sum()
    total_dur_sec = df_manifest["actual_duration_sec"].sum()

    hours = int(total_dur_sec // 3600)
    mins = int((total_dur_sec % 3600) // 60)
    secs = int(total_dur_sec % 60)

    print("\n" + "=" * 70)
    print("ACTUAL DATASET MEASURED METRICS (No Estimates):")
    print("=" * 70)
    print(f"Total source recordings  : {total_rec:,}")
    print(f"Total audio duration     : {hours}h {mins}m {secs}s ({total_dur_sec:.2f} seconds)")
    print(f"Total 100ms windows      : {total_win:,}")

    print("\nActual Windows per Class (All Retained):")
    for c in BEHAVIORAL_CLASSES:
        w = df_manifest[df_manifest["behavior_class"] == c]["num_windows"].sum()
        r = len(df_manifest[df_manifest["behavior_class"] == c])
        pct = (w / total_win * 100.0) if total_win > 0 else 0.0
        print(f"  • {c:<16}: {w:>6,} windows ({pct:>5.1f}%) across {r} recordings")

    df_cp = df_manifest[df_manifest["relevance_tier"].isin(["CORE", "PROXY"])]
    total_win_cp = df_cp["num_windows"].sum()
    print(f"\nActual Windows per Class (Core + Proxy Only: {len(df_cp)} recordings, {total_win_cp:,} windows):")
    for c in BEHAVIORAL_CLASSES:
        w = df_cp[df_cp["behavior_class"] == c]["num_windows"].sum()
        pct = (w / total_win_cp * 100.0) if total_win_cp > 0 else 0.0
        print(f"  • {c:<16}: {w:>6,} windows ({pct:>5.1f}%)")

    print("\nActual Windows per Relevance Tier:")
    for t in RELEVANCE_TIERS:
        w = df_manifest[df_manifest["relevance_tier"] == t]["num_windows"].sum()
        r = len(df_manifest[df_manifest["relevance_tier"] == t])
        pct = (w / total_win * 100.0) if total_win > 0 else 0.0
        print(f"  • {t:<16}: {w:>6,} windows ({pct:>5.1f}%) across {r} recordings")

    if not df_summary.empty:
        print("\nActual Split Counts (by Recording & Windows):")
        for _, row in df_summary.iterrows():
            sp = row['split'].upper()
            print(f"  Split: {sp:<6} | Recordings: {row['recordings_all']} (All) / {row['recordings_core_proxy']} (CP) | Windows: {row['windows_all']:,} (All) / {row['windows_core_proxy']:,} (CP)")

    print("\nGenerating figures...")
    plot_distributions(df_manifest, df_summary)
    plot_feature_boxplots()
    print("=" * 70)


if __name__ == "__main__":
    main()
