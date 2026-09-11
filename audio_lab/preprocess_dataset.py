# ==============================================================================
# Project: AI/ML-enabled Adaptive Noise Cancellation (ANC) for Defence Vehicles
# PS ID: SIH26052 | Team: Echo Shield | Theme: Smart Vehicles
#
# Audio Preprocessing & 11-Feature Acoustic Descriptor Extraction
# Standardizes sample rate to 22.05 kHz, converts to mono, windows at 100ms/50ms hop,
# and extracts 11 time/frequency acoustic features per window.
# Records actual measured duration, window counts, and relevance tier metadata.
# ==============================================================================

import os
import sys
import csv
import argparse
import math
import numpy as np
import scipy.signal
from scipy.io import wavfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset_sources import (
    DATASET_REGISTRY,
    BEHAVIORAL_CLASSES,
    map_category_to_behavior,
    get_relevance_tier,
    DOMAIN_GAP_STATEMENT
)

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

TARGET_SR = 22050               # 22.05 kHz standard rate (Nyquist = 11,025 Hz)
WINDOW_DURATION_MS = 100        # 100 ms window
HOP_DURATION_MS = 50            # 50 ms hop (50% overlap)
MIN_DURATION_SEC = 0.5          # Minimum audio length required

WINDOW_SAMPLES = int(TARGET_SR * (WINDOW_DURATION_MS / 1000.0))  # 2205 samples
HOP_SAMPLES = int(TARGET_SR * (HOP_DURATION_MS / 1000.0))        # 1102 samples

# Frequency bands (Hz)
BANDS = [
    ("band_energy_0_500", 0.0, 500.0),
    ("band_energy_500_2000", 500.0, 2000.0),
    ("band_energy_2000_5000", 2000.0, 5000.0),
    ("band_energy_5000_10000", 5000.0, 10000.0),
]

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

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
MANIFEST_PATH = os.path.join(OUTPUTS_DIR, "dataset_manifest.csv")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def load_and_resample(file_path: str):
    """
    Loads an audio file, converts to mono float64 [-1.0, 1.0], and resamples to TARGET_SR.
    Returns: (signal, original_sample_rate, actual_duration_seconds) or None on failure.
    """
    try:
        if HAS_SOUNDFILE:
            data, orig_sr = sf.read(file_path, dtype="float64")
        else:
            orig_sr, data = wavfile.read(file_path)
            if data.dtype == np.int16:
                data = data.astype(np.float64) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float64) / 2147483648.0
            elif data.dtype == np.uint8:
                data = (data.astype(np.float64) - 128.0) / 128.0
            else:
                data = data.astype(np.float64)
    except Exception:
        return None

    if orig_sr <= 0 or data is None or len(data) == 0:
        return None

    # Mono conversion (Channel 0)
    if data.ndim > 1:
        signal = data[:, 0]
    else:
        signal = data

    signal = np.asarray(signal, dtype=np.float64)
    orig_duration = len(signal) / orig_sr

    if orig_duration < MIN_DURATION_SEC:
        return None

    if np.max(np.abs(signal)) < 1e-6:
        return None

    # Rational resampling to TARGET_SR
    if orig_sr != TARGET_SR:
        g = math.gcd(int(TARGET_SR), int(orig_sr))
        up = int(TARGET_SR) // g
        down = int(orig_sr) // g
        try:
            signal = scipy.signal.resample_poly(signal, up, down)
        except Exception:
            target_len = int(len(signal) * (TARGET_SR / orig_sr))
            signal = scipy.signal.resample(signal, target_len)

    # Normalize amplitude to [-1.0, 1.0]
    max_val = np.max(np.abs(signal))
    if max_val > 1e-6:
        signal = signal / max_val

    actual_duration = len(signal) / TARGET_SR
    return signal, orig_sr, actual_duration


def extract_features_from_window(w: np.ndarray, freq_axis: np.ndarray, nyquist_freq: float) -> np.ndarray:
    """Calculates 11 acoustic features for a single 100ms window."""
    n_samples = len(w)
    half_n = n_samples // 2

    # 1. RMS Energy
    rms = float(np.sqrt(np.mean(w ** 2)))

    # 2. Zero Crossing Rate
    if n_samples > 1:
        zc = int(np.sum(np.abs(np.diff(np.signbit(w)))))
        zcr = float(zc / (n_samples - 1))
    else:
        zcr = 0.0

    raw_fft = np.fft.fft(w)
    mag = (2.0 / n_samples) * np.abs(raw_fft[:half_n])
    if len(mag) > 0:
        mag[0] = mag[0] / 2.0
    tot_mag = float(np.sum(mag))

    # 3. Spectral Centroid
    centroid = float(np.sum(freq_axis * mag) / tot_mag) if tot_mag > 1e-9 else 0.0

    # 4. Spectral Rolloff (85%)
    if tot_mag > 1e-9:
        cum_mag = np.cumsum(mag)
        rolloff_idx = np.searchsorted(cum_mag, 0.85 * tot_mag)
        rolloff_idx = min(rolloff_idx, len(freq_axis) - 1)
        rolloff = float(freq_axis[rolloff_idx])
    else:
        rolloff = 0.0

    # 5-8. Sub-Band Energies
    band_vals = []
    for _, f_low, f_high in BANDS:
        if f_low < nyquist_freq:
            f_end = min(f_high, nyquist_freq)
            mask = (freq_axis >= f_low) & (freq_axis < f_end)
            band_vals.append(float(np.sum(mag[mask])))
        else:
            band_vals.append(0.0)

    # 9. Spectral Bandwidth
    bandwidth = float(np.sqrt(np.sum(((freq_axis - centroid) ** 2) * mag) / tot_mag)) if tot_mag > 1e-9 else 0.0

    # 10. Crest Factor
    peak = float(np.max(np.abs(w)))
    crest = float(peak / (rms + 1e-9))

    # 11. Band-Energy Ratio (0-500Hz / 2000-5000Hz)
    b0 = band_vals[0]
    b2 = band_vals[2]
    ber = float(b0 / (b2 + 1e-9))

    return np.array([
        rms, zcr, centroid, rolloff,
        band_vals[0], band_vals[1], band_vals[2], band_vals[3],
        bandwidth, crest, ber
    ], dtype=np.float64)


def process_audio_file(file_path: str):
    """Processes audio file and returns feature matrix [N, 11], original_sr, actual_duration."""
    res = load_and_resample(file_path)
    if res is None:
        return None, None, None
    signal, orig_sr, duration = res

    n_samples = len(signal)
    if n_samples < WINDOW_SAMPLES:
        return None, None, None

    half_n = WINDOW_SAMPLES // 2
    nyquist_freq = TARGET_SR / 2.0
    freq_axis = np.linspace(0.0, nyquist_freq, half_n, endpoint=False)

    windows_features = []
    start = 0
    while start + WINDOW_SAMPLES <= n_samples:
        window = signal[start : start + WINDOW_SAMPLES]
        feat = extract_features_from_window(window, freq_axis, nyquist_freq)
        windows_features.append(feat)
        start += HOP_SAMPLES

    if not windows_features:
        return None, None, None

    return np.array(windows_features, dtype=np.float64), orig_sr, duration


def discover_audited_clips():
    """Discovers all clips matching the approved audited classes with relevance tier metadata."""
    clips = []

    # 1. Baseline Defense Audio
    baseline_files = [
        ("tank.wav", "tank", "STATIONARY", "CORE"),
        ("heli.wav", "heli", "NON_STATIONARY", "CORE"),
        ("gun_fight.wav", "gun_fight", "IMPULSIVE", "CORE"),
    ]
    for filename, cat, beh, tier in baseline_files:
        p = os.path.join(DATA_DIR, filename)
        if os.path.isfile(p):
            clips.append({
                "clip_id": f"baseline_{cat}",
                "file_path": p,
                "dataset": "Baseline",
                "category": cat,
                "behavior": beh,
                "relevance_tier": tier,
                "license": "DRDO_Baseline_Open"
            })

    # 2. ESC-50 (only audited retained classes)
    esc_meta = os.path.join(PROJECT_ROOT, DATASET_REGISTRY["ESC-50"]["meta_file"])
    esc_audio_dir = os.path.join(PROJECT_ROOT, DATASET_REGISTRY["ESC-50"]["audio_dir"])
    if os.path.isfile(esc_meta) and os.path.isdir(esc_audio_dir):
        with open(esc_meta, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row["filename"]
                cat = row["category"]
                beh = map_category_to_behavior(cat)
                tier = get_relevance_tier(cat)
                if beh is not None and tier is not None:
                    p = os.path.join(esc_audio_dir, fname)
                    if os.path.isfile(p):
                        clip_id = os.path.splitext(fname)[0]
                        clips.append({
                            "clip_id": f"esc50_{clip_id}",
                            "file_path": p,
                            "dataset": "ESC-50",
                            "category": cat,
                            "behavior": beh,
                            "relevance_tier": tier,
                            "license": "CC_BY_NC_4.0"
                        })

    return clips


def main():
    parser = argparse.ArgumentParser(description="Echo Shield Audio Preprocessor")
    parser.add_argument("--subset", type=int, default=None, help="Process first N clips only (for testing)")
    args = parser.parse_args()

    print("=" * 70)
    print("  ECHO SHIELD (SIH 2026) — Dataset Audio Preprocessing & Feature Pipeline")
    print("=" * 70)
    print(f"Target Sample Rate: {TARGET_SR} Hz")
    print(f"Window / Hop      : {WINDOW_DURATION_MS} ms / {HOP_DURATION_MS} ms ({WINDOW_SAMPLES} / {HOP_SAMPLES} samples)")
    print(f"Features Extracted: {len(FEATURE_NAMES)} descriptors per window")
    print(f"\n{DOMAIN_GAP_STATEMENT}\n")

    clips = discover_audited_clips()
    print(f"Total candidate clips matching audited taxonomy: {len(clips)}")

    if len(clips) == 0:
        print("\n[ERROR] No audio clips found! Please verify dataset availability.")
        sys.exit(1)

    if args.subset is not None and args.subset > 0:
        print(f"--> Running on SUBSET of {args.subset} clips for verification.")
        clips = clips[:args.subset]

    manifest_rows = []
    processed_count = 0
    total_windows = 0
    skipped_count = 0
    total_duration_sec = 0.0

    class_counts = {"STATIONARY": 0, "NON_STATIONARY": 0, "IMPULSIVE": 0}
    tier_counts = {"CORE": 0, "PROXY": 0, "ENVIRONMENTAL": 0}

    for i, item in enumerate(clips):
        clip_id = item["clip_id"]
        feat_matrix, orig_sr, dur = process_audio_file(item["file_path"])

        if feat_matrix is None:
            skipped_count += 1
            continue

        n_win = len(feat_matrix)
        out_npy = os.path.join(PROCESSED_DIR, f"{clip_id}.npy")
        np.save(out_npy, feat_matrix)

        manifest_rows.append({
            "clip_id": clip_id,
            "source_file": os.path.basename(item["file_path"]),
            "dataset": item["dataset"],
            "category": item["category"],
            "behavior_class": item["behavior"],
            "relevance_tier": item["relevance_tier"],
            "license": item["license"],
            "actual_duration_sec": round(dur, 4),
            "original_sr": orig_sr,
            "num_windows": n_win,
            "feature_file": os.path.basename(out_npy)
        })

        processed_count += 1
        total_windows += n_win
        total_duration_sec += dur
        class_counts[item["behavior"]] += n_win
        tier_counts[item["relevance_tier"]] += n_win

        if (i + 1) % 50 == 0 or (i + 1) == len(clips):
            print(f"  Processed {i + 1}/{len(clips)} clips ({total_windows:,} windows extracted)...")

    # Save manifest CSV
    manifest_fields = [
        "clip_id", "source_file", "dataset", "category",
        "behavior_class", "relevance_tier", "license",
        "actual_duration_sec", "original_sr", "num_windows", "feature_file"
    ]
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    hours = int(total_duration_sec // 3600)
    mins = int((total_duration_sec % 3600) // 60)
    secs = int(total_duration_sec % 60)

    print("\n" + "=" * 70)
    print("ACTUAL PREPROCESSED DATASET METRICS (Measured, Not Estimated):")
    print("=" * 70)
    print(f"Actual source recordings      : {processed_count} (Skipped: {skipped_count})")
    print(f"Actual total audio duration   : {hours}h {mins}m {secs}s ({total_duration_sec:.2f} s)")
    print(f"Actual total 100ms windows    : {total_windows:,}")
    print("\nActual Windows per Target Behavioral Class:")
    for beh, cnt in class_counts.items():
        pct = (cnt / total_windows * 100.0) if total_windows > 0 else 0.0
        print(f"  • {beh:<18}: {cnt:>6,} windows ({pct:>5.1f}%)")
    print("\nActual Windows per Relevance Tier:")
    for tier, cnt in tier_counts.items():
        pct = (cnt / total_windows * 100.0) if total_windows > 0 else 0.0
        print(f"  • {tier:<18}: {cnt:>6,} windows ({pct:>5.1f}%)")
    print(f"\nManifest saved to: {MANIFEST_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
