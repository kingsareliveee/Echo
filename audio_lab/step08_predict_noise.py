# ==============================================================================
# Project: AI/ML-enabled Adaptive Noise Cancellation (ANC) for Defence Vehicles
# PS ID: SIH26052 | Team: Echo Shield | Theme: Smart Vehicles
# 
# Step 08: Noise Prediction Demo (Inference on Unseen / Target Audio)
# ==============================================================================

import os
import sys
from collections import Counter
import numpy as np
# pyrefly: ignore [missing-import]
import joblib
from scipy.io import wavfile

# --- 1. Configuration Constants ---
WINDOW_DURATION_MS = 100  # Analysis window length in ms
HOP_DURATION_MS = 50      # Step size between consecutive windows in ms (50% overlap)

# Predefined frequency sub-bands (identical to Steps 04-07)
BANDS = [
    ("band_energy_0_500", 0.0, 500.0),
    ("band_energy_500_2000", 500.0, 2000.0),
    ("band_energy_2000_5000", 2000.0, 5000.0),
    ("band_energy_5000_10000", 5000.0, 10000.0)
]

# --- 2. Define File Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_PATH = os.path.join(OUTPUTS_DIR, "noise_classifier.joblib")

print("=" * 60)
print("ECHO SHIELD - Step 08: Noise Prediction Demo")
print("=" * 60)

# --- 3. Command Line Argument & Input Validation ---
if len(sys.argv) < 2:
    print("\n[INFO] No input WAV file specified. Using default: data/test.wav")
    print("Usage: python audio_lab/step08_predict_noise.py <path_to_wav_file>\n")
    raw_path = os.path.join(PROJECT_ROOT, "data", "test.wav")
else:
    raw_path = sys.argv[1]

if os.path.isfile(raw_path):
    INPUT_FILE_PATH = os.path.abspath(raw_path)
elif os.path.isfile(os.path.join(PROJECT_ROOT, raw_path)):
    INPUT_FILE_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, raw_path))
else:
    INPUT_FILE_PATH = os.path.abspath(raw_path)

if not os.path.isfile(INPUT_FILE_PATH):
    print(f"\n[ERROR] Audio file not found at: {INPUT_FILE_PATH}")
    sys.exit(1)

# --- 4. Check for Trained Model ---
if not os.path.isfile(MODEL_PATH):
    print(f"\n[ERROR] Trained classifier model not found at: {MODEL_PATH}")
    print("\nPlease train the classifier model first by running Step 07:")
    print("  python audio_lab/step07_train_classifier.py\n")
    sys.exit(1)

# Load existing trained model without retraining
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"\n[ERROR] Failed to load classifier model: {e}")
    sys.exit(1)

# --- 5. Load and Validate WAV File ---
try:
    sample_rate, audio_data = wavfile.read(INPUT_FILE_PATH)
except Exception as e:
    print(f"\n[ERROR] Failed to read WAV file: {e}")
    sys.exit(1)

if sample_rate <= 0:
    print(f"\n[ERROR] Invalid sample rate ({sample_rate} Hz).")
    sys.exit(1)

if audio_data is None or len(audio_data) == 0:
    print("\n[ERROR] Audio file is empty (0 samples).")
    sys.exit(1)

# Channel selection (Channel 1 only, matching Steps 01-07)
if audio_data.ndim == 1:
    num_channels = 1
    channel_signal = audio_data
    channel_desc = "Channel 1 (Mono)"
else:
    num_channels = audio_data.shape[1]
    channel_signal = audio_data[:, 0]  # Channel 1 (index 0)
    channel_desc = f"Channel 1 of {num_channels} (Stereo / Multi-Channel)"

signal = channel_signal.astype(np.float64)
total_samples = len(signal)
duration_seconds = total_samples / sample_rate
nyquist_freq = sample_rate / 2.0

window_samples = int(sample_rate * (WINDOW_DURATION_MS / 1000.0))
hop_samples = int(sample_rate * (HOP_DURATION_MS / 1000.0))

if total_samples < window_samples:
    print(f"\n[ERROR] Audio duration ({duration_seconds:.3f} s) is shorter than one analysis window ({WINDOW_DURATION_MS} ms).")
    sys.exit(1)

# --- 6. Windowing and Feature Extraction (Identical to Step 05 & Step 07) ---
half_n = window_samples // 2
freq_axis = np.linspace(0.0, nyquist_freq, half_n, endpoint=False)

feature_rows = []
start_sample = 0

while start_sample + window_samples <= total_samples:
    end_sample = start_sample + window_samples
    window_signal = signal[start_sample:end_sample]
    
    # 1. RMS Energy
    rms_energy = float(np.sqrt(np.mean(window_signal ** 2)))
    
    # 2. Zero Crossing Rate (ZCR)
    if window_samples > 1:
        zero_crossings = int(np.sum(np.abs(np.diff(np.signbit(window_signal)))))
        zcr = float(zero_crossings / (window_samples - 1))
    else:
        zcr = 0.0
    
    # 3. FFT Positive Frequencies
    raw_fft = np.fft.fft(window_signal)
    magnitude = (2.0 / window_samples) * np.abs(raw_fft[:half_n])
    if len(magnitude) > 0:
        magnitude[0] = magnitude[0] / 2.0  # DC component
    
    total_magnitude = float(np.sum(magnitude))
    
    # 4. Spectral Centroid
    if total_magnitude > 0:
        spectral_centroid = float(np.sum(freq_axis * magnitude) / total_magnitude)
    else:
        spectral_centroid = 0.0
    
    # 5. Spectral Rolloff (85% energy threshold)
    if total_magnitude > 0:
        cumulative_mag = np.cumsum(magnitude)
        rolloff_threshold = 0.85 * total_magnitude
        rolloff_idx = int(np.searchsorted(cumulative_mag, rolloff_threshold))
        rolloff_idx = min(rolloff_idx, len(freq_axis) - 1)
        spectral_rolloff = float(freq_axis[rolloff_idx])
    else:
        spectral_rolloff = 0.0
    
    # 6-9. Sub-band Energies
    band_energies = []
    for _, f_low, f_high in BANDS:
        if f_low < nyquist_freq:
            f_high_actual = min(f_high, nyquist_freq)
            band_mask = (freq_axis >= f_low) & (freq_axis < f_high_actual)
            band_energies.append(float(np.sum(magnitude[band_mask])))
        else:
            band_energies.append(0.0)
    
    # 8 features in exact order expected by trained model
    window_feature_vector = [
        rms_energy,
        zcr,
        spectral_centroid,
        spectral_rolloff,
        band_energies[0],
        band_energies[1],
        band_energies[2],
        band_energies[3]
    ]
    
    feature_rows.append(window_feature_vector)
    start_sample += hop_samples

X_eval = np.array(feature_rows, dtype=np.float64)
num_windows = len(X_eval)

# --- 7. Perform Inference with Loaded Model ---
predictions = model.predict(X_eval)

# --- 8. Print Summary & Window Predictions ---
print(f"File            : {INPUT_FILE_PATH}")
print(f"Sample Rate     : {sample_rate} Hz")
print(f"Duration        : {duration_seconds:.3f} seconds")
print(f"Channels        : {num_channels} ({channel_desc})")
print(f"Windows analyzed: {num_windows:,}")
print("-" * 60)

print("Window Predictions (First 10 windows):")
for idx in range(min(10, num_windows)):
    start_t = idx * (HOP_DURATION_MS / 1000.0)
    end_t = start_t + (WINDOW_DURATION_MS / 1000.0)
    print(f"Window {idx:>2} [{start_t:.2f}s - {end_t:.2f}s] -> {predictions[idx]}")

if num_windows > 10:
    print(f"... ({num_windows - 10} additional windows evaluated)")

# --- 9. Majority Voting Across All Windows ---
vote_counts = Counter(predictions)
majority_winner, majority_votes = vote_counts.most_common(1)[0]

print("\n" + "=" * 60)
print(f"Overall Prediction: {majority_winner}")
print("=" * 60)

print("\nClass Distribution Across Windows:")
# Sort classes for clean consistent display
for class_name in sorted(model.classes_):
    count = vote_counts.get(class_name, 0)
    pct = (count / num_windows) * 100.0
    print(f"{class_name:<14}: {count:>4} windows ({pct:>5.1f}%)")

# --- 10. Average Model Probabilities Across Windows (Diagnostic) ---
if hasattr(model, "predict_proba"):
    probas = model.predict_proba(X_eval)
    avg_probas = np.mean(probas, axis=0)
    
    print("\n------------------------------------------------------------")
    print("Average model probability across windows (Diagnostic):")
    print("------------------------------------------------------------")
    for class_name, avg_p in zip(model.classes_, avg_probas):
        print(f"{class_name:<14}: {avg_p * 100.0:>5.1f}%")
    print("------------------------------------------------------------")
    print("Note: Probabilities represent average leaf distribution across")
    print("      all windows, not a guaranteed confidence score.")

print("=" * 60)
