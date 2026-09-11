# ==============================================================================
# Project: AI/ML-enabled Adaptive Noise Cancellation (ANC) for Defence Vehicles
# PS ID: SIH26052 | Team: Echo Shield | Theme: Smart Vehicles
# 
# Step 04: Audio Feature Extraction (Acoustic Descriptors)
# ==============================================================================

import os
import sys
import csv
import numpy as np
from scipy.io import wavfile

# --- 1. Define File Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Ensure outputs and data directories exist
os.makedirs(os.path.join(PROJECT_ROOT, "outputs"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)

# Check if a WAV file path was provided as a command-line argument
if len(sys.argv) > 1:
    raw_path = sys.argv[1]
    if os.path.isfile(raw_path):
        INPUT_FILE_PATH = os.path.abspath(raw_path)
    elif os.path.isfile(os.path.join(PROJECT_ROOT, raw_path)):
        INPUT_FILE_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, raw_path))
    else:
        INPUT_FILE_PATH = os.path.abspath(raw_path)
else:
    # Default file path if no argument is passed
    INPUT_FILE_PATH = os.path.join(PROJECT_ROOT, "data", "test.wav")

# Automatically determine the output CSV filename based on input WAV filename
file_stem = os.path.splitext(os.path.basename(INPUT_FILE_PATH))[0]
OUTPUT_CSV_PATH = os.path.join(PROJECT_ROOT, "outputs", f"{file_stem}_features.csv")

# --- 2. Input Validation: Check File Existence ---
if not os.path.isfile(INPUT_FILE_PATH):
    print("\n[ERROR] Audio file not found!")
    print(f"Target location: {INPUT_FILE_PATH}")
    print("\nPlease verify the file path or place a valid .wav file in the 'data/' folder:")
    print(f"  --> {os.path.join(PROJECT_ROOT, 'data', 'test.wav')}")
    print("\nUsage Examples:")
    print("  python audio_lab/step04_features.py")
    print("  python audio_lab/step04_features.py data/tank.wav")
    print("  python audio_lab/step04_features.py data/heli.wav")
    print("  python audio_lab/step04_features.py data/gun_fight.wav\n")
    sys.exit(1)

# --- 3. Load Audio with Validation ---
try:
    sample_rate, audio_data = wavfile.read(INPUT_FILE_PATH)
except Exception as e:
    print(f"\n[ERROR] Failed to load WAV file: {e}")
    sys.exit(1)

if sample_rate <= 0:
    print(f"\n[ERROR] Invalid sample rate ({sample_rate} Hz) detected in audio file.")
    sys.exit(1)

if audio_data is None or len(audio_data) == 0:
    print("\n[ERROR] The audio file contains no sample data (empty signal).")
    sys.exit(1)

# --- 4. Channel Selection for Single Channel Analysis ---
# For stereo or multi-channel audio, analyze Channel 1 (index 0) only, matching Steps 01-03
if audio_data.ndim == 1:
    num_channels = 1
    channel_signal = audio_data
    channel_desc = "Channel 1 (Mono)"
else:
    num_channels = audio_data.shape[1]
    channel_signal = audio_data[:, 0]  # Channel 1 (index 0)
    channel_desc = f"Channel 1 of {num_channels} (Stereo / Multi-Channel)"

# Convert audio samples to float64 to prevent numerical overflow in mathematical operations
signal = channel_signal.astype(np.float64)

num_samples = len(signal)
duration_seconds = num_samples / sample_rate
nyquist_freq = sample_rate / 2.0

# ==============================================================================
# FEATURE CALCULATIONS & MATHEMATICS:
# ==============================================================================

# A. Root Mean Square (RMS) Energy:
#    Formula: RMS = sqrt(mean(signal^2))
#    Measures the overall power / perceived loudness of the audio signal.
rms_energy = np.sqrt(np.mean(signal ** 2))

# B. Zero Crossing Rate (ZCR):
#    Formula: fraction of consecutive samples with different signs
#    Measures how often the signal waveform crosses the zero-amplitude line.
#    High ZCR often corresponds to noisy, high-frequency, or percussive sounds (gunfire/friction).
if num_samples > 1:
    zero_crossings = np.sum(np.abs(np.diff(np.signbit(signal))))
    zcr = zero_crossings / (num_samples - 1)
else:
    zcr = 0.0

# Compute Frequency Spectrum (Positive Frequencies) via FFT for Spectral Descriptors:
raw_fft = np.fft.fft(signal)
half_n = num_samples // 2

# Positive frequency magnitude normalized by N
magnitude = (2.0 / num_samples) * np.abs(raw_fft[:half_n])
if len(magnitude) > 0:
    magnitude[0] = magnitude[0] / 2.0  # DC component

# Frequency axis (0 Hz to Nyquist)
freq_axis = np.linspace(0.0, nyquist_freq, half_n, endpoint=False)

# C. Spectral Centroid:
#    Formula: Centroid = sum(freq * magnitude) / sum(magnitude)
#    Represents the "center of mass" of the spectrum (the average pitch / brightness).
total_magnitude = np.sum(magnitude)
if total_magnitude > 0:
    spectral_centroid = np.sum(freq_axis * magnitude) / total_magnitude
else:
    spectral_centroid = 0.0

# D. Spectral Rolloff (85% Threshold):
#    The frequency below which 85% (0.85) of the total cumulative spectral magnitude lies.
#    Helps distinguish between low-frequency rumble (tanks) and wideband noise (gunfire).
if total_magnitude > 0:
    cumulative_magnitude = np.cumsum(magnitude)
    rolloff_threshold = 0.85 * total_magnitude
    rolloff_idx = np.searchsorted(cumulative_magnitude, rolloff_threshold)
    # Ensure index is within frequency axis bounds
    rolloff_idx = min(rolloff_idx, len(freq_axis) - 1)
    spectral_rolloff = freq_axis[rolloff_idx]
else:
    spectral_rolloff = 0.0

# E. Band Energy Distribution:
#    Calculates the sum of spectral magnitudes within specific acoustic bands.
#    Only bands with lower bounds below the Nyquist frequency are valid.
predefined_bands = [
    ("0-500 Hz", 0.0, 500.0),
    ("500-2000 Hz", 500.0, 2000.0),
    ("2000-5000 Hz", 2000.0, 5000.0),
    ("5000-10000 Hz", 5000.0, 10000.0)
]

band_energies = {}
for band_label, f_low, f_high in predefined_bands:
    if f_low < nyquist_freq:
        # Band is within the valid Nyquist range
        f_high_actual = min(f_high, nyquist_freq)
        band_mask = (freq_axis >= f_low) & (freq_axis < f_high_actual)
        band_energy = float(np.sum(magnitude[band_mask]))
        band_energies[band_label] = band_energy

# --- 5. Print Clean Report ---
print("=" * 60)
print("ECHO SHIELD - Step 04: Audio Feature Extraction")
print("=" * 60)
print(f"File Path       : {INPUT_FILE_PATH}")
print(f"Sample Rate     : {sample_rate} Hz")
print(f"Duration        : {duration_seconds:.3f} seconds")
print(f"Channels        : {num_channels}")
print(f"Channel Analyzed: Channel 1")
print("\n--- Features ---")
print(f"RMS Energy        : {rms_energy:.4f}")
print(f"Zero Crossing Rate: {zcr:.6f}")
print(f"Spectral Centroid : {spectral_centroid:.2f} Hz")
print(f"Spectral Rolloff  : {spectral_rolloff:.2f} Hz")
print("\n--- Band Energy ---")
for band_label, energy_val in band_energies.items():
    print(f"{band_label:<16}: {energy_val:.4f}")
print("=" * 60)

# --- 6. Save Extracted Features to CSV ---
feature_rows = [
    ("RMS Energy", f"{rms_energy:.6f}"),
    ("Zero Crossing Rate", f"{zcr:.6f}"),
    ("Spectral Centroid (Hz)", f"{spectral_centroid:.2f}"),
    ("Spectral Rolloff (Hz)", f"{spectral_rolloff:.2f}")
]

for band_label, energy_val in band_energies.items():
    feature_rows.append((f"Band Energy ({band_label})", f"{energy_val:.6f}"))

with open(OUTPUT_CSV_PATH, mode="w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["feature", "value"])
    for row in feature_rows:
        writer.writerow(row)

print(f"\n[SUCCESS] Extracted features saved to: {OUTPUT_CSV_PATH}")
print("=" * 60)
