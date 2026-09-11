# ==============================================================================
# Project: AI/ML-enabled Adaptive Noise Cancellation (ANC) for Defence Vehicles
# PS ID: SIH26052 | Team: Echo Shield | Theme: Smart Vehicles
# 
# Step 05: Audio Windowing + Per-Window Feature Extraction
# ==============================================================================

import os
import sys
import csv
import numpy as np
from scipy.io import wavfile

# --- 1. Configuration Constants ---
WINDOW_DURATION_MS = 100  # Length of each analysis window in milliseconds
HOP_DURATION_MS = 50      # Step / advance between consecutive windows (50% overlap)

# --- 2. Define File Paths ---
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

# Automatically determine output CSV filename
file_stem = os.path.splitext(os.path.basename(INPUT_FILE_PATH))[0]
OUTPUT_CSV_PATH = os.path.join(PROJECT_ROOT, "outputs", f"{file_stem}_window_features.csv")

# --- 3. Input Validation: Check File Existence ---
if not os.path.isfile(INPUT_FILE_PATH):
    print("\n[ERROR] Audio file not found!")
    print(f"Target location: {INPUT_FILE_PATH}")
    print("\nPlease verify the file path or place a valid .wav file in the 'data/' folder:")
    print(f"  --> {os.path.join(PROJECT_ROOT, 'data', 'test.wav')}")
    print("\nUsage Examples:")
    print("  python audio_lab/step05_window_features.py")
    print("  python audio_lab/step05_window_features.py data/tank.wav")
    print("  python audio_lab/step05_window_features.py data/heli.wav")
    print("  python audio_lab/step05_window_features.py data/gun_fight.wav\n")
    sys.exit(1)

# --- 4. Load Audio with Validation ---
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

# --- 5. Channel Selection for Single Channel Analysis ---
# For stereo or multi-channel audio, analyze Channel 1 (index 0) only, matching Steps 02-04
if audio_data.ndim == 1:
    num_channels = 1
    channel_signal = audio_data
    channel_desc = "Channel 1 (Mono)"
else:
    num_channels = audio_data.shape[1]
    channel_signal = audio_data[:, 0]  # Channel 1 (index 0)
    channel_desc = f"Channel 1 of {num_channels} (Stereo / Multi-Channel)"

# Convert audio samples to float64 to prevent numerical overflow
signal = channel_signal.astype(np.float64)
total_samples = len(signal)
duration_seconds = total_samples / sample_rate
nyquist_freq = sample_rate / 2.0

# --- 6. Calculate Window and Hop Sizes in Samples ---
window_samples = int(sample_rate * (WINDOW_DURATION_MS / 1000.0))
hop_samples = int(sample_rate * (HOP_DURATION_MS / 1000.0))

if total_samples < window_samples:
    print(f"\n[ERROR] Audio duration ({duration_seconds:.3f} s, {total_samples} samples) is shorter than one analysis window ({WINDOW_DURATION_MS} ms, {window_samples} samples).")
    sys.exit(1)

# Predefined frequency bands for sub-band energy calculation
BANDS = [
    ("band_energy_0_500", 0.0, 500.0),
    ("band_energy_500_2000", 500.0, 2000.0),
    ("band_energy_2000_5000", 2000.0, 5000.0),
    ("band_energy_5000_10000", 5000.0, 10000.0),
]

# Frequency axis corresponding to positive FFT bins of a single window
half_n = window_samples // 2
freq_axis = np.linspace(0.0, nyquist_freq, half_n, endpoint=False)

# --- 7. Iterate Over Audio Windows and Extract Features ---
window_features_list = []
window_index = 0
start_sample = 0

while start_sample + window_samples <= total_samples:
    end_sample = start_sample + window_samples
    window_signal = signal[start_sample:end_sample]
    
    start_time_sec = start_sample / sample_rate
    end_time_sec = end_sample / sample_rate
    
    # A. RMS Energy: sqrt(mean(x^2))
    rms_energy = float(np.sqrt(np.mean(window_signal ** 2)))
    
    # B. Zero Crossing Rate (ZCR): fraction of consecutive samples with different signs
    if window_samples > 1:
        zero_crossings = int(np.sum(np.abs(np.diff(np.signbit(window_signal)))))
        zcr = float(zero_crossings / (window_samples - 1))
    else:
        zcr = 0.0
    
    # FFT for Frequency-Domain Features on the Window
    raw_fft = np.fft.fft(window_signal)
    magnitude = (2.0 / window_samples) * np.abs(raw_fft[:half_n])
    if len(magnitude) > 0:
        magnitude[0] = magnitude[0] / 2.0  # DC component
    
    total_magnitude = float(np.sum(magnitude))
    
    # C. Spectral Centroid: sum(freq * mag) / sum(mag)
    if total_magnitude > 0:
        spectral_centroid = float(np.sum(freq_axis * magnitude) / total_magnitude)
    else:
        spectral_centroid = 0.0
    
    # D. Spectral Rolloff (85% energy threshold)
    if total_magnitude > 0:
        cumulative_mag = np.cumsum(magnitude)
        rolloff_threshold = 0.85 * total_magnitude
        rolloff_idx = int(np.searchsorted(cumulative_mag, rolloff_threshold))
        rolloff_idx = min(rolloff_idx, len(freq_axis) - 1)
        spectral_rolloff = float(freq_axis[rolloff_idx])
    else:
        spectral_rolloff = 0.0
    
    # E. Band Energy for Each Sub-band
    band_energy_values = {}
    for band_name, f_low, f_high in BANDS:
        if f_low < nyquist_freq:
            f_high_actual = min(f_high, nyquist_freq)
            band_mask = (freq_axis >= f_low) & (freq_axis < f_high_actual)
            band_energy = float(np.sum(magnitude[band_mask]))
        else:
            band_energy = 0.0
        band_energy_values[band_name] = band_energy
    
    row = {
        "window_index": window_index,
        "start_time_sec": round(start_time_sec, 4),
        "end_time_sec": round(end_time_sec, 4),
        "rms_energy": round(rms_energy, 6),
        "zero_crossing_rate": round(zcr, 6),
        "spectral_centroid_hz": round(spectral_centroid, 2),
        "spectral_rolloff_hz": round(spectral_rolloff, 2),
        "band_energy_0_500": round(band_energy_values["band_energy_0_500"], 6),
        "band_energy_500_2000": round(band_energy_values["band_energy_500_2000"], 6),
        "band_energy_2000_5000": round(band_energy_values["band_energy_2000_5000"], 6),
        "band_energy_5000_10000": round(band_energy_values["band_energy_5000_10000"], 6),
    }
    
    window_features_list.append(row)
    window_index += 1
    start_sample += hop_samples

num_windows = len(window_features_list)

# --- 8. Print Summary Report ---
print("=" * 60)
print("ECHO SHIELD - Step 05: Windowed Feature Extraction")
print("=" * 60)
print(f"File Path        : {INPUT_FILE_PATH}")
print(f"Sample Rate      : {sample_rate} Hz")
print(f"Duration         : {duration_seconds:.3f} seconds")
print(f"Channels         : {num_channels}")
print(f"Channel Analyzed : {channel_desc}")
print(f"Window Duration  : {WINDOW_DURATION_MS} ms ({window_samples} samples)")
print(f"Hop Duration     : {HOP_DURATION_MS} ms ({hop_samples} samples)")
print(f"Number of Windows: {num_windows:,}")
print("-" * 60)
print("First 5 windows:")
print(f"{'Idx':<4} {'Start(s)':<9} {'End(s)':<8} {'RMS':<10} {'ZCR':<9} {'Centroid(Hz)':<13} {'Rolloff(Hz)':<12}")
for w in window_features_list[:5]:
    print(f"{w['window_index']:<4} {w['start_time_sec']:<9.3f} {w['end_time_sec']:<8.3f} {w['rms_energy']:<10.2f} {w['zero_crossing_rate']:<9.4f} {w['spectral_centroid_hz']:<13.1f} {w['spectral_rolloff_hz']:<12.1f}")
print("=" * 60)

# --- 9. Save Features to CSV ---
fieldnames = [
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
    "band_energy_5000_10000",
]

with open(OUTPUT_CSV_PATH, mode="w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(window_features_list)

print(f"\n[ SUCCESS ] Windowed features saved to:")
print(f"  --> {OUTPUT_CSV_PATH}")
print("=" * 60)
