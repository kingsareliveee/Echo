# ==============================================================================
# Project: AI/ML-enabled Adaptive Noise Cancellation (ANC) for Defence Vehicles
# PS ID: SIH26052 | Team: Echo Shield | Theme: Smart Vehicles
# 
# Step 14: Software Signal Model (Reference, Primary & Error Signals)
# ==============================================================================

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# --- 1. Define File Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(OUTPUTS_DIR, exist_ok=True)

INPUT_FILE_PATH = os.path.join(DATA_DIR, "tank.wav")
OUTPUT_PLOT_PATH = os.path.join(OUTPUTS_DIR, "step14_signals.png")

print("=" * 75)
print("  ECHO SHIELD - Step 14: Software ANC Signal Model")
print("=" * 75)

# --- 2. Input Validation ---
if not os.path.isfile(INPUT_FILE_PATH):
    print(f"\n[ERROR] Audio file not found at: {INPUT_FILE_PATH}")
    sys.exit(1)

# --- 3. Load and Normalize Audio ---
sample_rate, audio_data = wavfile.read(INPUT_FILE_PATH)

if audio_data.ndim > 1:
    channel_signal = audio_data[:, 0]
    channel_desc = f"Channel 1 of {audio_data.shape[1]} (Stereo)"
else:
    channel_signal = audio_data
    channel_desc = "Channel 1 (Mono)"

raw_max = np.max(np.abs(channel_signal))
if raw_max > 0:
    if np.issubdtype(channel_signal.dtype, np.integer):
        norm_factor = 32768.0 if np.issubdtype(channel_signal.dtype, np.int16) else float(raw_max)
        signal = channel_signal.astype(np.float64) / norm_factor
    else:
        signal = channel_signal.astype(np.float64) / float(raw_max)
else:
    signal = channel_signal.astype(np.float64)

num_samples = len(signal)
duration_sec = num_samples / sample_rate

# --- 4. Signal Model Definition ---
# In this baseline software experiment:
# - reference_signal x[n] = upstream reference microphone capturing engine/vehicle noise
# - desired_signal d[n]   = primary disturbance arriving at the listener/error sensor
reference_signal = signal.copy()
desired_signal = signal.copy()

ref_rms = float(np.sqrt(np.mean(reference_signal ** 2)))
desired_rms = float(np.sqrt(np.mean(desired_signal ** 2)))

# --- 5. Print Summary & Architectural Explanation ---
print(f"File Analyzed       : {os.path.basename(INPUT_FILE_PATH)}")
print(f"Sample Rate         : {sample_rate} Hz")
print(f"Number of Samples   : {num_samples:,} ({duration_sec:.3f} seconds)")
print(f"Channel Analyzed    : {channel_desc}")
print("-" * 75)
print(f"Reference Signal RMS: {ref_rms:.6f}")
print(f"Desired Signal RMS  : {desired_rms:.6f}")
print("-" * 75)
print("\n[SIGNAL MODEL DEFINITION & FLOW]")
print("1. Reference Microphone -> x[n] : Senses noise close to acoustic source (e.g. engine bay).")
print("2. Desired Signal        -> d[n] : Primary noise propagating toward vehicle crew/cabin.")
print("3. Residual Error        -> e[n] : e[n] = d[n] - y_sec[n] (Acoustic cancellation at ear).")
print("-" * 75)
print("[IMPORTANT DISCLAIMER]")
print("This is a simplified software simulation model, NOT a physical microphone arrangement.")
print("No physical acoustic cancellation is occurring here.")
print("=" * 75)

# --- 6. Plot and Save Signal Waveforms ---
time_axis = np.linspace(0.0, duration_sec, num_samples, endpoint=False)

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time_axis, reference_signal, color="#2980b9", linewidth=0.7)
plt.title("Reference Signal x[n] (Upstream Reference Noise Sensor)", fontsize=11, fontweight="bold")
plt.ylabel("Amplitude", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(2, 1, 2)
plt.plot(time_axis, desired_signal, color="#8e44ad", linewidth=0.7)
plt.title("Desired Primary Noise Signal d[n] (Disturbance at Error Sensor)", fontsize=11, fontweight="bold")
plt.xlabel("Time (seconds)", fontsize=10)
plt.ylabel("Amplitude", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT_PATH, dpi=300)
plt.close()

print(f"[SUCCESS] Signal model plot saved to: {OUTPUT_PLOT_PATH}")
print("=" * 75)
