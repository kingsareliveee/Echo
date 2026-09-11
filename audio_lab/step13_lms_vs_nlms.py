# ==============================================================================
# Project: AI/ML-enabled Adaptive Noise Cancellation (ANC) for Defence Vehicles
# PS ID: SIH26052 | Team: Echo Shield | Theme: Smart Vehicles
# 
# Step 13: LMS vs. Normalized LMS (NLMS) Adaptive Filter Comparison
# ==============================================================================

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# ==============================================================================
# BEGINNER-FRIENDLY CONCEPTS: LMS vs. NLMS
#
# 1. THE PROBLEM WITH STANDARD LMS:
#    In standard LMS (w[n+1] = w[n] + \u03bc * e[n] * x[n]), the weight update magnitude
#    is directly proportional to the squared input signal energy (||x[n]||^2).
#    If the audio suddenly gets very loud (e.g., engine rev or explosion), the update
#    becomes massive, causing filter divergence and distortion.
#
# 2. HOW NLMS (NORMALIZED LMS) FIXES THIS:
#    NLMS normalizes the step size by dividing by the instantaneous energy of the
#    input buffer (x^T * x):
#        \u03bc_norm = \u03bc / (\u03b5 + ||x[n]||^2)
#        w[n+1] = w[n] + \u03bc_norm * e[n] * x[n]
#
# 3. WHAT IS EPSILON (\u03b5):
#    A tiny positive regularization constant (e.g., 1e-8) added to the denominator
#    to prevent division by zero during periods of quiet or complete silence.
# ==============================================================================

# --- 1. Configuration Constants ---
FILTER_LENGTH = 32

# LMS Parameters
LMS_MU = 0.005

# NLMS Parameters
NLMS_MU = 0.5
NLMS_EPSILON = 1e-8

# --- 2. Define File Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Support optional CLI argument, defaulting to data/tank.wav
if len(sys.argv) > 1:
    raw_path = sys.argv[1]
    if os.path.isfile(raw_path):
        INPUT_FILE_PATH = os.path.abspath(raw_path)
    elif os.path.isfile(os.path.join(PROJECT_ROOT, raw_path)):
        INPUT_FILE_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, raw_path))
    else:
        INPUT_FILE_PATH = os.path.abspath(raw_path)
else:
    INPUT_FILE_PATH = os.path.join(DATA_DIR, "tank.wav")

OUTPUT_PLOT_PATH = os.path.join(OUTPUTS_DIR, "step13_lms_vs_nlms.png")

print("=" * 75)
print("  ECHO SHIELD - Step 13: LMS vs. NLMS Adaptive Filter Comparison")
print("=" * 75)

# --- 3. Input Validation ---
if not os.path.isfile(INPUT_FILE_PATH):
    print(f"\n[ERROR] Audio file not found at: {INPUT_FILE_PATH}")
    sys.exit(1)

# --- 4. Load and Normalize Audio File ---
sample_rate, audio_data = wavfile.read(INPUT_FILE_PATH)

# Select Channel 1 (index 0) if stereo/multichannel
if audio_data.ndim > 1:
    channel_signal = audio_data[:, 0]
    channel_desc = f"Channel 1 of {audio_data.shape[1]} (Stereo)"
else:
    channel_signal = audio_data
    channel_desc = "Channel 1 (Mono)"

# Normalize audio to floating point range [-1.0, 1.0]
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
input_rms = float(np.sqrt(np.mean(signal ** 2)))

# --- 5. Algorithm 1: Standard LMS Adaptive Filter ---
weights_lms = np.zeros(FILTER_LENGTH, dtype=np.float64)
buffer_lms = np.zeros(FILTER_LENGTH, dtype=np.float64)
e_lms = np.zeros(num_samples, dtype=np.float64)

for n in range(num_samples):
    d_n = signal[n]
    y_n = np.dot(weights_lms, buffer_lms)
    err = d_n - y_n
    e_lms[n] = err
    
    # Standard LMS update
    weights_lms += LMS_MU * err * buffer_lms
    
    # Update delay line
    buffer_lms[1:] = buffer_lms[:-1]
    buffer_lms[0] = d_n

# --- 6. Algorithm 2: Normalized LMS (NLMS) Adaptive Filter ---
weights_nlms = np.zeros(FILTER_LENGTH, dtype=np.float64)
buffer_nlms = np.zeros(FILTER_LENGTH, dtype=np.float64)
e_nlms = np.zeros(num_samples, dtype=np.float64)

for n in range(num_samples):
    d_n = signal[n]
    y_n = np.dot(weights_nlms, buffer_nlms)
    err = d_n - y_n
    e_nlms[n] = err
    
    # Energy of input buffer: sum(x^2)
    buffer_energy = float(np.dot(buffer_nlms, buffer_nlms))
    
    # Normalized LMS update with epsilon regularization
    norm_mu = NLMS_MU / (NLMS_EPSILON + buffer_energy)
    weights_nlms += norm_mu * err * buffer_nlms
    
    # Update delay line
    buffer_nlms[1:] = buffer_nlms[:-1]
    buffer_nlms[0] = d_n

# --- 7. Calculate Quantitative Comparison Metrics ---
lms_error_rms = float(np.sqrt(np.mean(e_lms ** 2)))
nlms_error_rms = float(np.sqrt(np.mean(e_nlms ** 2)))

lms_reduction_db = 20.0 * np.log10(input_rms / lms_error_rms) if lms_error_rms > 0 else 0.0
nlms_reduction_db = 20.0 * np.log10(input_rms / nlms_error_rms) if nlms_error_rms > 0 else 0.0

# --- 8. Print Comparative Table ---
print(f"File Analyzed       : {os.path.basename(INPUT_FILE_PATH)}")
print(f"Sample Rate         : {sample_rate} Hz ({duration_sec:.3f} seconds)")
print(f"Channel Analyzed    : {channel_desc}")
print(f"Input Signal RMS    : {input_rms:.6f}")
print("-" * 75)

header = f"{'Algorithm':<12} | {'Filter Length':<13} | {'Step Size':<11} | {'Error RMS':<12} | {'RMS Reduction (dB)':<18}"
print(header)
print("-" * 75)
print(f"{'LMS':<12} | {FILTER_LENGTH:<13} | {LMS_MU:<11.4f} | {lms_error_rms:<12.6f} | {lms_reduction_db:>14.2f} dB")
print(f"{'NLMS':<12} | {FILTER_LENGTH:<13} | {NLMS_MU:<11.4f} | {nlms_error_rms:<12.6f} | {nlms_reduction_db:>14.2f} dB")
print("-" * 75)

# --- 9. Short Interpretation ---
print("\n[EXPERIMENT INTERPRETATION]")
if nlms_error_rms < lms_error_rms:
    print(f"1. Error Level : NLMS achieved a lower residual error RMS ({nlms_error_rms:.6f} vs {lms_error_rms:.6f}, +{nlms_reduction_db - lms_reduction_db:.2f} dB better).")
else:
    print(f"1. Error Level : LMS achieved a lower residual error RMS ({lms_error_rms:.6f} vs {nlms_error_rms:.6f}).")

print("2. Stability   : NLMS dynamically scales its step size by the input energy, making it inherently")
print("                 more robust against signal amplitude fluctuations and sudden acoustic bursts.")
print("3. Caveat      : These observations apply specifically to this recording configuration.")
print("                 Universal conclusions should not be drawn from a single audio sample.")
print("-" * 75)
print("[IMPORTANT DISCLAIMER]")
print("This is an offline DSP experiment. Results are recording-dependent and do not")
print("represent final physical ANC performance.")
print("=" * 75)

# --- 10. Generate and Save Comparison Plot ---
time_axis = np.linspace(0.0, duration_sec, num_samples, endpoint=False)

plt.figure(figsize=(12, 7))

# Subplot 1: LMS Error Signal
plt.subplot(2, 1, 1)
plt.plot(time_axis, e_lms, color="#e67e22", linewidth=0.7, label=f"LMS Error (Reduction: {lms_reduction_db:.2f} dB)")
plt.title(f"LMS Error Signal e[n] (Taps={FILTER_LENGTH}, \u03bc={LMS_MU})", fontsize=11, fontweight="bold")
plt.ylabel("Amplitude", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="upper right")

# Subplot 2: NLMS Error Signal
plt.subplot(2, 1, 2)
plt.plot(time_axis, e_nlms, color="#27ae60", linewidth=0.7, label=f"NLMS Error (Reduction: {nlms_reduction_db:.2f} dB)")
plt.title(f"NLMS Error Signal e[n] (Taps={FILTER_LENGTH}, \u03bc_norm={NLMS_MU}, \u03b5={NLMS_EPSILON})", fontsize=11, fontweight="bold")
plt.xlabel("Time (seconds)", fontsize=10)
plt.ylabel("Amplitude", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="upper right")

plt.tight_layout()
plt.savefig(OUTPUT_PLOT_PATH, dpi=300)
plt.close()

print(f"\n[SUCCESS] Comparison plot saved to: {OUTPUT_PLOT_PATH}")
print("=" * 75)
