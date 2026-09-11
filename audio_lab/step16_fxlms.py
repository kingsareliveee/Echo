# ==============================================================================
# Project: AI/ML-enabled Adaptive Noise Cancellation (ANC) for Defence Vehicles
# PS ID: SIH26052 | Team: Echo Shield | Theme: Smart Vehicles
# 
# Step 16: Filtered-X Least Mean Squares (FxLMS) Algorithm
# ==============================================================================

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# ==============================================================================
# BEGINNER-FRIENDLY FxLMS DSP CONCEPTS & MATHEMATICS:
#
# 1. WHAT IS FxLMS (FILTERED-X LMS)?
#    Standard LMS fails in physical ANC because the secondary path S(z) (D/A, amp,
#    speaker, air travel) delays the anti-noise before it reaches the error microphone.
#    FxLMS solves this by passing the reference signal x[n] through an estimate of the
#    secondary path S_hat(z) to create a "Filtered Reference Signal" x_f[n]:
#        x_f[n] = sum_{j=0}^{M-1} s[j] * x[n - j]
#
# 2. ADAPTIVE WEIGHT UPDATE FORMULA:
#        y[n]   = sum_{k=0}^{L-1} w[k] * x[n - 1 - k]   (Anti-noise command from DSP)
#        y_s[n] = sum_{j=0}^{M-1} s[j] * y[n - j]       (Anti-noise arriving at ear)
#        e[n]   = d[n] - y_s[n]                         (Residual acoustic error)
#        w[n+1] = w[n] + \u03bc * e[n] * x_f_vector
# ==============================================================================

# --- 1. Configuration Constants ---
FILTER_LENGTH = 32     # Adaptive FIR filter length (taps)
STEP_SIZE_MU = 0.0005  # Conservative step size for stable FxLMS convergence
SEC_PATH_LEN = 12      # Secondary path FIR length
SEC_DELAY = 2          # Secondary path propagation delay in samples

# Synthetic Secondary Path FIR (identical to Step 15)
synthetic_sec_path = np.zeros(SEC_PATH_LEN, dtype=np.float64)
for i in range(SEC_DELAY, SEC_PATH_LEN):
    synthetic_sec_path[i] = 0.8 * (0.6 ** (i - SEC_DELAY)) * np.cos(0.5 * (i - SEC_DELAY))
synthetic_sec_path = synthetic_sec_path / np.sum(np.abs(synthetic_sec_path))

# --- 2. Define File Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(OUTPUTS_DIR, exist_ok=True)

INPUT_FILE_PATH = os.path.join(DATA_DIR, "tank.wav")
OUTPUT_WAV_PATH = os.path.join(OUTPUTS_DIR, "tank_fxlms_error.wav")
OUTPUT_PLOT_PATH = os.path.join(OUTPUTS_DIR, "step16_fxlms.png")

print("=" * 75)
print("  ECHO SHIELD - Step 16: Filtered-X LMS (FxLMS) Algorithm")
print("=" * 75)

# --- 3. Load and Normalize Audio File ---
if not os.path.isfile(INPUT_FILE_PATH):
    print(f"\n[ERROR] Audio file not found at: {INPUT_FILE_PATH}")
    sys.exit(1)

sample_rate, audio_data = wavfile.read(INPUT_FILE_PATH)
if audio_data.ndim > 1:
    channel_signal = audio_data[:, 0]
else:
    channel_signal = audio_data

raw_max = np.max(np.abs(channel_signal))
if raw_max > 0:
    norm_factor = 32768.0 if np.issubdtype(channel_signal.dtype, np.int16) else float(raw_max)
    signal = channel_signal.astype(np.float64) / norm_factor
else:
    signal = channel_signal.astype(np.float64)

num_samples = len(signal)
duration_sec = num_samples / sample_rate

# --- 4. Implement FxLMS Sample-by-Sample Loop ---
weights = np.zeros(FILTER_LENGTH, dtype=np.float64)

# Buffers
x_ref_buf = np.zeros(FILTER_LENGTH, dtype=np.float64)       # Reference buffer for W(z)
x_filt_buf = np.zeros(FILTER_LENGTH, dtype=np.float64)      # Filtered-X buffer for weight update
sec_x_history = np.zeros(SEC_PATH_LEN, dtype=np.float64)    # Raw reference history for S(z)
y_history = np.zeros(SEC_PATH_LEN, dtype=np.float64)        # Filter output history for S(z)

e_error = np.zeros(num_samples, dtype=np.float64)
y_anti = np.zeros(num_samples, dtype=np.float64)

for n in range(num_samples):
    x_n = signal[n]  # Reference microphone sample
    d_n = signal[n]  # Primary noise sample at error microphone
    
    # Update raw reference history for secondary path filtering
    sec_x_history[1:] = sec_x_history[:-1]
    sec_x_history[0] = x_n
    
    # 1. Compute Filtered-X sample: x_f[n] = S_hat(z) * x[n]
    x_f_n = np.dot(synthetic_sec_path, sec_x_history)
    
    # Update Filtered-X buffer
    x_filt_buf[1:] = x_filt_buf[:-1]
    x_filt_buf[0] = x_f_n
    
    # 2. Compute adaptive filter anti-noise command: y[n] = W(z) * x[n]
    y_n = np.dot(weights, x_ref_buf)
    y_anti[n] = y_n
    
    # Update anti-noise history for secondary path convolution
    y_history[1:] = y_history[:-1]
    y_history[0] = y_n
    
    # 3. Anti-noise arriving at error sensor after secondary path S(z): y_s[n] = S(z) * y[n]
    y_s_n = np.dot(synthetic_sec_path, y_history)
    
    # 4. Residual error at sensor: e[n] = d[n] - y_s[n]
    e_n = d_n - y_s_n
    e_error[n] = e_n
    
    # 5. FxLMS Weight Update: w[n+1] = w[n] + mu * e[n] * x_f_vector
    weights += STEP_SIZE_MU * e_n * x_filt_buf
    
    # Update reference buffer
    x_ref_buf[1:] = x_ref_buf[:-1]
    x_ref_buf[0] = x_n

# --- 5. Calculate Metrics ---
input_rms = float(np.sqrt(np.mean(signal ** 2)))
error_rms = float(np.sqrt(np.mean(e_error ** 2)))
rms_reduction_db = 20.0 * np.log10(input_rms / error_rms) if error_rms > 0 else 0.0

# --- 6. Print Report ---
print(f"File Analyzed             : {os.path.basename(INPUT_FILE_PATH)}")
print(f"Sample Rate               : {sample_rate} Hz ({duration_sec:.3f} seconds)")
print(f"Adaptive Filter Taps (L)  : {FILTER_LENGTH}")
print(f"Step Size (mu)            : {STEP_SIZE_MU}")
print(f"Secondary Path Taps (M)   : {SEC_PATH_LEN}")
print("-" * 75)
print(f"Input Signal RMS          : {input_rms:.6f}")
print(f"Residual Error RMS        : {error_rms:.6f}")
print(f"Simulated RMS Reduction   : {rms_reduction_db:.2f} dB")
print("=" * 75)

# --- 7. Save Error WAV File ---
clipped_error = np.clip(e_error, -1.0, 1.0)
error_wav = (clipped_error * 32767.0).astype(np.int16)
wavfile.write(OUTPUT_WAV_PATH, sample_rate, error_wav)
print(f"[SUCCESS] FxLMS residual error audio saved to: {OUTPUT_WAV_PATH}")

# --- 8. Plot and Save Comparison ---
time_axis = np.linspace(0.0, duration_sec, num_samples, endpoint=False)

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time_axis, signal, color="#2980b9", linewidth=0.7)
plt.title(f"Primary Noise Signal d[n] ({os.path.basename(INPUT_FILE_PATH)})", fontsize=11, fontweight="bold")
plt.ylabel("Amplitude", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(2, 1, 2)
plt.plot(time_axis, e_error, color="#27ae60", linewidth=0.7)
plt.title(f"FxLMS Residual Error e[n] (Simulated Reduction: {rms_reduction_db:.2f} dB)", fontsize=11, fontweight="bold")
plt.xlabel("Time (seconds)", fontsize=10)
plt.ylabel("Amplitude", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT_PATH, dpi=300)
plt.close()

print(f"[SUCCESS] FxLMS comparison plot saved to: {OUTPUT_PLOT_PATH}")
print("-" * 75)
print("[IMPORTANT DISCLAIMER]")
print("This is a simplified software simulation demonstrating the FxLMS algorithm.")
print("It does NOT represent physical in-vehicle acoustic cancellation.")
print("=" * 75)
