# ==============================================================================
# Project: AI/ML-enabled Adaptive Noise Cancellation (ANC) for Defence Vehicles
# PS ID: SIH26052 | Team: Echo Shield | Theme: Smart Vehicles
# 
# Step 17: End-to-End Offline ANC Simulation
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
ORIGINAL_WAV_PATH = os.path.join(OUTPUTS_DIR, "step17_anc_original.wav")
RESIDUAL_WAV_PATH = os.path.join(OUTPUTS_DIR, "step17_anc_residual.wav")
OUTPUT_PLOT_PATH = os.path.join(OUTPUTS_DIR, "step17_anc_simulation.png")

print("=" * 75)
print("  ECHO SHIELD - Step 17: Offline ANC Simulation")
print("=" * 75)

# --- 2. Load and Normalize Audio ---
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

# --- 3. Simulation Parameters ---
FILTER_LENGTH = 32
STEP_SIZE_MU = 0.0005
SEC_PATH_LEN = 12
SEC_DELAY = 2

# Synthetic Secondary Path Model
synthetic_sec_path = np.zeros(SEC_PATH_LEN, dtype=np.float64)
for i in range(SEC_DELAY, SEC_PATH_LEN):
    synthetic_sec_path[i] = 0.8 * (0.6 ** (i - SEC_DELAY)) * np.cos(0.5 * (i - SEC_DELAY))
synthetic_sec_path = synthetic_sec_path / np.sum(np.abs(synthetic_sec_path))

# --- 4. Run FxLMS Simulation Loop ---
weights = np.zeros(FILTER_LENGTH, dtype=np.float64)
x_ref_buf = np.zeros(FILTER_LENGTH, dtype=np.float64)
x_filt_buf = np.zeros(FILTER_LENGTH, dtype=np.float64)
sec_x_history = np.zeros(SEC_PATH_LEN, dtype=np.float64)
y_history = np.zeros(SEC_PATH_LEN, dtype=np.float64)

e_error = np.zeros(num_samples, dtype=np.float64)
anti_noise = np.zeros(num_samples, dtype=np.float64)

for n in range(num_samples):
    x_n = signal[n]
    d_n = signal[n]
    
    sec_x_history[1:] = sec_x_history[:-1]
    sec_x_history[0] = x_n
    x_f_n = np.dot(synthetic_sec_path, sec_x_history)
    
    x_filt_buf[1:] = x_filt_buf[:-1]
    x_filt_buf[0] = x_f_n
    
    y_n = np.dot(weights, x_ref_buf)
    anti_noise[n] = y_n
    
    y_history[1:] = y_history[:-1]
    y_history[0] = y_n
    y_s_n = np.dot(synthetic_sec_path, y_history)
    
    e_n = d_n - y_s_n
    e_error[n] = e_n
    
    weights += STEP_SIZE_MU * e_n * x_filt_buf
    x_ref_buf[1:] = x_ref_buf[:-1]
    x_ref_buf[0] = x_n

# --- 5. Quantitative Metrics ---
orig_rms = float(np.sqrt(np.mean(signal ** 2)))
res_rms = float(np.sqrt(np.mean(e_error ** 2)))
red_db = 20.0 * np.log10(orig_rms / res_rms) if res_rms > 0 else 0.0

print(f"File Analyzed           : {os.path.basename(INPUT_FILE_PATH)}")
print(f"Duration                : {duration_sec:.3f} seconds ({num_samples:,} samples)")
print(f"Original Signal RMS     : {orig_rms:.6f}")
print(f"Residual (Error) RMS    : {res_rms:.6f}")
print(f"Simulated RMS Reduction : {red_db:.2f} dB")
print("-" * 75)
print("SIMULATION ONLY \u2014 secondary path is synthetic and not measured hardware data.")
print("=" * 75)

# --- 6. Save Audio Files ---
orig_wav = (np.clip(signal, -1.0, 1.0) * 32767.0).astype(np.int16)
res_wav = (np.clip(e_error, -1.0, 1.0) * 32767.0).astype(np.int16)

wavfile.write(ORIGINAL_WAV_PATH, sample_rate, orig_wav)
wavfile.write(RESIDUAL_WAV_PATH, sample_rate, res_wav)

print(f"[SUCCESS] Original audio saved to : {ORIGINAL_WAV_PATH}")
print(f"[SUCCESS] Residual audio saved to : {RESIDUAL_WAV_PATH}")

# --- 7. Plot and Save Simulation Results ---
time_axis = np.linspace(0.0, duration_sec, num_samples, endpoint=False)

plt.figure(figsize=(12, 7))

plt.subplot(3, 1, 1)
plt.plot(time_axis, signal, color="#2980b9", linewidth=0.7)
plt.title("Original Primary Disturbance d[n] (Tank Engine Noise)", fontsize=11, fontweight="bold")
plt.ylabel("Amplitude", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(3, 1, 2)
plt.plot(time_axis, anti_noise, color="#e67e22", linewidth=0.7)
plt.title("FxLMS Anti-Noise Output Command y[n]", fontsize=11, fontweight="bold")
plt.ylabel("Amplitude", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(3, 1, 3)
plt.plot(time_axis, e_error, color="#27ae60", linewidth=0.7)
plt.title(f"Residual Error e[n] After Simulated ANC (Reduction: {red_db:.2f} dB)", fontsize=11, fontweight="bold")
plt.xlabel("Time (seconds)", fontsize=10)
plt.ylabel("Amplitude", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT_PATH, dpi=300)
plt.close()

print(f"[SUCCESS] Simulation plot saved to: {OUTPUT_PLOT_PATH}")
print("=" * 75)
