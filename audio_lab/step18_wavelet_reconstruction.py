# ==============================================================================
# Project: AI/ML-enabled Adaptive Noise Cancellation (ANC) for Defence Vehicles
# PS ID: SIH26052 | Team: Echo Shield | Theme: Smart Vehicles
# 
# Step 18: Wavelet-Based Denoising & Signal Reconstruction
# ==============================================================================

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.io import wavfile

# ==============================================================================
# BEGINNER-FRIENDLY CONCEPT: WAVELET RECONSTRUCTION FOR SPEECH & AUDIO
#
# 1. WHAT IS DISCRETE WAVELET TRANSFORM (DWT)?
#    Unlike Fourier transforms (which use infinite sine waves), Wavelets are localized
#    mini-waves that simultaneously provide excellent time AND frequency localization.
#
# 2. HOW WAVELET THRESHOLDING WORKS:
#    - DECOMPOSE : Signal is split into approximation (low-freq) and detail (high-freq)
#                  coefficients across multiple octaves (scales).
#    - THRESHOLD : Noise energy is scattered across many small coefficients, while speech
#                  and harmonic features produce large peaks. Soft-thresholding zeroes
#                  out small noise coefficients without blurring sharp acoustic edges.
#    - RECONSTRUCT: Inverse DWT (IDWT) reconstructs the cleaned time-domain signal.
# ==============================================================================

# --- 1. Define File Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(OUTPUTS_DIR, exist_ok=True)

INPUT_FILE_PATH = os.path.join(DATA_DIR, "tank.wav")
OUTPUT_WAV_PATH = os.path.join(OUTPUTS_DIR, "tank_wavelet_denoised.wav")
OUTPUT_PLOT_PATH = os.path.join(OUTPUTS_DIR, "step18_wavelet_comparison.png")

print("=" * 75)
print("  ECHO SHIELD - Step 18: Wavelet Denoising & Reconstruction")
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

# --- 3. Wavelet Decomposition & Soft Thresholding ---
wavelet_name = "db4"  # Daubechies-4 wavelet
wavelet = pywt.Wavelet(wavelet_name)
max_level = pywt.dwt_max_level(num_samples, wavelet.dec_len)
level = min(5, max_level)

# Decompose signal into wavelet coefficients: [cA_n, cD_n, cD_n-1, ..., cD_1]
coeffs = pywt.wavedec(signal, wavelet_name, level=level)

# Calculate robust Universal Threshold (Donoho & Johnstone / VisuShrink)
# sigma estimated from median absolute deviation (MAD) of the finest detail coefficients (cD1)
detail_cd1 = coeffs[-1]
sigma = np.median(np.abs(detail_cd1)) / 0.6745
threshold = sigma * np.sqrt(2.0 * np.log(num_samples))

# Apply soft thresholding to detail coefficients only (leave approximation cA intact)
thresholded_coeffs = [coeffs[0]]
for i in range(1, len(coeffs)):
    thresholded_detail = pywt.threshold(coeffs[i], threshold, mode="soft")
    thresholded_coeffs.append(thresholded_detail)

# Reconstruct signal via Inverse Discrete Wavelet Transform (IDWT)
denoised_signal = pywt.waverec(thresholded_coeffs, wavelet_name)

# Ensure reconstructed length matches original length
if len(denoised_signal) > num_samples:
    denoised_signal = denoised_signal[:num_samples]
elif len(denoised_signal) < num_samples:
    denoised_signal = np.pad(denoised_signal, (0, num_samples - len(denoised_signal)))

# --- 4. Calculate Metrics ---
input_rms = float(np.sqrt(np.mean(signal ** 2)))
denoised_rms = float(np.sqrt(np.mean(denoised_signal ** 2)))
reduction_db = 20.0 * np.log10(input_rms / denoised_rms) if denoised_rms > 0 else 0.0

print(f"File Analyzed           : {os.path.basename(INPUT_FILE_PATH)}")
print(f"Wavelet Used            : {wavelet_name.upper()} (Decomposition Level = {level})")
print(f"Computed Noise Sigma    : {sigma:.6f}")
print(f"Soft Threshold (T)      : {threshold:.6f}")
print("-" * 75)
print(f"Input Signal RMS        : {input_rms:.6f}")
print(f"Denoised Signal RMS     : {denoised_rms:.6f}")
print(f"Magnitude Modification  : {reduction_db:.2f} dB")
print("-" * 75)
print("[ARCHITECTURAL EXPLANATION]")
print("Wavelet denoising -> suppresses high-entropy noise sub-bands -> reconstructs signal.")
print("\n[IMPORTANT DISCLAIMER]")
print("This is a signal-processing demonstration.")
print("It does NOT prove improved speech intelligibility unless evaluated with actual")
print("speech recordings and standardized objective metrics (e.g. STOI/PESQ).")
print("=" * 75)

# --- 5. Save Denoised Audio ---
denoised_wav = (np.clip(denoised_signal, -1.0, 1.0) * 32767.0).astype(np.int16)
wavfile.write(OUTPUT_WAV_PATH, sample_rate, denoised_wav)
print(f"[SUCCESS] Denoised audio saved to: {OUTPUT_WAV_PATH}")

# --- 6. Plot and Save Comparison ---
time_axis = np.linspace(0.0, duration_sec, num_samples, endpoint=False)

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time_axis, signal, color="#2980b9", linewidth=0.7)
plt.title(f"Input Signal Waveform ({os.path.basename(INPUT_FILE_PATH)})", fontsize=11, fontweight="bold")
plt.ylabel("Amplitude", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(2, 1, 2)
plt.plot(time_axis, denoised_signal, color="#16a085", linewidth=0.7)
plt.title(f"Wavelet Denoised & Reconstructed Signal ({wavelet_name.upper()}, Level {level})", fontsize=11, fontweight="bold")
plt.xlabel("Time (seconds)", fontsize=10)
plt.ylabel("Amplitude", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT_PATH, dpi=300)
plt.close()

print(f"[SUCCESS] Wavelet comparison plot saved to: {OUTPUT_PLOT_PATH}")
print("=" * 75)
