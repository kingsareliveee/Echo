# ==============================================================================
# Project: AI/ML-enabled Adaptive Noise Cancellation (ANC) for Defence Vehicles
# PS ID: SIH26052 | Team: Echo Shield | Theme: Smart Vehicles
# 
# Step 22: Complete Software Prototype Final End-to-End Demo
# ==============================================================================

import os
import sys
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import pywt
# pyrefly: ignore [missing-import]
import joblib
from scipy.io import wavfile

# --- 1. Configuration Constants ---
WINDOW_DURATION_MS = 100
HOP_DURATION_MS = 50

BANDS = [
    ("band_energy_0_500", 0.0, 500.0),
    ("band_energy_500_2000", 500.0, 2000.0),
    ("band_energy_2000_5000", 2000.0, 5000.0),
    ("band_energy_5000_10000", 5000.0, 10000.0)
]

DSP_PROFILES = {
    "tank": {
        "profile": "LOW_FREQ_MECHANICAL_PROFILE",
        "filter_length": 32,
        "step_size_mu": 0.005,
        "processing_mode": "LOW_FREQUENCY_ADAPTIVE"
    },
    "helicopter": {
        "profile": "LOW_FREQ_TONAL_PROFILE",
        "filter_length": 32,
        "step_size_mu": 0.003,
        "processing_mode": "LOW_FREQ_TONAL_ADAPTIVE"
    },
    "gun_fight": {
        "profile": "IMPULSIVE_PROFILE",
        "filter_length": 32,
        "step_size_mu": 0.001,
        "processing_mode": "IMPULSIVE_PROTECTION"
    }
}

# --- 2. Define File Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_PATH = os.path.join(OUTPUTS_DIR, "noise_classifier.joblib")
OUTPUT_PLOT_PATH = os.path.join(OUTPUTS_DIR, "step22_final_demo.png")

os.makedirs(OUTPUTS_DIR, exist_ok=True)

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

filename = os.path.basename(INPUT_FILE_PATH)

# --- 3. Input Validation & Model Loading ---
if not os.path.isfile(INPUT_FILE_PATH):
    print(f"\n[ERROR] Audio file not found at: {INPUT_FILE_PATH}")
    sys.exit(1)

if not os.path.isfile(MODEL_PATH):
    print(f"\n[ERROR] Classifier model not found at: {MODEL_PATH}")
    sys.exit(1)

model = joblib.load(MODEL_PATH)

# --- 4. Load and Prepare Audio ---
sample_rate, audio_data = wavfile.read(INPUT_FILE_PATH)
if audio_data.ndim > 1:
    ch1 = audio_data[:, 0]
else:
    ch1 = audio_data

# Unnormalized raw scale for feature extraction (matching Steps 04-08)
raw_signal = ch1.astype(np.float64)

# Normalized [-1.0, 1.0] scale for DSP arithmetic (matching Steps 12-16)
raw_max = np.max(np.abs(ch1))
if raw_max > 0:
    norm_factor = 32768.0 if np.issubdtype(ch1.dtype, np.integer) else float(raw_max)
    signal = ch1.astype(np.float64) / norm_factor
else:
    signal = ch1.astype(np.float64)

num_samples = len(signal)
duration_sec = num_samples / sample_rate
orig_rms = float(np.sqrt(np.mean(signal ** 2)))

# --- 5. Extract Windowed Features & Classify ---
window_samples = int(sample_rate * (WINDOW_DURATION_MS / 1000.0))
hop_samples = int(sample_rate * (HOP_DURATION_MS / 1000.0))
half_n = window_samples // 2
freq_axis = np.linspace(0.0, sample_rate / 2.0, half_n, endpoint=False)

feature_rows = []
start = 0
while start + window_samples <= num_samples:
    w_sig = raw_signal[start:start + window_samples]
    rms = float(np.sqrt(np.mean(w_sig ** 2)))
    zcr = float(np.sum(np.abs(np.diff(np.signbit(w_sig)))) / (window_samples - 1)) if window_samples > 1 else 0.0
    
    raw_fft = np.fft.fft(w_sig)
    mag = (2.0 / window_samples) * np.abs(raw_fft[:half_n])
    if len(mag) > 0:
        mag[0] /= 2.0
    tot_mag = float(np.sum(mag))
    centroid = float(np.sum(freq_axis * mag) / tot_mag) if tot_mag > 0 else 0.0
    
    if tot_mag > 0:
        cum_mag = np.cumsum(mag)
        r_idx = min(int(np.searchsorted(cum_mag, 0.85 * tot_mag)), len(freq_axis) - 1)
        rolloff = float(freq_axis[r_idx])
    else:
        rolloff = 0.0
        
    b_energies = []
    for _, fl, fh in BANDS:
        if fl < sample_rate / 2.0:
            fh_act = min(fh, sample_rate / 2.0)
            b_energies.append(float(np.sum(mag[(freq_axis >= fl) & (freq_axis < fh_act)])))
        else:
            b_energies.append(0.0)
            
    feature_rows.append([rms, zcr, centroid, rolloff] + b_energies)
    start += hop_samples

X_feat = np.array(feature_rows, dtype=np.float64)
preds = model.predict(X_feat)
counts = Counter(preds)
majority_class = counts.most_common(1)[0][0]
pct_winner = (counts[majority_class] / len(preds)) * 100.0

dsp_cfg = DSP_PROFILES.get(majority_class, {
    "profile": "DEFAULT_PROFILE",
    "filter_length": 32,
    "step_size_mu": 0.001,
    "processing_mode": "DEFAULT_MODE"
})

# --- 6. Adaptive Filtering Benchmark (LMS & NLMS) ---
# LMS
w_lms = np.zeros(dsp_cfg["filter_length"], dtype=np.float64)
b_lms = np.zeros(dsp_cfg["filter_length"], dtype=np.float64)
e_lms = np.zeros(num_samples, dtype=np.float64)
for n in range(num_samples):
    err = signal[n] - np.dot(w_lms, b_lms)
    e_lms[n] = err
    w_lms += dsp_cfg["step_size_mu"] * err * b_lms
    b_lms[1:] = b_lms[:-1]; b_lms[0] = signal[n]
lms_rms = float(np.sqrt(np.mean(e_lms ** 2)))
lms_red = 20.0 * np.log10(orig_rms / lms_rms) if lms_rms > 0 else 0.0

# NLMS
w_nlms = np.zeros(dsp_cfg["filter_length"], dtype=np.float64)
b_nlms = np.zeros(dsp_cfg["filter_length"], dtype=np.float64)
e_nlms = np.zeros(num_samples, dtype=np.float64)
for n in range(num_samples):
    err = signal[n] - np.dot(w_nlms, b_nlms)
    e_nlms[n] = err
    norm_mu = 0.5 / (1e-8 + float(np.dot(b_nlms, b_nlms)))
    w_nlms += norm_mu * err * b_nlms
    b_nlms[1:] = b_nlms[:-1]; b_nlms[0] = signal[n]
nlms_rms = float(np.sqrt(np.mean(e_nlms ** 2)))
nlms_red = 20.0 * np.log10(orig_rms / nlms_rms) if nlms_rms > 0 else 0.0

# --- 7. FxLMS Simulation Path ---
sec_path = np.zeros(12, dtype=np.float64)
for i in range(2, 12):
    sec_path[i] = 0.8 * (0.6 ** (i - 2)) * np.cos(0.5 * (i - 2))
sec_path = sec_path / np.sum(np.abs(sec_path))

w_fx = np.zeros(32, dtype=np.float64)
x_buf = np.zeros(32, dtype=np.float64)
xf_buf = np.zeros(32, dtype=np.float64)
sec_x_hist = np.zeros(12, dtype=np.float64)
y_hist = np.zeros(12, dtype=np.float64)
e_fx = np.zeros(num_samples, dtype=np.float64)

for n in range(num_samples):
    sec_x_hist[1:] = sec_x_hist[:-1]; sec_x_hist[0] = signal[n]
    xf_n = np.dot(sec_path, sec_x_hist)
    xf_buf[1:] = xf_buf[:-1]; xf_buf[0] = xf_n
    
    y_n = np.dot(w_fx, x_buf)
    y_hist[1:] = y_hist[:-1]; y_hist[0] = y_n
    ys_n = np.dot(sec_path, y_hist)
    
    err = signal[n] - ys_n
    e_fx[n] = err
    w_fx += 0.0005 * err * xf_buf
    x_buf[1:] = x_buf[:-1]; x_buf[0] = signal[n]

fx_rms = float(np.sqrt(np.mean(e_fx ** 2)))
fx_red = 20.0 * np.log10(orig_rms / fx_rms) if fx_rms > 0 else 0.0

# --- 8. Wavelet Denoising Path ---
coeffs = pywt.wavedec(signal, "db4", level=min(5, pywt.dwt_max_level(num_samples, pywt.Wavelet("db4").dec_len)))
sigma = np.median(np.abs(coeffs[-1])) / 0.6745
t_thresh = sigma * np.sqrt(2.0 * np.log(num_samples))
denoised_sig = pywt.waverec([coeffs[0]] + [pywt.threshold(c, t_thresh, "soft") for c in coeffs[1:]], "db4")[:num_samples]
denoised_rms = float(np.sqrt(np.mean(denoised_sig ** 2)))

# --- 9. Clean Final Report ---
print("\n" + "=" * 60)
print("ECHO SHIELD - SOFTWARE PROTOTYPE FINAL DEMO")
print("=" * 60)
print(f"Input           : {filename} ({duration_sec:.2f}s, {sample_rate} Hz)")
print(f"Noise Class     : {majority_class}")
print(f"Noise Profile   : {dsp_cfg['profile']}")
print(f"DSP Mode        : {dsp_cfg['processing_mode']}")
print("\nML:")
print(f"Prediction      : {majority_class} ({pct_winner:.1f}% majority confidence)")
print("\nDSP:")
print(f"Filter Length   : {dsp_cfg['filter_length']} Taps")
print(f"Step Size       : mu = {dsp_cfg['step_size_mu']}")
print("\nAdaptive Filtering:")
print(f"LMS Result      : RMS = {lms_rms:.6f} (Reduction: {lms_red:.2f} dB)")
print(f"NLMS Result     : RMS = {nlms_rms:.6f} (Reduction: {nlms_red:.2f} dB)")
print("\nFxLMS Simulation:")
print(f"Original RMS    : {orig_rms:.6f}")
print(f"Residual RMS    : {fx_rms:.6f}")
print(f"Simulated RMS Reduction: {fx_red:.2f} dB")
print("\nWavelet:")
print(f"Denoising Output: RMS = {denoised_rms:.6f} (db4 Soft Threshold, Level 5)")
print("\n" + "=" * 60)
print("IMPORTANT:")
print("This final demo validates the software/algorithmic pipeline only.")
print("Physical ANC performance requires microphone/ADC/DAC/speaker/error-microphone")
print("hardware and a measured secondary path.")
print("=" * 60)

# --- 10. Presentation-Quality Multi-Panel Plot ---
t_axis = np.linspace(0.0, duration_sec, num_samples, endpoint=False)

plt.figure(figsize=(13, 9))

# Panel 1: Original Input Noise
plt.subplot(4, 1, 1)
plt.plot(t_axis, signal, color="#2980b9", linewidth=0.7)
plt.title(f"1. Ingested Noise Waveform: {filename} (Classified: '{majority_class}', Profile: '{dsp_cfg['profile']}')", fontsize=10, fontweight="bold")
plt.ylabel("Amplitude", fontsize=9)
plt.grid(True, linestyle="--", alpha=0.5)

# Panel 2: NLMS Error
plt.subplot(4, 1, 2)
plt.plot(t_axis, e_nlms, color="#27ae60", linewidth=0.7)
plt.title(f"2. DSP Adaptive Filtering: NLMS Error Signal (RMS Reduction: {nlms_red:.2f} dB)", fontsize=10, fontweight="bold")
plt.ylabel("Amplitude", fontsize=9)
plt.grid(True, linestyle="--", alpha=0.5)

# Panel 3: FxLMS ANC Simulation
plt.subplot(4, 1, 3)
plt.plot(t_axis, e_fx, color="#e67e22", linewidth=0.7)
plt.title(f"3. Active Noise Cancellation Path: FxLMS Simulated Residual Error (Reduction: {fx_red:.2f} dB)", fontsize=10, fontweight="bold")
plt.ylabel("Amplitude", fontsize=9)
plt.grid(True, linestyle="--", alpha=0.5)

# Panel 4: Wavelet Denoising
plt.subplot(4, 1, 4)
plt.plot(t_axis, denoised_sig, color="#8e44ad", linewidth=0.7)
plt.title(f"4. Communication Path: Wavelet Denoised & Reconstructed Signal (db4 Wavelet)", fontsize=10, fontweight="bold")
plt.xlabel("Time (seconds)", fontsize=9)
plt.ylabel("Amplitude", fontsize=9)
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT_PATH, dpi=300)
plt.close()

print(f"\n[SUCCESS] Final demo visualization saved to: {OUTPUT_PLOT_PATH}")
print("=" * 60)
