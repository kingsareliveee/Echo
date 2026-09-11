# ==============================================================================
# Project: AI/ML-enabled Adaptive Noise Cancellation (ANC) for Defence Vehicles
# PS ID: SIH26052 | Team: Echo Shield | Theme: Smart Vehicles
# 
# Step 19: Full Dual-Pipeline Integration (ML + DSP + Wavelet + ANC Simulation)
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
OUTPUT_PLOT_PATH = os.path.join(OUTPUTS_DIR, "step19_full_pipeline.png")

os.makedirs(OUTPUTS_DIR, exist_ok=True)

TARGET_FILES = ["tank.wav", "heli.wav", "gun_fight.wav", "test.wav"]

print("=" * 80)
print("  ECHO SHIELD - Step 19: Full Dual-Pipeline Software Integration")
print("=" * 80)

# --- 3. Check and Load ML Model ---
if not os.path.isfile(MODEL_PATH):
    print(f"\n[ERROR] Trained classifier model not found at: {MODEL_PATH}")
    sys.exit(1)

model = joblib.load(MODEL_PATH)

# --- 4. Helper Function: Extract 8 Features for Windows ---
def extract_features(signal, sample_rate):
    total_samples = len(signal)
    nyquist_freq = sample_rate / 2.0
    window_samples = int(sample_rate * (WINDOW_DURATION_MS / 1000.0))
    hop_samples = int(sample_rate * (HOP_DURATION_MS / 1000.0))
    
    if total_samples < window_samples:
        return None
        
    half_n = window_samples // 2
    freq_axis = np.linspace(0.0, nyquist_freq, half_n, endpoint=False)
    
    rows = []
    start = 0
    while start + window_samples <= total_samples:
        w_sig = signal[start:start + window_samples]
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
            if fl < nyquist_freq:
                fh_act = min(fh, nyquist_freq)
                b_energies.append(float(np.sum(mag[(freq_axis >= fl) & (freq_axis < fh_act)])))
            else:
                b_energies.append(0.0)
                
        rows.append([rms, zcr, centroid, rolloff] + b_energies)
        start += hop_samples
        
    return np.array(rows, dtype=np.float64)

# --- 5. Process Each Audio File Through the Full Architecture ---
file_reports = []
tank_sim_data = {}

for filename in TARGET_FILES:
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.isfile(filepath):
        continue
        
    sr, raw_data = wavfile.read(filepath)
    ch1 = raw_data[:, 0] if raw_data.ndim > 1 else raw_data
    
    # Feature extraction uses unnormalized raw sample scale (matching Steps 04-08)
    raw_signal = ch1.astype(np.float64)
    
    # DSP & simulation uses normalized [-1.0, 1.0] float scale (matching Steps 12-16)
    raw_max = np.max(np.abs(ch1))
    if raw_max > 0:
        norm_factor = 32768.0 if np.issubdtype(ch1.dtype, np.integer) else float(raw_max)
        norm_signal = ch1.astype(np.float64) / norm_factor
    else:
        norm_signal = ch1.astype(np.float64)
        
    X_feat = extract_features(raw_signal, sr)
    if X_feat is None:
        continue
        
    preds = model.predict(X_feat)
    counts = Counter(preds)
    majority_class = counts.most_common(1)[0][0]
    total_w = len(preds)
    
    cfg = DSP_PROFILES.get(majority_class, {
        "profile": "DEFAULT_PROFILE",
        "filter_length": 32,
        "step_size_mu": 0.001,
        "processing_mode": "DEFAULT_MODE"
    })
    
    distribution_str = ", ".join([f"{c}: {(counts.get(c, 0)/total_w)*100.0:.1f}%" for c in sorted(model.classes_)])
    
    file_reports.append({
        "filename": filename,
        "class": majority_class,
        "profile": cfg["profile"],
        "mode": cfg["processing_mode"],
        "filter_len": cfg["filter_length"],
        "mu": cfg["step_size_mu"],
        "dist": distribution_str
    })
    
    # Run detailed dual-path for tank.wav as reference demonstration
    if filename == "tank.wav":
        # 1. Simulated FxLMS Path
        sec_path = np.zeros(12, dtype=np.float64)
        for i in range(2, 12):
            sec_path[i] = 0.8 * (0.6 ** (i - 2)) * np.cos(0.5 * (i - 2))
        sec_path = sec_path / np.sum(np.abs(sec_path))
        
        w_fx = np.zeros(32, dtype=np.float64)
        x_buf = np.zeros(32, dtype=np.float64)
        xf_buf = np.zeros(32, dtype=np.float64)
        sec_x_hist = np.zeros(12, dtype=np.float64)
        y_hist = np.zeros(12, dtype=np.float64)
        e_fx = np.zeros(len(norm_signal), dtype=np.float64)
        
        for n in range(len(norm_signal)):
            x_n = norm_signal[n]
            sec_x_hist[1:] = sec_x_hist[:-1]; sec_x_hist[0] = x_n
            xf_n = np.dot(sec_path, sec_x_hist)
            xf_buf[1:] = xf_buf[:-1]; xf_buf[0] = xf_n
            
            y_n = np.dot(w_fx, x_buf)
            y_hist[1:] = y_hist[:-1]; y_hist[0] = y_n
            ys_n = np.dot(sec_path, y_hist)
            
            err = norm_signal[n] - ys_n
            e_fx[n] = err
            w_fx += 0.0005 * err * xf_buf
            x_buf[1:] = x_buf[:-1]; x_buf[0] = x_n
            
        # 2. Communication Wavelet Denoising Path
        coeffs = pywt.wavedec(norm_signal, "db4", level=min(5, pywt.dwt_max_level(len(norm_signal), pywt.Wavelet("db4").dec_len)))
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        t_thresh = sigma * np.sqrt(2.0 * np.log(len(norm_signal)))
        denoised = pywt.waverec([coeffs[0]] + [pywt.threshold(c, t_thresh, "soft") for c in coeffs[1:]], "db4")[:len(norm_signal)]
        
        tank_sim_data = {
            "signal": norm_signal,
            "e_fxlms": e_fx,
            "denoised": denoised,
            "sample_rate": sr
        }

# --- 6. Print Report ---
for r in file_reports:
    print(f"\n[FILE: {r['filename']}]")
    print(f"  Noise Class       : {r['class']}")
    print(f"  Noise Profile     : {r['profile']}")
    print(f"  DSP Processing    : {r['mode']} (Taps={r['filter_len']}, mu={r['mu']})")
    print(f"  ML Distribution   : {r['dist']}")

if tank_sim_data:
    orig_rms = float(np.sqrt(np.mean(tank_sim_data["signal"] ** 2)))
    fx_rms = float(np.sqrt(np.mean(tank_sim_data["e_fxlms"] ** 2)))
    wav_rms = float(np.sqrt(np.mean(tank_sim_data["denoised"] ** 2)))
    print("-" * 80)
    print(f"[DETAILED DUAL-PATH METRICS FOR tank.wav]")
    print(f"  Input Original RMS          : {orig_rms:.6f}")
    print(f"  ANC Path (FxLMS Error RMS)  : {fx_rms:.6f} (Simulated Reduction: {20*np.log10(orig_rms/fx_rms):.2f} dB)")
    print(f"  Comm Path (Wavelet Out RMS) : {wav_rms:.6f}")

print("=" * 80)
print("ARCHITECTURAL FLOW:")
print("Common Front-End (Features + ML) -> Noise Profile & DSP Configuration")
print("  |-- 1. Communication Path : Adaptive Noise Suppression + Wavelet Reconstruction")
print("  \\-- 2. Active Noise Path  : Reference Sensor -> FxLMS -> Secondary Path -> Error")
print("-" * 80)
print("[IMPORTANT DISCLAIMER]")
print("This integrated pipeline connects all software algorithms for offline experimentation.")
print("It does not operate physical microphone hardware or real-time FPGA chips.")
print("=" * 80)

# --- 7. Generate and Save Dual-Pipeline Diagram / Plot ---
if tank_sim_data:
    t_axis = np.linspace(0.0, len(tank_sim_data["signal"]) / tank_sim_data["sample_rate"], len(tank_sim_data["signal"]), endpoint=False)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(t_axis, tank_sim_data["signal"], color="#2980b9", linewidth=0.7)
    plt.title("Common Front-End: Input Audio Signal (tank.wav - Classified as 'tank')", fontsize=11, fontweight="bold")
    plt.ylabel("Amplitude", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.subplot(3, 1, 2)
    plt.plot(t_axis, tank_sim_data["e_fxlms"], color="#27ae60", linewidth=0.7)
    plt.title("Path A (Active Noise Cancellation): FxLMS Residual Error Signal e[n]", fontsize=11, fontweight="bold")
    plt.ylabel("Amplitude", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.subplot(3, 1, 3)
    plt.plot(t_axis, tank_sim_data["denoised"], color="#8e44ad", linewidth=0.7)
    plt.title("Path B (Communication Speech Reconstruction): Wavelet Denoised Output", fontsize=11, fontweight="bold")
    plt.xlabel("Time (seconds)", fontsize=10)
    plt.ylabel("Amplitude", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH, dpi=300)
    plt.close()
    print(f"[SUCCESS] Dual-pipeline integration plot saved to: {OUTPUT_PLOT_PATH}")
    print("=" * 80)
