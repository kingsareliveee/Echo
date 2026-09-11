# ==============================================================================
# Project: AI/ML-enabled Adaptive Noise Cancellation (ANC) for Defence Vehicles
# PS ID: SIH26052 | Team: Echo Shield | Theme: Smart Vehicles
# 
# Step 09: Batch Audio Prediction & Comparison
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
HOP_DURATION_MS = 50      # Hop step between windows in ms (50% overlap)

BANDS = [
    ("band_energy_0_500", 0.0, 500.0),
    ("band_energy_500_2000", 500.0, 2000.0),
    ("band_energy_2000_5000", 2000.0, 5000.0),
    ("band_energy_5000_10000", 5000.0, 10000.0)
]

# --- 2. Define File Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_PATH = os.path.join(OUTPUTS_DIR, "noise_classifier.joblib")

# List of WAV audio files to batch analyze
TARGET_FILES = [
    "tank.wav",
    "heli.wav",
    "gun_fight.wav",
    "test.wav"
]

print("=" * 75)
print("  ECHO SHIELD - Step 09: Batch Audio Prediction & Comparison")
print("=" * 75)

# --- 3. Check and Load Trained Model ---
if not os.path.isfile(MODEL_PATH):
    print(f"\n[ERROR] Trained classifier model not found at: {MODEL_PATH}")
    print("\nPlease train the classifier model first by running Step 07:")
    print("  python audio_lab/step07_train_classifier.py\n")
    sys.exit(1)

model = joblib.load(MODEL_PATH)
classes = sorted(model.classes_)  # ['gun_fight', 'helicopter', 'tank']

# --- 4. Helper Function: Extract 8 Features for a WAV File ---
def extract_window_features(wav_path):
    """
    Extracts the exact 8 acoustic features for every 100 ms window (50 ms hop)
    matching Steps 05, 07, and 08.
    """
    sample_rate, audio_data = wavfile.read(wav_path)
    
    # Analyze Channel 1 (index 0) only for stereo/multi-channel
    if audio_data.ndim > 1:
        channel_signal = audio_data[:, 0]
    else:
        channel_signal = audio_data
        
    signal = channel_signal.astype(np.float64)
    total_samples = len(signal)
    duration_sec = total_samples / sample_rate
    nyquist_freq = sample_rate / 2.0
    
    window_samples = int(sample_rate * (WINDOW_DURATION_MS / 1000.0))
    hop_samples = int(sample_rate * (HOP_DURATION_MS / 1000.0))
    
    if total_samples < window_samples:
        return None, sample_rate, duration_sec, 0
        
    half_n = window_samples // 2
    freq_axis = np.linspace(0.0, nyquist_freq, half_n, endpoint=False)
    
    feature_rows = []
    start_sample = 0
    
    while start_sample + window_samples <= total_samples:
        end_sample = start_sample + window_samples
        w_sig = signal[start_sample:end_sample]
        
        # 1. RMS Energy
        rms = float(np.sqrt(np.mean(w_sig ** 2)))
        
        # 2. Zero Crossing Rate
        if window_samples > 1:
            zcr = float(np.sum(np.abs(np.diff(np.signbit(w_sig)))) / (window_samples - 1))
        else:
            zcr = 0.0
            
        # 3. FFT Magnitude Spectrum
        raw_fft = np.fft.fft(w_sig)
        mag = (2.0 / window_samples) * np.abs(raw_fft[:half_n])
        if len(mag) > 0:
            mag[0] /= 2.0
        total_mag = float(np.sum(mag))
        
        # 4. Spectral Centroid
        if total_mag > 0:
            centroid = float(np.sum(freq_axis * mag) / total_mag)
        else:
            centroid = 0.0
            
        # 5. Spectral Rolloff (85%)
        if total_mag > 0:
            cum_mag = np.cumsum(mag)
            rolloff_threshold = 0.85 * total_mag
            r_idx = min(int(np.searchsorted(cum_mag, rolloff_threshold)), len(freq_axis) - 1)
            rolloff = float(freq_axis[r_idx])
        else:
            rolloff = 0.0
            
        # 6-9. Sub-band Energies
        band_energies = []
        for _, f_low, f_high in BANDS:
            if f_low < nyquist_freq:
                f_high_actual = min(f_high, nyquist_freq)
                mask = (freq_axis >= f_low) & (freq_axis < f_high_actual)
                band_energies.append(float(np.sum(mag[mask])))
            else:
                band_energies.append(0.0)
                
        feature_rows.append([rms, zcr, centroid, rolloff] + band_energies)
        start_sample += hop_samples
        
    X_features = np.array(feature_rows, dtype=np.float64)
    return X_features, sample_rate, duration_sec, len(feature_rows)

# --- 5. Run Predictions on All Target Files ---
results = []

for filename in TARGET_FILES:
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.isfile(filepath):
        print(f"[WARNING] Skipping missing audio file: {filepath}")
        continue
        
    X_feat, sr, duration, num_windows = extract_window_features(filepath)
    if X_feat is None or num_windows == 0:
        print(f"[WARNING] Skipping invalid or short audio file: {filename}")
        continue
        
    preds = model.predict(X_feat)
    counts = Counter(preds)
    overall_winner = counts.most_common(1)[0][0]
    
    # Calculate percentages for all 3 classes
    pct_gun = (counts.get("gun_fight", 0) / num_windows) * 100.0
    pct_heli = (counts.get("helicopter", 0) / num_windows) * 100.0
    pct_tank = (counts.get("tank", 0) / num_windows) * 100.0
    
    results.append({
        "filename": filename,
        "duration_sec": duration,
        "sample_rate": sr,
        "windows": num_windows,
        "overall_prediction": overall_winner,
        "gun_fight_pct": pct_gun,
        "helicopter_pct": pct_heli,
        "tank_pct": pct_tank
    })

# --- 6. Print Summary Table ---
print("\n" + "-" * 88)
print(f"{'Filename':<14} | {'Duration':<9} | {'Rate(Hz)':<8} | {'Windows':<7} | {'Overall Pred':<12} | {'GunFight%':<9} | {'Heli%':<7} | {'Tank%':<6}")
print("-" * 88)

for r in results:
    print(
        f"{r['filename']:<14} | "
        f"{r['duration_sec']:>6.2f} s   | "
        f"{r['sample_rate']:>8} | "
        f"{r['windows']:>7} | "
        f"{r['overall_prediction']:<12} | "
        f"{r['gun_fight_pct']:>8.1f}% | "
        f"{r['helicopter_pct']:>6.1f}% | "
        f"{r['tank_pct']:>5.1f}%"
    )

print("-" * 88)
print("\n[NOTE] These are model predictions, not measured real-world classification accuracy.")
print("=" * 88)
