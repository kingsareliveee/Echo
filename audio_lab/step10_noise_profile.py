# ==============================================================================
# Project: AI/ML-enabled Adaptive Noise Cancellation (ANC) for Defence Vehicles
# PS ID: SIH26052 | Team: Echo Shield | Theme: Smart Vehicles
# 
# Step 10: Prototype Noise Profile Selector (Decision Layer for Future DSP/ANC)
# ==============================================================================

import os
import sys
from collections import Counter
import numpy as np
# pyrefly: ignore [missing-import]
import joblib
from scipy.io import wavfile

# ==============================================================================
# BEGINNER-FRIENDLY CONCEPTS & TERMINOLOGY:
#
# 1. WHAT IS A "NOISE PROFILE"?
#    A Noise Profile is a categorized description of the acoustic disturbance.
#    Different defence vehicle noise types have completely different physical properties:
#    - TANK NOISE: Heavy, continuous mechanical vibrations from diesel engines, tracks,
#      and gearboxes (mostly 0-500 Hz low frequencies).
#    - HELICOPTER NOISE: Periodic tonal sound caused by main and tail rotor blade pass
#      frequencies (BPF) repeating at discrete harmonic pitches.
#    - GUNFIRE NOISE: Sudden, violent, high-amplitude acoustic shockwaves (transients/impulses)
#      spanning across all frequencies instantaneously.
#
# 2. WHY MAP CLASSIFICATION TO A NOISE PROFILE?
#    In a smart adaptive system, a single noise cancellation algorithm cannot handle all
#    noise types optimally:
#    - Tonal/periodic noise requires narrow notch filters or standard adaptive algorithms.
#    - Low-frequency engine rumble requires high-order FIR adaptive filtering.
#    - Impulsive gunfire requires non-linear thresholding or fast-acting transient suppression
#      to avoid acoustic clipping and filter divergence.
#
# 3. PIPELINE ARCHITECTURE:
#    [Audio Signal] -> [Feature Extraction] -> [Noise Classification] ->
#    [Noise Profile Mapping] -> [Future DSP/ANC Adaptive Parameter Selection]
# ==============================================================================

# --- 1. Configuration Constants ---
WINDOW_DURATION_MS = 100  # 100 ms window
HOP_DURATION_MS = 50      # 50 ms hop (50% overlap)

BANDS = [
    ("band_energy_0_500", 0.0, 500.0),
    ("band_energy_500_2000", 500.0, 2000.0),
    ("band_energy_2000_5000", 2000.0, 5000.0),
    ("band_energy_5000_10000", 5000.0, 10000.0)
]

# Prototype Profile Mapping Dictionary
PROFILE_LOOKUP = {
    "gun_fight": {
        "processing_profile": "IMPULSIVE_PROFILE",
        "reason": "Transient/impulsive noise profile"
    },
    "helicopter": {
        "processing_profile": "LOW_FREQ_TONAL_PROFILE",
        "reason": "Low-frequency and tonal noise profile"
    },
    "tank": {
        "processing_profile": "LOW_FREQ_MECHANICAL_PROFILE",
        "reason": "Low-frequency mechanical noise profile"
    }
}

# --- 2. Define File Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_PATH = os.path.join(OUTPUTS_DIR, "noise_classifier.joblib")

print("=" * 70)
print("  ECHO SHIELD - Step 10: Prototype Noise Profile Selector")
print("=" * 70)

# --- 3. Check and Load Trained Model ---
if not os.path.isfile(MODEL_PATH):
    print(f"\n[ERROR] Trained classifier model not found at: {MODEL_PATH}")
    print("\nPlease train the classifier model first by running Step 07:")
    print("  python audio_lab/step07_train_classifier.py\n")
    sys.exit(1)

model = joblib.load(MODEL_PATH)

# --- 4. Helper Function: Extract 8 Features and Predict Class ---
def predict_audio_profile(wav_path):
    """
    Extracts windowed features from a WAV file and predicts the overall noise class
    and corresponding DSP processing profile.
    """
    sample_rate, audio_data = wavfile.read(wav_path)
    
    # Analyze Channel 1 (index 0) only
    if audio_data.ndim > 1:
        channel_signal = audio_data[:, 0]
    else:
        channel_signal = audio_data
        
    signal = channel_signal.astype(np.float64)
    total_samples = len(signal)
    nyquist_freq = sample_rate / 2.0
    
    window_samples = int(sample_rate * (WINDOW_DURATION_MS / 1000.0))
    hop_samples = int(sample_rate * (HOP_DURATION_MS / 1000.0))
    
    if total_samples < window_samples:
        return None
        
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
    preds = model.predict(X_features)
    majority_class = Counter(preds).most_common(1)[0][0]
    
    # Map class to profile
    mapping = PROFILE_LOOKUP.get(majority_class, {
        "processing_profile": "GENERIC_PROFILE",
        "reason": "Unspecified noise category"
    })
    
    return {
        "noise_class": majority_class,
        "processing_profile": mapping["processing_profile"],
        "reason": mapping["reason"]
    }

# --- 5. Determine Target Files (CLI Argument or Default Suite) ---
if len(sys.argv) > 1:
    target_files = [sys.argv[1]]
else:
    target_files = [
        os.path.join(DATA_DIR, "tank.wav"),
        os.path.join(DATA_DIR, "heli.wav"),
        os.path.join(DATA_DIR, "gun_fight.wav"),
        os.path.join(DATA_DIR, "test.wav")
    ]

# --- 6. Process Files and Print Results ---
for file_path in target_files:
    if not os.path.isfile(file_path):
        if os.path.isfile(os.path.join(DATA_DIR, file_path)):
            file_path = os.path.join(DATA_DIR, file_path)
        else:
            print(f"\n[WARNING] File not found: {file_path}")
            continue
            
    filename = os.path.basename(file_path)
    result = predict_audio_profile(file_path)
    
    if result is None:
        print(f"\n[WARNING] Could not process file: {filename} (too short or empty)")
        continue
        
    print("\n------------------------------------------------------------")
    print(f"File              : {filename}")
    print(f"Noise Class       : {result['noise_class']}")
    print(f"Processing Profile: {result['processing_profile']}")
    print(f"Reason            : {result['reason']}")

print("\n" + "=" * 70)
print("SYSTEM EXPLANATION & ARCHITECTURAL WORKFLOW:")
print("=" * 70)
print("1. Noise Classification : ML model identifies dominant acoustic environment.")
print("2. Noise Profile Mapping: Maps class to high-level DSP handling strategy.")
print("3. Future DSP/ANC Hook  : Guides future parameter selection (e.g. filter")
print("                          order, step size mu, notch frequencies, or transient")
print("                          suppression thresholds) in the upcoming DSP pipeline.")
print("=" * 70)
