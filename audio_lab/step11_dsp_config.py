# ==============================================================================
# Project: AI/ML-enabled Adaptive Noise Cancellation (ANC) for Defence Vehicles
# PS ID: SIH26052 | Team: Echo Shield | Theme: Smart Vehicles
# 
# Step 11: Automated DSP Configuration Generator
# ==============================================================================

import os
import sys
from collections import Counter
import numpy as np
# pyrefly: ignore [missing-import]
import joblib
from scipy.io import wavfile

# ==============================================================================
# BEGINNER-FRIENDLY DSP CONCEPTS & TERMINOLOGY:
#
# 1. FILTER LENGTH (TAPS / WEIGHTS):
#    - In digital filter design (FIR filters), filter length represents the number
#      of past audio samples stored in memory to estimate the noise path.
#    - Filter Length = 32 means the digital filter uses a 32-tap delay line.
#    - Longer filters can model complex acoustic room/cabin echoes but require more
#      computation and can converge more slowly.
#
# 2. STEP SIZE (MU - \u03bc):
#    - The step size (\u03bc) is the "learning rate" of an adaptive digital filter.
#    - A larger \u03bc (e.g., 0.005 for tank noise) adapts quickly to steady acoustic rumble
#      because the noise is continuous and stable.
#    - A smaller \u03bc (e.g., 0.001 for gunfire) prevents the adaptive filter from
#      exploding / diverging when high-energy sudden shockwaves hit the microphone.
#
# 3. PROCESSING MODE:
#    - The high-level algorithmic control strategy selected by the system:
#      * LOW_FREQUENCY_ADAPTIVE  : Configured for low-pitch diesel rumble and track noise.
#      * LOW_FREQ_TONAL_ADAPTIVE : Configured for rotor blade passing frequencies and harmonics.
#      * IMPULSIVE_PROTECTION    : Configured for transient thresholding and stability preservation.
# ==============================================================================

# --- 1. Configuration Constants ---
WINDOW_DURATION_MS = 100  # Analysis window duration in ms
HOP_DURATION_MS = 50      # Advance hop between windows in ms (50% overlap)

BANDS = [
    ("band_energy_0_500", 0.0, 500.0),
    ("band_energy_500_2000", 500.0, 2000.0),
    ("band_energy_2000_5000", 2000.0, 5000.0),
    ("band_energy_5000_10000", 5000.0, 10000.0)
]

# DSP Configuration Mapping Table for Prototype Experimentation
DSP_CONFIG_LOOKUP = {
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

print("=" * 70)
print("  ECHO SHIELD - Step 11: Automated DSP Configuration Generator")
print("=" * 70)

# --- 3. Check and Load Trained Classifier Model ---
if not os.path.isfile(MODEL_PATH):
    print(f"\n[ERROR] Trained classifier model not found at: {MODEL_PATH}")
    print("\nPlease train the model first by running Step 07:")
    print("  python audio_lab/step07_train_classifier.py\n")
    sys.exit(1)

model = joblib.load(MODEL_PATH)

# --- 4. Helper Function: Extract 8 Features and Predict Noise Class ---
def classify_audio_file(wav_path):
    """
    Extracts 8 windowed features from a WAV file and predicts the overall noise class
    using majority voting across all windows.
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
    return majority_class

# --- 5. Determine Target Files (CLI Argument or Full Audio Suite) ---
if len(sys.argv) > 1:
    target_files = [sys.argv[1]]
else:
    target_files = [
        os.path.join(DATA_DIR, "tank.wav"),
        os.path.join(DATA_DIR, "heli.wav"),
        os.path.join(DATA_DIR, "gun_fight.wav"),
        os.path.join(DATA_DIR, "test.wav")
    ]

# --- 6. Process Audio Files and Generate DSP Configurations ---
for file_path in target_files:
    if not os.path.isfile(file_path):
        if os.path.isfile(os.path.join(DATA_DIR, file_path)):
            file_path = os.path.join(DATA_DIR, file_path)
        else:
            print(f"\n[WARNING] File not found: {file_path}")
            continue
            
    filename = os.path.basename(file_path)
    predicted_class = classify_audio_file(file_path)
    
    if predicted_class is None:
        print(f"\n[WARNING] Could not process {filename} (empty or shorter than 100 ms).")
        continue
        
    config = DSP_CONFIG_LOOKUP.get(predicted_class, {
        "profile": "UNKNOWN_PROFILE",
        "filter_length": 32,
        "step_size_mu": 0.001,
        "processing_mode": "DEFAULT_MODE"
    })
    
    print("\n------------------------------------------------------------")
    print(f"File            : {filename}")
    print(f"Noise Class     : {predicted_class}")
    print(f"Noise Profile   : {config['profile']}")
    print(f"Filter Length   : {config['filter_length']}")
    print(f"Step Size (mu)  : {config['step_size_mu']}")
    print(f"Processing Mode : {config['processing_mode']}")

print("\n" + "=" * 70)
print("ARCHITECTURAL FLOW:")
print("ML classification -> Noise Profile -> DSP Configuration")
print("-" * 70)
print("These parameters are prototype starting values for experimentation,")
print("not measured optimal ANC parameters.")
print("=" * 70)
