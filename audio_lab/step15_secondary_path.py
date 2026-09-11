# ==============================================================================
# Project: AI/ML-enabled Adaptive Noise Cancellation (ANC) for Defence Vehicles
# PS ID: SIH26052 | Team: Echo Shield | Theme: Smart Vehicles
# 
# Step 15: Synthetic Secondary Path Modeling (S(z) Transfer Function)
# ==============================================================================

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import lfilter

# ==============================================================================
# BEGINNER-FRIENDLY CONCEPT: WHAT IS THE "SECONDARY PATH" (S(z))?
#
# 1. PHYSICAL MEANING:
#    In a physical ANC system, the anti-noise computed by the DSP chip does not
#    instantly appear at the user's ear. It must pass through:
#      [DSP Output] -> [D/A Converter] -> [Power Amp] -> [Speaker/Transducer]
#      -> [Acoustic Air Path (Flight/Cabin Space)] -> [Error Microphone] -> [A/D Converter]
#    This entire electro-acoustic transmission chain is called the SECONDARY PATH S(z).
#
# 2. WHY FxLMS REQUIRES SECONDARY PATH MODELING:
#    The secondary path introduces amplitude attenuation and, most critically, PHASE DELAY.
#    If the standard LMS algorithm ignores this delay, the weight updates will be calculated
#    out of phase, causing the adaptive filter to diverge into loud acoustic feedback (howling).
#    FxLMS (Filtered-X LMS) filters the reference signal x[n] with an estimate S_hat(z)
#    to maintain correct mathematical phase alignment.
# ==============================================================================

# --- 1. Define File Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(OUTPUTS_DIR, exist_ok=True)

INPUT_FILE_PATH = os.path.join(DATA_DIR, "tank.wav")
OUTPUT_PLOT_PATH = os.path.join(OUTPUTS_DIR, "step15_secondary_path.png")

print("=" * 75)
print("  ECHO SHIELD - Step 15: Secondary Path Model S(z)")
print("=" * 75)

# --- 2. Synthetic Secondary Path FIR Impulse Response ---
# A synthetic 12-tap FIR model with 2-sample electro-acoustic delay and damped resonance
# NOTE: This is a synthetic simulation impulse response, NOT measured hardware data.
delay_samples = 2
fir_length = 12

synthetic_secondary_path = np.zeros(fir_length, dtype=np.float64)
# Model a realistic acoustic propagation delay followed by an exponential damped impulse
for i in range(delay_samples, fir_length):
    synthetic_secondary_path[i] = 0.8 * (0.6 ** (i - delay_samples)) * np.cos(0.5 * (i - delay_samples))

# Normalize impulse response energy to maintain stable gain
synthetic_secondary_path = synthetic_secondary_path / np.sum(np.abs(synthetic_secondary_path))

# --- 3. Load Test Audio Signal ---
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
    test_signal = channel_signal.astype(np.float64) / norm_factor
else:
    test_signal = channel_signal.astype(np.float64)

# Apply synthetic secondary path to test signal using FIR convolution / lfilter
output_signal = lfilter(synthetic_secondary_path, 1.0, test_signal)

input_rms = float(np.sqrt(np.mean(test_signal ** 2)))
output_rms = float(np.sqrt(np.mean(output_signal ** 2)))

# --- 4. Print Summary ---
print(f"Secondary Path FIR Length: {fir_length} taps")
print(f"Acoustic Delay          : {delay_samples} samples ({delay_samples / sample_rate * 1000.0:.3f} ms)")
print(f"Input Signal RMS        : {input_rms:.6f}")
print(f"Secondary Output RMS    : {output_rms:.6f}")
print("-" * 75)
print("\n[WHY FxLMS NEEDS SECONDARY PATH MODEL S_hat(z)]")
print("1. Phase Alignment : Physical speakers and acoustic paths introduce phase shifts.")
print("2. Filtered-X Rule : Filtering x[n] with S_hat(z) ensures weight gradient updates")
print("                     point in the true direction of minimum acoustic error.")
print("-" * 75)
print("[IMPORTANT DISCLAIMER]")
print("This secondary-path model is a synthetic software FIR template.")
print("It is NOT measured hardware data and does NOT represent the final vehicle cabin.")
print("=" * 75)

# --- 5. Plot and Save Visualization ---
time_axis = np.linspace(0.0, len(test_signal) / sample_rate, len(test_signal), endpoint=False)
zoom_samples = min(500, len(test_signal))

plt.figure(figsize=(12, 8))

# Subplot 1: Secondary Path Impulse Response
plt.subplot(3, 1, 1)
markerline, stemlines, baseline = plt.stem(range(fir_length), synthetic_secondary_path)
plt.title(f"Synthetic Secondary-Path Impulse Response S(z) ({fir_length} Taps, Delay={delay_samples})", fontsize=11, fontweight="bold")
plt.xlabel("Sample Tap (k)", fontsize=10)
plt.ylabel("Weight Value", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)

# Subplot 2: Input Test Signal (Zoomed)
plt.subplot(3, 1, 2)
plt.plot(time_axis[:zoom_samples] * 1000.0, test_signal[:zoom_samples], color="#2980b9", linewidth=0.9)
plt.title("Input Anti-Noise Test Signal x[n] (Zoomed View)", fontsize=11, fontweight="bold")
plt.xlabel("Time (milliseconds)", fontsize=10)
plt.ylabel("Amplitude", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)

# Subplot 3: Output Convolved Signal (Zoomed)
plt.subplot(3, 1, 3)
plt.plot(time_axis[:zoom_samples] * 1000.0, output_signal[:zoom_samples], color="#e67e22", linewidth=0.9)
plt.title("Secondary-Path Output Signal (Delayed & Filtered)", fontsize=11, fontweight="bold")
plt.xlabel("Time (milliseconds)", fontsize=10)
plt.ylabel("Amplitude", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT_PATH, dpi=300)
plt.close()

print(f"[SUCCESS] Secondary-path plot saved to: {OUTPUT_PLOT_PATH}")
print("=" * 75)
