# ==============================================================================
# Project: AI/ML-enabled Adaptive Noise Cancellation (ANC) for Defence Vehicles
# PS ID: SIH26052 | Team: Echo Shield | Theme: Smart Vehicles
# 
# Step 12: Least Mean Squares (LMS) Adaptive Filter Prototype
# ==============================================================================

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# ==============================================================================
# BEGINNER-FRIENDLY LMS DSP CONCEPTS & MATHEMATICS:
#
# 1. WHAT IS AN ADAPTIVE FILTER?
#    Unlike fixed digital filters (which have unchanging frequency responses), an
#    adaptive filter automatically adjusts its own internal coefficients in real-time
#    to track changing statistical properties of an acoustic sound wave.
#
# 2. FILTER COEFFICIENTS (w):
#    The array of adjustable weights w[0], w[1], ..., w[L-1] inside the FIR filter.
#    Multiplying past audio samples by these weights produces the estimated noise signal y[n].
#
# 3. ERROR SIGNAL (e[n]):
#    e[n] = d[n] - y[n]
#    The difference between the desired audio signal d[n] and the filter's output y[n].
#    In noise cancellation experiments, e[n] represents the residual noise left over.
#
# 4. STEP SIZE (mu - \u03bc):
#    Controls how much the filter coefficients are updated at each single sample step:
#        w[n+1] = w[n] + \u03bc * e[n] * x[n]
#    - Larger \u03bc = faster convergence, but risk of instability/distortion.
#    - Smaller \u03bc = slower convergence, but cleaner and more stable filtering.
#
# 5. WHY LMS COEFFICIENTS CHANGE OVER TIME:
#    At every incoming audio sample, the LMS algorithm calculates the gradient of the
#    squared error and nudges the weights in the direction that minimizes total error power.
# ==============================================================================

# --- 1. Configuration Constants ---
FILTER_LENGTH = 32     # Number of filter taps (weights)
STEP_SIZE_MU = 0.005   # Adaptation step size (\u03bc) for normalized signal range

# --- 2. Define File Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Support optional CLI argument for audio file, defaulting to tank.wav
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

file_stem = os.path.splitext(os.path.basename(INPUT_FILE_PATH))[0]
OUTPUT_WAV_PATH = os.path.join(OUTPUTS_DIR, f"{file_stem}_lms_error.wav")
OUTPUT_PLOT_PATH = os.path.join(OUTPUTS_DIR, f"{file_stem}_lms_comparison.png")

print("=" * 70)
print("  ECHO SHIELD - Step 12: LMS Adaptive Filter Prototype")
print("=" * 70)

# --- 3. Input Validation ---
if not os.path.isfile(INPUT_FILE_PATH):
    print(f"\n[ERROR] Audio file not found at: {INPUT_FILE_PATH}")
    sys.exit(1)

# --- 4. Load Audio File ---
sample_rate, audio_data = wavfile.read(INPUT_FILE_PATH)

# Select Channel 1 (index 0) if stereo/multichannel
if audio_data.ndim > 1:
    channel_signal = audio_data[:, 0]
    channel_desc = f"Channel 1 of {audio_data.shape[1]} (Stereo)"
else:
    channel_signal = audio_data
    channel_desc = "Channel 1 (Mono)"

# Normalize audio to floating point range [-1.0, 1.0] for stable LMS arithmetic
raw_max = np.max(np.abs(channel_signal))
if raw_max > 0:
    if np.issubdtype(channel_signal.dtype, np.integer):
        # Normalize 16-bit / integer audio by standard max bit-depth range
        norm_factor = 32768.0 if np.issubdtype(channel_signal.dtype, np.int16) else float(raw_max)
        signal = channel_signal.astype(np.float64) / norm_factor
    else:
        signal = channel_signal.astype(np.float64) / float(raw_max)
else:
    signal = channel_signal.astype(np.float64)

num_samples = len(signal)
duration_sec = num_samples / sample_rate

# --- 5. Implement LMS Adaptive Filter (Sample-by-Sample Loop) ---
# Weights vector initialized to zeros
weights = np.zeros(FILTER_LENGTH, dtype=np.float64)

# Delay buffer holding the past 32 samples [x[n-1], x[n-2], ..., x[n-L]]
x_buffer = np.zeros(FILTER_LENGTH, dtype=np.float64)

# Output arrays for filter estimation y[n] and error e[n]
y_output = np.zeros(num_samples, dtype=np.float64)
e_error = np.zeros(num_samples, dtype=np.float64)

# Sample-by-sample LMS adaptive processing
for n in range(num_samples):
    d_n = signal[n]  # Desired primary signal sample
    
    # Calculate filter output: y[n] = sum(w[k] * x[n - 1 - k])
    y_n = np.dot(weights, x_buffer)
    y_output[n] = y_n
    
    # Calculate error: e[n] = d[n] - y[n]
    e_n = d_n - y_n
    e_error[n] = e_n
    
    # Update filter coefficients: w[n+1] = w[n] + mu * e[n] * x_vector
    weights += STEP_SIZE_MU * e_n * x_buffer
    
    # Shift delay line buffer and insert current sample as past sample for next iteration
    x_buffer[1:] = x_buffer[:-1]
    x_buffer[0] = d_n

# --- 6. Compute Quantitative Performance Metrics ---
input_rms = float(np.sqrt(np.mean(signal ** 2)))
error_rms = float(np.sqrt(np.mean(e_error ** 2)))

if error_rms > 0 and input_rms > 0:
    rms_reduction_db = 20.0 * np.log10(input_rms / error_rms)
else:
    rms_reduction_db = 0.0

# --- 7. Print Summary Metrics ---
print(f"File Path           : {INPUT_FILE_PATH}")
print(f"Sample Rate         : {sample_rate} Hz")
print(f"Number of Samples   : {num_samples:,} ({duration_sec:.3f} seconds)")
print(f"Channel             : {channel_desc}")
print(f"Filter Length (Taps): {FILTER_LENGTH}")
print(f"Step Size (mu)      : {STEP_SIZE_MU}")
print("-" * 70)
print(f"Input Signal RMS    : {input_rms:.6f}")
print(f"Error Signal RMS    : {error_rms:.6f}")
print(f"RMS Reduction       : {rms_reduction_db:.2f} dB")
print("=" * 70)

# --- 8. Save Error Signal to WAV File ---
# Scale error back to 16-bit PCM integer range for audible playback
clipped_error = np.clip(e_error, -1.0, 1.0)
error_wav_data = (clipped_error * 32767.0).astype(np.int16)
wavfile.write(OUTPUT_WAV_PATH, sample_rate, error_wav_data)
print(f"[SUCCESS] LMS error audio saved to: {OUTPUT_WAV_PATH}")

# --- 9. Plot and Save Waveform Comparison ---
time_axis = np.linspace(0.0, duration_sec, num_samples, endpoint=False)

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time_axis, signal, color="#2980b9", linewidth=0.7)
plt.title(f"Original Input Signal ({file_stem}.wav) - Normalized", fontsize=11, fontweight="bold")
plt.ylabel("Amplitude", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(2, 1, 2)
plt.plot(time_axis, e_error, color="#e74c3c", linewidth=0.7)
plt.title(f"LMS Adaptive Filter Error Signal e[n] (RMS Reduction: {rms_reduction_db:.2f} dB)", fontsize=11, fontweight="bold")
plt.xlabel("Time (seconds)", fontsize=10)
plt.ylabel("Amplitude", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT_PATH, dpi=300)
plt.close()

print(f"[SUCCESS] Waveform comparison plot saved to: {OUTPUT_PLOT_PATH}")
print("-" * 70)
print("[IMPORTANT DISCLAIMER]")
print("LMS adaptive filtering is demonstrated here as a software DSP building block.")
print("Full FxLMS ANC requires a reference signal, secondary-path model, anti-noise")
print("output, and acoustic/error feedback.")
print("=" * 70)
