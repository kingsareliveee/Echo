# ==============================================================================
# Project: AI/ML-enabled Adaptive Noise Cancellation (ANC) for Defence Vehicles
# PS ID: SIH26052 | Team: Echo Shield | Theme: Smart Vehicles
# 
# Step 02: Frequency Domain Analysis using Fast Fourier Transform (FFT)
# ==============================================================================

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# --- 1. Define File Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Ensure outputs and data directories exist
os.makedirs(os.path.join(PROJECT_ROOT, "outputs"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)

# Check if a WAV file path was provided as a command-line argument
if len(sys.argv) > 1:
    raw_path = sys.argv[1]
    # Check if path exists relative to current working dir or project root
    if os.path.isfile(raw_path):
        INPUT_FILE_PATH = os.path.abspath(raw_path)
    elif os.path.isfile(os.path.join(PROJECT_ROOT, raw_path)):
        INPUT_FILE_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, raw_path))
    else:
        INPUT_FILE_PATH = os.path.abspath(raw_path)
else:
    # Default file path if no argument is passed
    INPUT_FILE_PATH = os.path.join(PROJECT_ROOT, "data", "test.wav")

# Automatically determine the output image filename based on the input WAV filename
# Example: data/tank.wav -> outputs/tank_fft.png
file_stem = os.path.splitext(os.path.basename(INPUT_FILE_PATH))[0]
OUTPUT_IMAGE_PATH = os.path.join(PROJECT_ROOT, "outputs", f"{file_stem}_fft.png")

print("=" * 65)
print("  ECHO SHIELD - Step 02: FFT Frequency Spectrum Analyzer")
print("=" * 65)

# --- 2. Check If Audio File Exists ---
if not os.path.isfile(INPUT_FILE_PATH):
    print("\n[ERROR] Audio file not found!")
    print(f"Target location: {INPUT_FILE_PATH}")
    print("\nPlease verify the file path or place a valid .wav file in the 'data/' folder:")
    print(f"  --> {os.path.join(PROJECT_ROOT, 'data', 'test.wav')}")
    print("\nUsage Examples:")
    print("  python audio_lab/step02_fft.py")
    print("  python audio_lab/step02_fft.py data/tank.wav\n")
    sys.exit(1)

# --- 3. Load the Audio File ---
# wavfile.read returns the sampling rate (Hz) and raw data array
sample_rate, audio_data = wavfile.read(INPUT_FILE_PATH)

# --- 4. Channel Selection for Single Channel Analysis ---
# For stereo or multi-channel audio, we select Channel 1 (index 0) only.
# We do not mix channels to keep this fundamental experiment clean and isolated.
if audio_data.ndim == 1:
    num_channels = 1
    channel_signal = audio_data
    channel_desc = "Mono (Single Channel)"
else:
    num_channels = audio_data.shape[1]
    channel_signal = audio_data[:, 0]  # Channel 1 (index 0)
    channel_desc = f"Channel 1 of {num_channels} (Stereo / Multi-Channel)"

# Convert audio samples to float64 to prevent numerical overflow during mathematical transforms
signal = channel_signal.astype(np.float64)

num_samples = len(signal)
duration_seconds = num_samples / sample_rate

# --- 5. Compute the Fast Fourier Transform (FFT) ---
# WHAT FFT DOES:
# The Fast Fourier Transform (FFT) converts a signal from the Time Domain (Amplitude vs Time)
# into the Frequency Domain (Magnitude vs Frequency). It shows which individual sine wave
# frequencies make up the complex acoustic sound.
#
# WHY np.fft.fft:
# NumPy's fft.fft function computes the Discrete Fourier Transform (DFT) in O(N log N) time,
# returning complex numbers containing both magnitude (strength) and phase information.
raw_fft = np.fft.fft(signal)

# --- 6. Extract Positive Frequencies Spectrum ---
# WHY WE KEEP ONLY POSITIVE FREQUENCIES:
# Real-valued audio signals have symmetric Fourier transforms where negative frequencies are
# exact mirror images of positive frequencies. The unique physical spectral information lies
# between 0 Hz (DC) and the Nyquist Frequency (sample_rate / 2).
# We take the first half of the FFT output: N // 2 points.
half_n = num_samples // 2

# Calculate magnitude (absolute value of complex FFT output) and normalize by N
# Multiplying by 2 accounts for combining the energy of positive and negative halves
magnitude = (2.0 / num_samples) * np.abs(raw_fft[:half_n])
magnitude[0] = magnitude[0] / 2.0  # DC component (0 Hz) is not duplicated in negative frequencies

# --- 7. Generate Frequency Axis in Hz ---
# HOW THE FREQUENCY AXIS IS CREATED:
# Frequency resolution = sample_rate / num_samples.
# np.linspace creates an array from 0 Hz up to (sample_rate / 2) Hz with half_n points.
freq_axis = np.linspace(0.0, sample_rate / 2.0, half_n, endpoint=False)

# --- 8. Detect the Dominant Frequency Component ---
# We find the peak magnitude in the positive spectrum (ignoring index 0 which is constant DC offset)
if len(magnitude) > 1:
    dominant_idx = 1 + np.argmax(magnitude[1:])
else:
    dominant_idx = 0

dominant_freq = freq_axis[dominant_idx]
dominant_mag = magnitude[dominant_idx]

# --- 9. Print Analysis Summary ---
print(f"File Path           : {INPUT_FILE_PATH}")
print(f"Sample Rate         : {sample_rate} Hz")
print(f"Total Samples (N)   : {num_samples:,}")
print(f"Duration            : {duration_seconds:.3f} seconds")
print(f"Channels in File    : {num_channels}")
print(f"Channel Analyzed    : {channel_desc}")
print(f"Dominant Frequency  : {dominant_freq:.2f} Hz")
print(f"Dominant Magnitude  : {dominant_mag:.2f}")
print("-" * 65)
print("[NOTE] The dominant frequency is simply the strongest individual frequency")
print("       component detected in this basic FFT. It is not necessarily the entire")
print("       noise profile or sole interference source.")
print("-" * 65)

# --- 10. Plot the Frequency Spectrum ---
plt.figure(figsize=(11, 5))
plt.plot(freq_axis, magnitude, color="#e74c3c", linewidth=0.9, label="Magnitude Spectrum")

# Highlight dominant peak with a marker
plt.axvline(dominant_freq, color="#2c3e50", linestyle="--", linewidth=1.2,
            label=f"Dominant Peak: {dominant_freq:.1f} Hz (Mag: {dominant_mag:.1f})")

plt.title(f"Frequency Spectrum (FFT) - {file_stem} [{channel_desc}]", fontsize=12, fontweight="bold")
plt.xlabel("Frequency (Hz)", fontsize=10)
plt.ylabel("Magnitude (Amplitude)", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="upper right", frameon=True)
plt.tight_layout()

# --- 11. Save the Plot ---
plt.savefig(OUTPUT_IMAGE_PATH, dpi=300)
plt.close()

print(f"[SUCCESS] Frequency spectrum saved to: {OUTPUT_IMAGE_PATH}")
print("=" * 65)
