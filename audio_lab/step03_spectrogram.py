# ==============================================================================
# Project: AI/ML-enabled Adaptive Noise Cancellation (ANC) for Defence Vehicles
# PS ID: SIH26052 | Team: Echo Shield | Theme: Smart Vehicles
# 
# Step 03: Time-Frequency Analysis using Short-Time Fourier Transform (STFT)
# ==============================================================================

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft

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
# Example: data/tank.wav -> outputs/tank_spectrogram.png
file_stem = os.path.splitext(os.path.basename(INPUT_FILE_PATH))[0]
OUTPUT_IMAGE_PATH = os.path.join(PROJECT_ROOT, "outputs", f"{file_stem}_spectrogram.png")

print("=" * 65)
print("  ECHO SHIELD - Step 03: STFT Spectrogram Analyzer")
print("=" * 65)

# --- 2. Check If Audio File Exists ---
if not os.path.isfile(INPUT_FILE_PATH):
    print("\n[ERROR] Audio file not found!")
    print(f"Target location: {INPUT_FILE_PATH}")
    print("\nPlease verify the file path or place a valid .wav file in the 'data/' folder:")
    print(f"  --> {os.path.join(PROJECT_ROOT, 'data', 'test.wav')}")
    print("\nUsage Examples:")
    print("  python audio_lab/step03_spectrogram.py")
    print("  python audio_lab/step03_spectrogram.py data/tank.wav")
    print("  python audio_lab/step03_spectrogram.py data/heli.wav\n")
    sys.exit(1)

# --- 3. Load the Audio File ---
# wavfile.read returns the sampling rate (Hz) and raw data array
sample_rate, audio_data = wavfile.read(INPUT_FILE_PATH)

# --- 4. Channel Selection for Single Channel Analysis ---
# For stereo or multi-channel audio, we analyze Channel 1 (index 0) only.
# We do not mix channels to keep this analysis clean and isolated.
if audio_data.ndim == 1:
    num_channels = 1
    channel_signal = audio_data
    channel_desc = "Mono (Single Channel)"
else:
    num_channels = audio_data.shape[1]
    channel_signal = audio_data[:, 0]  # Channel 1 (index 0)
    channel_desc = f"Channel 1 of {num_channels} (Stereo / Multi-Channel)"

# Convert audio samples to float64 to prevent numerical overflow
signal = channel_signal.astype(np.float64)

num_samples = len(signal)
duration_seconds = num_samples / sample_rate

# --- 5. Print Audio and Analysis Information ---
print(f"File Path           : {INPUT_FILE_PATH}")
print(f"Sample Rate         : {sample_rate} Hz")
print(f"Total Samples (N)   : {num_samples:,}")
print(f"Duration            : {duration_seconds:.3f} seconds")
print(f"Channels in File    : {num_channels}")
print(f"Channel Analyzed    : {channel_desc}")
print("-" * 65)

# ==============================================================================
# CONCEPTUAL EXPLANATION OF STFT & SPECTROGRAM:
#
# 1. WHY THE SIGNAL IS DIVIDED INTO WINDOWS:
#    Real-world sounds (e.g., helicopter blades spinning up, gunfire, engine acceleration)
#    are non-stationary — their frequency characteristics change continuously over time.
#    A single FFT over the entire audio averages everything into one static spectrum,
#    completely losing "WHEN" specific frequencies occurred. By chopping the audio
#    into short time windows (segments), we can assume the signal is stationary within
#    each tiny slice.
#
# 2. WHAT nperseg MEANS:
#    nperseg (Number of points per segment) = 1024.
#    This determines the length of each slice. Larger nperseg gives finer frequency
#    resolution (narrower frequency bins) but coarser time resolution. Smaller nperseg
#    gives sharper time resolution but broader frequency bins.
#
# 3. WHAT noverlap MEANS:
#    noverlap (Number of overlapping points) = 512 (50% overlap).
#    Because window functions taper the signal to zero at the segment edges, data at the
#    edges would be lost or attenuated without overlap. Overlapping consecutive windows
#    ensures smooth transitions and preserves transient acoustic events.
#
# 4. WHY FFT IS PERFORMED ON EVERY WINDOW:
#    Performing an FFT on each windowed chunk extracts the active frequencies during that
#    exact moment in time.
#
# 5. HOW STFT DIFFERS FROM ONE FFT OF THE WHOLE SIGNAL:
#    - Standard FFT: Produces a 1D graph (Frequency vs Magnitude) for the entire recording.
#      Time information is completely lost.
#    - STFT: Produces a 2D time-frequency matrix (Time vs Frequency vs Magnitude),
#      showing how frequencies evolve and change dynamically over time.
# ==============================================================================

# --- 6. Compute the Short-Time Fourier Transform (STFT) ---
# Parameters:
#   - window='hann' : Smooth Hann window to reduce spectral leakage at edges
#   - nperseg=1024   : Length of each segment (1024 samples)
#   - noverlap=512   : Overlap between segments (512 samples, 50% overlap)
frequencies, times, zxx = stft(
    signal,
    fs=sample_rate,
    window="hann",
    nperseg=1024,
    noverlap=512
)

# STFT produces complex values Zxx. The spectrogram uses the absolute magnitude.
magnitude = np.abs(zxx)

# --- 7. Plot the Spectrogram ---
plt.figure(figsize=(11, 6))

# Plot magnitude across Time (X) and Frequency (Y)
# frequencies automatically range from 0 Hz to Nyquist frequency (sample_rate / 2)
mesh = plt.pcolormesh(times, frequencies, magnitude, shading="gouraud", cmap="inferno")

plt.title(f"Spectrogram (STFT) - {file_stem} [{channel_desc}]", fontsize=12, fontweight="bold")
plt.xlabel("Time (seconds)", fontsize=10)
plt.ylabel("Frequency (Hz)", fontsize=10)
plt.ylim(0, sample_rate / 2.0)  # Explicitly bound from 0 Hz to Nyquist frequency

# Add colorbar with label "Magnitude"
cbar = plt.colorbar(mesh)
cbar.set_label("Magnitude", fontsize=10)

plt.tight_layout()

# --- 8. Save the Plot ---
plt.savefig(OUTPUT_IMAGE_PATH, dpi=300)
plt.close()

print(f"[SUCCESS] Spectrogram saved to: {OUTPUT_IMAGE_PATH}")
print("=" * 65)
