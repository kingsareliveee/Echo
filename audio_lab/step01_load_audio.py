# ==============================================================================
# Project: AI/ML-enabled Adaptive Noise Cancellation (ANC) for Defence Vehicles
# PS ID: SIH26052 | Team: Echo Shield | Theme: Smart Vehicles
# 
# Step 01: Loading, Inspecting, and Visualizing a WAV Audio File
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
file_stem = os.path.splitext(os.path.basename(INPUT_FILE_PATH))[0]
OUTPUT_IMAGE_PATH = os.path.join(PROJECT_ROOT, "outputs", f"{file_stem}_waveform.png")

print("=" * 65)
print("  ECHO SHIELD - Step 01: Audio Inspection & Waveform Generator")
print("=" * 65)

# --- 2. Check If Audio File Exists ---
if not os.path.isfile(INPUT_FILE_PATH):
    print("\n[ERROR] Audio file not found!")
    print(f"Target location: {INPUT_FILE_PATH}")
    print("\nPlease verify the file path or place a valid .wav file in the 'data/' folder:")
    print(f"  --> {os.path.join(PROJECT_ROOT, 'data', 'test.wav')}")
    print("\nUsage Examples:")
    print("  python audio_lab/step01_load_audio.py")
    print("  python audio_lab/step01_load_audio.py data/tank.wav\n")
    sys.exit(1)

# --- 3. Load the WAV File ---
sample_rate, audio_data = wavfile.read(INPUT_FILE_PATH)

# --- 4. Inspect Audio Properties ---
if audio_data.ndim == 1:
    num_channels = 1  # Mono audio
    num_samples = len(audio_data)
else:
    num_channels = audio_data.shape[1]  # Stereo or multi-channel
    num_samples = audio_data.shape[0]

duration_seconds = num_samples / sample_rate
data_type = audio_data.dtype
min_val = np.min(audio_data)
max_val = np.max(audio_data)

# Print audio metadata
print(f"File Path           : {INPUT_FILE_PATH}")
print(f"Sample Rate         : {sample_rate} Hz")
print(f"Total Samples       : {num_samples:,}")
print(f"Channels            : {num_channels} ({'Mono' if num_channels == 1 else 'Stereo / Multi-Channel'})")
print(f"Audio Data Type     : {data_type}")
print(f"Duration            : {duration_seconds:.3f} seconds")
print(f"Min Sample Value    : {min_val}")
print(f"Max Sample Value    : {max_val}")
print("-" * 65)

# --- 5. Generate Time Axis for Waveform Plotting ---
time_axis = np.linspace(0.0, duration_seconds, num_samples, endpoint=False)

# --- 6. Plot the Waveform ---
if num_channels == 1:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_axis, audio_data, color="#007acc", linewidth=0.8)
    ax.set_title(f"Audio Waveform - Mono Channel ({file_stem})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_ylabel("Amplitude", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
else:
    fig, axes = plt.subplots(num_channels, 1, figsize=(10, 3 * num_channels), sharex=True)
    for ch in range(num_channels):
        ax = axes[ch] if num_channels > 1 else axes
        channel_data = audio_data[:, ch]
        ax.plot(time_axis, channel_data, color="#007acc" if ch == 0 else "#e056fd", linewidth=0.8)
        ax.set_title(f"Audio Waveform - Channel {ch + 1} ({file_stem})", fontsize=11, fontweight="bold")
        ax.set_ylabel("Amplitude", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)
    axes[-1].set_xlabel("Time (seconds)", fontsize=10)

plt.tight_layout()

# --- 7. Save the Plot ---
plt.savefig(OUTPUT_IMAGE_PATH, dpi=300)
plt.close()

print(f"[SUCCESS] Waveform saved to: {OUTPUT_IMAGE_PATH}")
print("=" * 65)
