# Echo Shield: AI/ML-enabled Adaptive Noise Cancellation (ANC) for Defence Vehicles

**PS ID:** SIH26052  
**Theme:** Smart Vehicles  
**Category:** Hardware  
**Team:** Echo Shield  

---

## Step 01: Loading & Inspecting WAV Audio

### 1. Overview
Step 01 provides the fundamental audio I/O pipeline for the project. It loads an uncompressed WAV audio file, inspects essential acoustic parameters (sample rate, bit depth/type, duration, min/max amplitude, channel count), and generates a multi-channel time-domain waveform visualization.

---

### 2. Requirements & Python Version
- **Python Version:** Python 3.8+ (Python 3 recommended)
- **Required Libraries:**
  - `numpy` (Numerical array operations)
  - `matplotlib` (Waveform visualization)
  - `scipy` (WAV file reading via `scipy.io.wavfile`)

---

### 3. Installation
Open your terminal / command prompt and install the dependencies using `pip`:

```bash
pip install numpy matplotlib scipy
```

---

### 4. Placing the Audio File
Place any standard `.wav` file in the `data/` folder and name it `test.wav`:

```text
echo_shield/
└── data/
    └── test.wav
```

> **Note:** If `data/test.wav` is missing, the script will clearly notify you of the exact required path.

---

### 5. Running the Script
Run the script from inside the project directory or using python directly:

```bash
# Navigate to the echo_shield folder:
cd echo_shield

# Execute Step 01:
python audio_lab/step01_load_audio.py
```

---

### 6. Expected Output

#### Console Output:
```text
=================================================================
  ECHO SHIELD - Step 01: Audio Inspection & Waveform Generator
=================================================================
File Path           : .../echo_shield/data/test.wav
Sample Rate         : 44100 Hz
Total Samples       : 132,300
Channels            : 1 (Mono)
Audio Data Type     : int16
Duration            : 3.000 seconds
Min Sample Value    : -16384
Max Sample Value    : 16383
-----------------------------------------------------------------
[SUCCESS] Waveform saved to: .../echo_shield/outputs/step01_waveform.png
=================================================================
```

#### Output File:
A PNG image of the time-domain waveform will be saved at:
```text
outputs/<filename>_waveform.png
```

---

## Step 02: FFT-Based Frequency Spectrum Analysis

### 1. Time-Domain vs. Frequency-Domain
- **Time-Domain (Step 01):** Shows how the signal's amplitude changes over time ($x = \text{Time}$, $y = \text{Amplitude}$). It tells us *when* events happen and *how loud* the sound is, but cannot tell us which specific tones or pitches are present.
- **Frequency-Domain (Step 02):** Decomposes the signal into its individual sinusoidal frequency components ($x = \text{Frequency (Hz)}$, $y = \text{Magnitude}$). It reveals *what specific pitches/frequencies* make up the sound.

### 2. What FFT Does
The **Fast Fourier Transform (FFT)** is an efficient mathematical algorithm that computes the Discrete Fourier Transform (DFT). It transforms complex acoustic sound waves (like engine hum, rotor blades, gunfire) into a spectrum of individual frequencies and their corresponding power/magnitude.

### 3. Running the Script
Run Step 02 with the default file or any custom audio file:

```bash
# Default file (data/test.wav):
python audio_lab/step02_fft.py

# Custom defence vehicle audio:
python audio_lab/step02_fft.py data/tank.wav
python audio_lab/step02_fft.py data/heli.wav
```

### 4. What the Graph Means
- **X-axis (Frequency in Hz):** Represents pitch/frequency (e.g., low engine rumblings at 20–200 Hz, human speech at 300–3400 Hz, high-pitch whines above 4000 Hz).
- **Y-axis (Magnitude):** Represents the strength or amplitude of each frequency component.
- **Peaks:** Sharp vertical peaks indicate strong harmonic tones or resonant frequencies (e.g., engine firing rates or rotor blade pass frequencies).
- **Saved Output:** Saved automatically to `outputs/<filename>_fft.png`.

---

## Step 03: Time-Frequency Analysis using STFT & Spectrogram

### 1. Limitation of a Full-Recording FFT
A standard full-recording FFT calculates the frequency content across the entire audio duration at once. While it tells us *what* frequencies are present, it completely averages out time: we cannot tell *when* a frequency started, changed, or ended. In defence vehicle acoustic environments (e.g., engine acceleration, gear shifts, gunfire, changing rotor speeds), the noise is highly dynamic (non-stationary).

### 2. What STFT Is
The **Short-Time Fourier Transform (STFT)** breaks the audio signal into small overlapping time segments (windows) and computes the FFT on each segment individually. This allows us to track how the frequency spectrum evolves over time.

### 3. What a Spectrogram Shows
A **Spectrogram** is a 2D visual representation of the STFT magnitude over time:
- **X-axis (Time in seconds):** When the acoustic events occur.
- **Y-axis (Frequency in Hz):** The frequency content from 0 Hz up to the Nyquist frequency ($f_s / 2$).
- **Color / Intensity (Magnitude):** The energy or strength of a particular frequency at a given moment in time (brighter/warmer colors = higher energy).

### 4. Running the Script
Run Step 03 with the default file or any custom audio file:

```bash
# Default file (data/test.wav):
python audio_lab/step03_spectrogram.py

# Custom defence vehicle audio:
python audio_lab/step03_spectrogram.py data/tank.wav
python audio_lab/step03_spectrogram.py data/heli.wav
```

### 5. Interpreting the Spectrogram
- **Horizontal Lines / Bands:** Continuous stationary or tonal noise at specific frequencies over time (e.g., steady engine hum, generator whining, constant rotor harmonics).
- **Vertical Bursts / Streaks:** Sudden broadband acoustic impulses spanning a wide range of frequencies instantaneously (e.g., gunfire, explosions, mechanical impacts).
- **Bright / High-Intensity Regions:** High acoustic power at that specific time and frequency bin.
- **Saved Output:** Saved automatically to `outputs/<filename>_spectrogram.png`.

---

## Step 04 - Audio Feature Extraction

### 1. Overview & Feature Descriptions
Step 04 computes key numerical acoustic descriptors directly from the audio signal and its FFT magnitude spectrum:
- **RMS Energy:** Measures total signal strength and perceived acoustic energy ($\text{RMS} = \sqrt{\frac{1}{N}\sum x^2}$).
- **Zero Crossing Rate (ZCR):** Measures the rate at which the audio signal changes sign across adjacent samples. High ZCR values often indicate noisy, high-frequency, or percussive sounds (e.g., gunfire, metal friction), while low ZCR values characterize smooth low-frequency tones (e.g., diesel engines).
- **Spectral Centroid:** Represents the "center of mass" of the frequency spectrum ($\sum (f \cdot |X(f)|) / \sum |X(f)|$). It indicates the brightness or average frequency center of the sound.
- **Spectral Rolloff:** Identifies the frequency below which 85% of the total spectral magnitude is contained. It helps differentiate between low-frequency concentrated noise (e.g., tanks) and broadband noise (e.g., gunfire, artillery).
- **Band Energy:** Shows how acoustic energy is partitioned across critical frequency sub-bands (0–500 Hz, 500–2000 Hz, 2000–5000 Hz, and 5000–10000 Hz).

### 2. Why These Features Are Useful for Noise Classification
In defence vehicle environments, raw audio waveforms contain millions of raw sample values which are redundant and difficult to feed directly into lightweight classifiers. These extracted scalar features act as compact acoustic signatures:
- **Tanks / Heavy Armor:** High energy in the 0–500 Hz band, low spectral centroid, and low ZCR.
- **Helicopters / Rotors:** Characteristic blade passing frequency harmonics with strong energy in low-to-mid bands (0–2000 Hz).
- **Gunfire / Weapon Bursts:** High RMS transients, high ZCR, and elevated spectral rolloff due to broadband high-frequency dispersion.

### 3. Running Step 04
```bash
# Default file (data/test.wav):
python audio_lab/step04_features.py

# Defence vehicle noise audio:
python audio_lab/step04_features.py data/gun_fight.wav
python audio_lab/step04_features.py data/tank.wav
python audio_lab/step04_features.py data/heli.wav
```

### 4. Saved Output
Extracted features are automatically saved as a tabular CSV file at:
```text
outputs/<filename>_features.csv
```

---

## Step 05 - Windowed Feature Extraction

### 1. Overview
In real-world scenarios, acoustic environments change dynamically over time. Step 05 divides a long audio recording into short, overlapping windows and computes an individual feature vector for each window.
- **Window Length:** 100 ms (captures local frequency and energy dynamics)
- **Hop Size:** 50 ms (50% overlap ensuring smooth temporal continuity)

### 2. Why Windowed Features Matter
- Instead of producing a single averaged feature row for an entire recording (which loses temporal variations), windowing produces a sequence of feature rows tracking changes over time.
- Each row contains: RMS Energy, Zero Crossing Rate, Spectral Centroid, Spectral Rolloff, and Sub-band Energies.
- This structured time-series tabular data provides the dataset format needed for downstream acoustic characterization and noise classification.
- *(Note: This is an offline batch analysis on recorded WAV files and is not yet real-time embedded processing).*

### 3. Running Step 05
```bash
# Default file (data/test.wav):
python audio_lab/step05_window_features.py

# Defence vehicle noise audio:
python audio_lab/step05_window_features.py data/gun_fight.wav
python audio_lab/step05_window_features.py data/tank.wav
python audio_lab/step05_window_features.py data/heli.wav
```

### 4. Saved Output
The windowed feature matrix is saved to:
```text
outputs/<filename>_window_features.csv
```

---

## Step 06 - ML Dataset Builder

### 1. Overview
Step 06 aggregates the window-level feature vectors generated in Step 05 across multiple distinct noise recordings into a unified supervised dataset:
- `outputs/tank_window_features.csv` $\rightarrow$ labelled as `tank`
- `outputs/heli_window_features.csv` $\rightarrow$ labelled as `helicopter`
- `outputs/gun_fight_window_features.csv` $\rightarrow$ labelled as `gun_fight`

> **Note:** `test.wav` is intentionally excluded from the training dataset as it is a synthetic test audio file.

### 2. Supervised Learning Dataset
- Each row in the resulting dataset represents a 100 ms audio window with its acoustic feature vector and ground-truth noise class `label`.
- This structured table provides the exact format required for training downstream lightweight machine learning classifiers (e.g., to identify whether incoming noise is a tank, helicopter, or battlefield gunfire).
- *(Note: This step constructs the dataset only; no ML model is trained yet).*

### 3. Running Step 06
```bash
python audio_lab/step06_build_dataset.py
```

### 4. Saved Output
The aggregated training dataset is saved to:
```text
outputs/noise_dataset.csv
```

---

## Step 07 - Lightweight Noise Classifier

### 1. Overview
Step 07 trains a baseline machine learning classifier (`DecisionTreeClassifier`) using the 8 acoustic features extracted from 100 ms audio windows to categorize defence vehicle noise profiles (`tank`, `helicopter`, `gun_fight`).

### 2. Decision Tree Architecture & Training
- **Input Features (8):** RMS Energy, Zero Crossing Rate, Spectral Centroid, Spectral Rolloff, and 4 Sub-band Energies (0–500 Hz, 500–2000 Hz, 2000–5000 Hz, 5000–10000 Hz).
- **Target Variable:** Ground truth class label (`tank`, `helicopter`, `gun_fight`).
- **Data Partitioning:** Stratified 80% train / 20% test split to maintain exact class proportions across partitions.
- **Model Constraints:** `max_depth=5` to prevent overfitting and ensure fast, lightweight decision boundaries.

### 3. Role in the Echo Shield System
- The classifier enables **noise understanding and acoustic environment awareness** (e.g., detecting if the system is currently inside an armored vehicle, rotorcraft, or active firing zone).
- **Important Notes:**
  - This model **does NOT generate anti-noise** and **does NOT perform Active Noise Cancellation (ANC)**.
  - It operates purely in software on windowed features and does not represent real-time embedded FPGA execution yet.
  - Accuracy and performance metrics are strictly evaluated on the held-out test split.

### 4. Running Step 07
```bash
python audio_lab/step07_train_classifier.py
```

### 5. Saved Output
The trained serialized model is saved to:
```text
outputs/noise_classifier.joblib
```

---

## Step 08 - Noise Prediction Demo

### 1. Overview
Step 08 loads the trained decision tree classifier (`outputs/noise_classifier.joblib`) without retraining and performs end-to-end inference on any arbitrary input WAV audio file.

### 2. Inference Workflow
1. **Audio Ingestion:** Loads the target WAV file and selects Channel 1.
2. **Dynamic Windowing:** Partitions the signal into consecutive 100 ms windows advanced by 50 ms hop intervals.
3. **Feature Extraction:** Extracts the exact 8 acoustic features per window (RMS Energy, ZCR, Spectral Centroid, Spectral Rolloff, and 4 Sub-band Energies).
4. **Window Classification:** The trained model evaluates each window independently.
5. **Majority Voting:** Aggregates window-level predictions across the duration to determine the dominant overall acoustic environment.
6. **Diagnostic Probability:** Reports average leaf probabilities across all evaluated windows.

### 3. Important Clarifications
- This script is an offline software demonstration of ML acoustic environment classification.
- It does **not** perform physical Active Noise Cancellation (ANC) or generate anti-noise audio.
- It does **not** represent embedded real-time FPGA execution.
- `test.wav` is a demonstration audio file; accuracy is not calculated for unlabelled demonstration files without verified ground-truth labels.

### 4. Running Step 08
```bash
# Test on synthetic demonstration audio:
python audio_lab/step08_predict_noise.py data/test.wav

# Test on defence vehicle recordings:
python audio_lab/step08_predict_noise.py data/tank.wav
python audio_lab/step08_predict_noise.py data/heli.wav
python audio_lab/step08_predict_noise.py data/gun_fight.wav
```







#   E c h o  
 