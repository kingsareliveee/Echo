# ==============================================================================
# Echo Shield — Shared Utility Module
# Project: AI/ML-enabled Adaptive Noise Cancellation for Defence Vehicles
# PS ID: SIH26052 | Team: Echo Shield | Theme: Smart Vehicles
#
# This module wraps the DSP / ML logic from Steps 01–20 into importable
# functions so that app.py (Streamlit dashboard) can call them without
# duplicating or altering the validated algorithms.
#
# Wrapping notes per step:
#   Steps 01-05  → load_audio, compute_fft, compute_spectrogram,
#                  extract_global_features, extract_window_features
#   Steps 10-11  → PROFILE_LOOKUP, DSP_CONFIG_LOOKUP (reused verbatim)
#   Steps 12-13  → run_lms, run_nlms
#   Steps 15-16  → _build_synthetic_secondary_path, run_fxlms
#   Step  18     → run_wavelet_denoise
#   Step  20     → METRICS_DATA (reused verbatim)
#   Steps 06-07  → skipped (dataset / model already built)
#   Steps 09,14,17,19,21,22 → superseded by this integrated module
# ==============================================================================

import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft

import warnings

try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

try:
    import pywt
    _PYWT_AVAILABLE = True
except ImportError:
    _PYWT_AVAILABLE = False

# Suppress sklearn model-persistence version mismatch warning.
# The model was trained under sklearn 1.7.2 and loads cleanly under 1.9.0.
# Results are validated by smoke_test.py — no breakage detected.
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass  # older sklearn without this exception class

# ---------------------------------------------------------------------------
# Constants — replicated verbatim from Steps 10, 11, 20
# ---------------------------------------------------------------------------

# From step10_noise_profile.py — PROFILE_LOOKUP dict
PROFILE_LOOKUP = {
    "gun_fight": {
        "processing_profile": "IMPULSIVE_PROFILE",
        "reason": "Transient/impulsive noise profile",
    },
    "helicopter": {
        "processing_profile": "LOW_FREQ_TONAL_PROFILE",
        "reason": "Low-frequency and tonal noise profile",
    },
    "tank": {
        "processing_profile": "LOW_FREQ_MECHANICAL_PROFILE",
        "reason": "Low-frequency mechanical noise profile",
    },
}

# From step11_dsp_config.py — DSP_CONFIG_LOOKUP dict
DSP_CONFIG_LOOKUP = {
    "tank": {
        "profile": "LOW_FREQ_MECHANICAL_PROFILE",
        "filter_length": 32,
        "step_size_mu": 0.005,
        "processing_mode": "LOW_FREQUENCY_ADAPTIVE",
    },
    "helicopter": {
        "profile": "LOW_FREQ_TONAL_PROFILE",
        "filter_length": 32,
        "step_size_mu": 0.003,
        "processing_mode": "LOW_FREQ_TONAL_ADAPTIVE",
    },
    "gun_fight": {
        "profile": "IMPULSIVE_PROFILE",
        "filter_length": 32,
        "step_size_mu": 0.001,
        "processing_mode": "IMPULSIVE_PROTECTION",
    },
}

# Default DSP config when class is unknown / test file
DSP_CONFIG_DEFAULT = {
    "profile": "GENERIC_PROFILE",
    "filter_length": 32,
    "step_size_mu": 0.001,
    "processing_mode": "DEFAULT_MODE",
}

# Windowing constants — identical to Steps 05, 08, 10, 11, 19, 22
WINDOW_DURATION_MS = 100
HOP_DURATION_MS = 50

# Sub-band definitions — identical to Steps 05, 08, 10, 11, 19, 22
BANDS = [
    ("band_energy_0_500",    0.0,    500.0),
    ("band_energy_500_2000", 500.0,  2000.0),
    ("band_energy_2000_5000",2000.0, 5000.0),
    ("band_energy_5000_10000",5000.0,10000.0),
]

# From step20_evaluation.py — METRICS_DATA (verbatim, see that file for provenance)
METRICS_DATA = [
    {
        "module": "ML Noise Classifier (Decision Tree)",
        "metric": "Classification Accuracy (Held-out Test Split)",
        "value": "98.97%",
        "numeric_val": 98.97,
        "type": "MODEL TEST",
        "notes": "Evaluated on 290 stratified held-out 100 ms test windows",
    },
    {
        "module": "ML Noise Classifier (Decision Tree)",
        "metric": "Macro F1-Score",
        "value": "0.97",
        "numeric_val": 97.0,
        "type": "MODEL TEST",
        "notes": "Balanced across tank (0.93), heli (1.00), gun_fight (0.99)",
    },
    {
        "module": "DSP Adaptive Filter (Standard LMS)",
        "metric": "RMS Noise Reduction (tank.wav)",
        "value": "20.65 dB",
        "numeric_val": 20.65,
        "type": "MEASURED SOFTWARE EXPERIMENT",
        "notes": "32 taps, mu=0.005, sample-by-sample loop",
    },
    {
        "module": "DSP Adaptive Filter (Normalized LMS)",
        "metric": "RMS Noise Reduction (tank.wav)",
        "value": "31.58 dB",
        "numeric_val": 31.58,
        "type": "MEASURED SOFTWARE EXPERIMENT",
        "notes": "32 taps, mu_norm=0.5, epsilon=1e-8",
    },
    {
        "module": "Active Noise Cancellation (FxLMS)",
        "metric": "Simulated Acoustic Error Reduction (tank.wav)",
        "value": "11.89 dB",
        "numeric_val": 11.89,
        "type": "SIMULATION",
        "notes": "Offline simulation using 12-tap synthetic S(z) secondary path",
    },
    {
        "module": "Speech Path (Wavelet Denoising)",
        "metric": "Signal RMS Attenuation (db4, Level 5)",
        "value": "0.02 dB",
        "numeric_val": 0.02,
        "type": "MEASURED SOFTWARE EXPERIMENT",
        "notes": "Soft-thresholding on detail sub-bands (DWT/IDWT)",
    },
    {
        "module": "Embedded Hardware — ARM SoC",
        "metric": "Real-Time Processing Latency",
        "value": "NOT MEASURED",
        "numeric_val": None,
        "type": "NOT MEASURED",
        "notes": "Pending Hardware Validation — ARM deployment phase",
    },
    {
        "module": "Physical Vehicle Acoustic ANC",
        "metric": "In-Cabin SPL Attenuation",
        "value": "NOT MEASURED",
        "numeric_val": None,
        "type": "NOT MEASURED",
        "notes": "Requires physical microphones, speakers, and in-vehicle field tests",
    },
    {
        "module": "Physical Vehicle Acoustic ANC",
        "metric": "Power Consumption",
        "value": "NOT MEASURED",
        "numeric_val": None,
        "type": "NOT MEASURED",
        "notes": "Pending Hardware Validation — power profiling on ARM SoC",
    },
]

# ---------------------------------------------------------------------------
# Step 01 / 02 / 03 — Audio loading & normalization
# ---------------------------------------------------------------------------

def load_audio(path: str):
    """
    Load a WAV file and return channel-1 float64 signal normalised to [-1, 1].
    Wraps the load + normalise logic from step01 & step12 (both use the same
    approach).

    Returns
    -------
    signal      : np.ndarray float64, normalised to [-1, 1]
    raw_signal  : np.ndarray float64, unnormalised (for feature extraction,
                  matching Steps 04-08 which use raw scale)
    sample_rate : int
    num_channels: int
    duration_sec: float
    """
    sample_rate, audio_data = wavfile.read(path)

    if audio_data.ndim == 1:
        num_channels = 1
        ch1 = audio_data
    else:
        num_channels = audio_data.shape[1]
        ch1 = audio_data[:, 0]

    # Raw (unnormalised) float64 — matching Steps 04-08 feature extraction
    raw_signal = ch1.astype(np.float64)

    # Normalised to [-1, 1] — matching Steps 12-16 DSP arithmetic
    raw_max = float(np.max(np.abs(ch1)))
    if raw_max > 0:
        if np.issubdtype(ch1.dtype, np.integer):
            norm_factor = 32768.0 if np.issubdtype(ch1.dtype, np.int16) else raw_max
        else:
            norm_factor = raw_max
        signal = raw_signal / norm_factor
    else:
        signal = raw_signal.copy()

    duration_sec = len(signal) / sample_rate
    return signal, raw_signal, sample_rate, num_channels, duration_sec


# ---------------------------------------------------------------------------
# Step 02 — FFT frequency spectrum
# ---------------------------------------------------------------------------

def compute_fft(signal: np.ndarray, sample_rate: int):
    """
    Compute positive-frequency FFT magnitude spectrum.
    Algorithm identical to step02_fft.py §§ 5-7.

    Returns
    -------
    freq_axis     : np.ndarray  Hz axis, 0 to Nyquist
    magnitude     : np.ndarray  normalised magnitude
    dominant_freq : float       frequency of the largest non-DC peak (Hz)
    dominant_mag  : float       magnitude at dominant_freq
    """
    num_samples = len(signal)
    raw_fft = np.fft.fft(signal)
    half_n = num_samples // 2

    magnitude = (2.0 / num_samples) * np.abs(raw_fft[:half_n])
    if len(magnitude) > 0:
        magnitude[0] = magnitude[0] / 2.0

    freq_axis = np.linspace(0.0, sample_rate / 2.0, half_n, endpoint=False)

    if len(magnitude) > 1:
        dominant_idx = 1 + int(np.argmax(magnitude[1:]))
    else:
        dominant_idx = 0

    dominant_freq = float(freq_axis[dominant_idx])
    dominant_mag  = float(magnitude[dominant_idx])

    return freq_axis, magnitude, dominant_freq, dominant_mag


# ---------------------------------------------------------------------------
# Step 03 — STFT Spectrogram
# ---------------------------------------------------------------------------

def compute_spectrogram(signal: np.ndarray, sample_rate: int):
    """
    Compute STFT spectrogram.
    Algorithm identical to step03_spectrogram.py § 6.

    Returns
    -------
    frequencies : np.ndarray
    times       : np.ndarray
    magnitude   : np.ndarray  |Zxx|
    """
    frequencies, times, zxx = stft(
        signal,
        fs=sample_rate,
        window="hann",
        nperseg=1024,
        noverlap=512,
    )
    magnitude = np.abs(zxx)
    return frequencies, times, magnitude


# ---------------------------------------------------------------------------
# Step 04 — Global feature extraction (whole-signal)
# ---------------------------------------------------------------------------

def extract_global_features(signal: np.ndarray, sample_rate: int) -> dict:
    """
    Extract the five global acoustic descriptors from the whole signal.
    Algorithm identical to step04_features.py §§ A-D.
    Uses unnormalised raw_signal (matching Steps 04-08).

    Returns dict with: rms, zcr, spectral_centroid, spectral_rolloff,
    band_energies (dict), nyquist_freq, duration_sec.
    """
    num_samples = len(signal)
    nyquist_freq = sample_rate / 2.0
    duration_sec = num_samples / sample_rate

    # A. RMS Energy
    rms_energy = float(np.sqrt(np.mean(signal ** 2)))

    # B. Zero Crossing Rate
    if num_samples > 1:
        zero_crossings = int(np.sum(np.abs(np.diff(np.signbit(signal)))))
        zcr = float(zero_crossings / (num_samples - 1))
    else:
        zcr = 0.0

    # FFT magnitude spectrum
    raw_fft = np.fft.fft(signal)
    half_n = num_samples // 2
    magnitude = (2.0 / num_samples) * np.abs(raw_fft[:half_n])
    if len(magnitude) > 0:
        magnitude[0] = magnitude[0] / 2.0
    freq_axis = np.linspace(0.0, nyquist_freq, half_n, endpoint=False)
    total_magnitude = float(np.sum(magnitude))

    # C. Spectral Centroid
    if total_magnitude > 0:
        spectral_centroid = float(np.sum(freq_axis * magnitude) / total_magnitude)
    else:
        spectral_centroid = 0.0

    # D. Spectral Rolloff (85%)
    if total_magnitude > 0:
        cumulative = np.cumsum(magnitude)
        rolloff_idx = int(np.searchsorted(cumulative, 0.85 * total_magnitude))
        rolloff_idx = min(rolloff_idx, len(freq_axis) - 1)
        spectral_rolloff = float(freq_axis[rolloff_idx])
    else:
        spectral_rolloff = 0.0

    # E. Band Energies
    band_energies = {}
    for band_label, f_low, f_high in BANDS:
        if f_low < nyquist_freq:
            f_high_actual = min(f_high, nyquist_freq)
            mask = (freq_axis >= f_low) & (freq_axis < f_high_actual)
            band_energies[band_label] = float(np.sum(magnitude[mask]))
        else:
            band_energies[band_label] = 0.0

    return {
        "rms": rms_energy,
        "zcr": zcr,
        "spectral_centroid": spectral_centroid,
        "spectral_rolloff": spectral_rolloff,
        "band_energies": band_energies,
        "nyquist_freq": nyquist_freq,
        "duration_sec": duration_sec,
    }


# ---------------------------------------------------------------------------
# Step 05 / 08 — Per-window feature extraction
# ---------------------------------------------------------------------------

def extract_window_features(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Extract the 8-feature vector for every 100 ms window (50 ms hop).
    Algorithm identical to step05_window_features.py §§ 6-7 and step08.

    Parameters
    ----------
    signal      : raw (unnormalised) float64 signal — matching Steps 05, 08
    sample_rate : int

    Returns
    -------
    X : np.ndarray shape (N_windows, 8)
        Columns: rms, zcr, centroid, rolloff, band_0_500, band_500_2000,
                 band_2000_5000, band_5000_10000
    """
    total_samples = len(signal)
    nyquist_freq = sample_rate / 2.0
    window_samples = int(sample_rate * (WINDOW_DURATION_MS / 1000.0))
    hop_samples    = int(sample_rate * (HOP_DURATION_MS    / 1000.0))

    if total_samples < window_samples:
        return np.empty((0, 8), dtype=np.float64)

    half_n    = window_samples // 2
    freq_axis = np.linspace(0.0, nyquist_freq, half_n, endpoint=False)

    rows = []
    start = 0
    while start + window_samples <= total_samples:
        w_sig = signal[start : start + window_samples]

        rms = float(np.sqrt(np.mean(w_sig ** 2)))
        zcr = (float(np.sum(np.abs(np.diff(np.signbit(w_sig)))) / (window_samples - 1))
               if window_samples > 1 else 0.0)

        raw_fft = np.fft.fft(w_sig)
        mag = (2.0 / window_samples) * np.abs(raw_fft[:half_n])
        if len(mag) > 0:
            mag[0] /= 2.0
        tot_mag = float(np.sum(mag))

        centroid = float(np.sum(freq_axis * mag) / tot_mag) if tot_mag > 0 else 0.0

        if tot_mag > 0:
            cum_mag = np.cumsum(mag)
            r_idx   = min(int(np.searchsorted(cum_mag, 0.85 * tot_mag)), len(freq_axis) - 1)
            rolloff = float(freq_axis[r_idx])
        else:
            rolloff = 0.0

        b_energies = []
        for _, f_low, f_high in BANDS:
            if f_low < nyquist_freq:
                fh_act = min(f_high, nyquist_freq)
                b_energies.append(float(np.sum(mag[(freq_axis >= f_low) & (freq_axis < fh_act)])))
            else:
                b_energies.append(0.0)

        rows.append([rms, zcr, centroid, rolloff] + b_energies)
        start += hop_samples

    return np.array(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# Step 07 / 08 — Model loading & inference
# ---------------------------------------------------------------------------

def load_model(model_path: str):
    """
    Load the trained Decision Tree classifier from disk.
    Returns (model, None) on success or (None, error_message) on failure.
    """
    if not _JOBLIB_AVAILABLE:
        return None, "joblib is not installed. Run: pip install joblib"
    if not os.path.isfile(model_path):
        return None, f"Model file not found: {model_path}"
    try:
        model = joblib.load(model_path)
        return model, None
    except Exception as exc:
        return None, f"Failed to load model: {exc}"


def classify_audio(raw_signal: np.ndarray, sample_rate: int, model):
    """
    Run window-level feature extraction + predict with majority voting.
    Mirrors step08_predict_noise.py §§ 6-9.

    Returns
    -------
    majority_class  : str
    window_preds    : np.ndarray of per-window predictions
    avg_probas      : np.ndarray (n_classes,) or None if predict_proba unavailable
    vote_pcts       : dict  class → percentage of windows
    """
    from collections import Counter

    X = extract_window_features(raw_signal, sample_rate)
    if len(X) == 0:
        return "unknown", np.array([]), None, {}

    window_preds = model.predict(X)
    counts       = Counter(window_preds)
    majority_class = counts.most_common(1)[0][0]
    num_windows  = len(window_preds)

    vote_pcts = {cls: (counts.get(cls, 0) / num_windows) * 100.0
                 for cls in model.classes_}

    avg_probas = None
    if hasattr(model, "predict_proba"):
        probas     = model.predict_proba(X)
        avg_probas = np.mean(probas, axis=0)

    return majority_class, window_preds, avg_probas, vote_pcts


# ---------------------------------------------------------------------------
# Step 12 — Standard LMS adaptive filter
# ---------------------------------------------------------------------------

def run_lms(signal: np.ndarray, filter_length: int = 32, mu: float = 0.005):
    """
    Run the Standard LMS adaptive filter sample-by-sample.
    Algorithm identical to step12_lms_filter.py § 5.

    Parameters
    ----------
    signal        : normalised float64 signal
    filter_length : number of taps (weights)
    mu            : step size

    Returns
    -------
    e_error          : np.ndarray  error signal e[n]
    rms_reduction_db : float       20*log10(input_rms / error_rms)
    input_rms        : float
    error_rms        : float
    """
    num_samples = len(signal)
    weights  = np.zeros(filter_length, dtype=np.float64)
    x_buffer = np.zeros(filter_length, dtype=np.float64)
    e_error  = np.zeros(num_samples,   dtype=np.float64)

    for n in range(num_samples):
        d_n   = signal[n]
        y_n   = np.dot(weights, x_buffer)
        e_n   = d_n - y_n
        e_error[n] = e_n
        weights   += mu * e_n * x_buffer
        x_buffer[1:] = x_buffer[:-1]
        x_buffer[0]  = d_n

    input_rms = float(np.sqrt(np.mean(signal ** 2)))
    error_rms = float(np.sqrt(np.mean(e_error ** 2)))
    rms_reduction_db = (20.0 * np.log10(input_rms / error_rms)
                        if error_rms > 0 and input_rms > 0 else 0.0)

    return e_error, rms_reduction_db, input_rms, error_rms


# ---------------------------------------------------------------------------
# Step 13 — Normalised LMS (NLMS) adaptive filter
# ---------------------------------------------------------------------------

def run_nlms(signal: np.ndarray, filter_length: int = 32,
             mu: float = 0.5, eps: float = 1e-8):
    """
    Run the Normalised LMS (NLMS) adaptive filter sample-by-sample.
    Algorithm identical to step13_lms_vs_nlms.py §§ 5-6 (NLMS branch).

    Returns same tuple as run_lms.
    """
    num_samples  = len(signal)
    weights_nlms = np.zeros(filter_length, dtype=np.float64)
    buffer_nlms  = np.zeros(filter_length, dtype=np.float64)
    e_nlms       = np.zeros(num_samples,   dtype=np.float64)

    for n in range(num_samples):
        d_n   = signal[n]
        y_n   = np.dot(weights_nlms, buffer_nlms)
        err   = d_n - y_n
        e_nlms[n] = err
        buf_energy = float(np.dot(buffer_nlms, buffer_nlms))
        norm_mu    = mu / (eps + buf_energy)
        weights_nlms += norm_mu * err * buffer_nlms
        buffer_nlms[1:] = buffer_nlms[:-1]
        buffer_nlms[0]  = d_n

    input_rms = float(np.sqrt(np.mean(signal ** 2)))
    error_rms = float(np.sqrt(np.mean(e_nlms ** 2)))
    rms_reduction_db = (20.0 * np.log10(input_rms / error_rms)
                        if error_rms > 0 and input_rms > 0 else 0.0)

    return e_nlms, rms_reduction_db, input_rms, error_rms


# ---------------------------------------------------------------------------
# Step 15 — Synthetic secondary path model
# ---------------------------------------------------------------------------

def _build_synthetic_secondary_path(sec_path_len: int = 12,
                                     delay_samples: int = 2) -> np.ndarray:
    """
    Build the synthetic 12-tap FIR secondary-path impulse response.
    Algorithm identical to step15_secondary_path.py § 2 and step16/17/19/22.
    NOT measured hardware data — synthetic simulation model only.
    """
    sp = np.zeros(sec_path_len, dtype=np.float64)
    for i in range(delay_samples, sec_path_len):
        sp[i] = 0.8 * (0.6 ** (i - delay_samples)) * np.cos(0.5 * (i - delay_samples))
    norm = np.sum(np.abs(sp))
    if norm > 0:
        sp /= norm
    return sp


# ---------------------------------------------------------------------------
# Step 16 / 17 — FxLMS ANC simulation
# ---------------------------------------------------------------------------

def run_fxlms(signal: np.ndarray, filter_length: int = 32, mu: float = 0.0005):
    """
    Run the Filtered-X LMS (FxLMS) ANC simulation sample-by-sample.
    Algorithm identical to step16_fxlms.py § 4 and step17_anc_simulation.py § 4.

    NOTE: Secondary path is SYNTHETIC — not a measured hardware path.
    This is an offline simulation; no physical acoustic cancellation occurs.

    Returns
    -------
    e_error          : np.ndarray  residual error e[n]
    anti_noise       : np.ndarray  anti-noise command y[n]
    rms_reduction_db : float
    input_rms        : float
    error_rms        : float
    """
    sec_path    = _build_synthetic_secondary_path()
    sec_path_len = len(sec_path)
    num_samples = len(signal)

    weights       = np.zeros(filter_length,  dtype=np.float64)
    x_ref_buf     = np.zeros(filter_length,  dtype=np.float64)
    x_filt_buf    = np.zeros(filter_length,  dtype=np.float64)
    sec_x_history = np.zeros(sec_path_len,   dtype=np.float64)
    y_history     = np.zeros(sec_path_len,   dtype=np.float64)

    e_error   = np.zeros(num_samples, dtype=np.float64)
    anti_noise = np.zeros(num_samples, dtype=np.float64)

    for n in range(num_samples):
        x_n = signal[n]
        d_n = signal[n]

        sec_x_history[1:] = sec_x_history[:-1]
        sec_x_history[0]  = x_n
        x_f_n = np.dot(sec_path, sec_x_history)

        x_filt_buf[1:] = x_filt_buf[:-1]
        x_filt_buf[0]  = x_f_n

        y_n = np.dot(weights, x_ref_buf)
        anti_noise[n] = y_n

        y_history[1:] = y_history[:-1]
        y_history[0]  = y_n
        y_s_n = np.dot(sec_path, y_history)

        e_n = d_n - y_s_n
        e_error[n] = e_n

        weights    += mu * e_n * x_filt_buf
        x_ref_buf[1:] = x_ref_buf[:-1]
        x_ref_buf[0]  = x_n

    input_rms = float(np.sqrt(np.mean(signal ** 2)))
    error_rms = float(np.sqrt(np.mean(e_error ** 2)))
    rms_reduction_db = (20.0 * np.log10(input_rms / error_rms)
                        if error_rms > 0 and input_rms > 0 else 0.0)

    return e_error, anti_noise, rms_reduction_db, input_rms, error_rms


# ---------------------------------------------------------------------------
# Step 18 — Wavelet denoising & speech reconstruction
# ---------------------------------------------------------------------------

def run_wavelet_denoise(signal: np.ndarray, wavelet_name: str = "db4",
                        max_level: int = 5):
    """
    Apply Donoho-Johnstone VisuShrink soft-thresholding wavelet denoising.
    Algorithm identical to step18_wavelet_reconstruction.py §§ 3-4.

    Returns
    -------
    denoised   : np.ndarray
    sigma      : float   estimated noise std
    threshold  : float   soft-threshold value T
    rms_out    : float   RMS of denoised signal
    level_used : int     actual DWT decomposition level used
    """
    if not _PYWT_AVAILABLE:
        # Fallback: return original signal unchanged
        return signal.copy(), 0.0, 0.0, float(np.sqrt(np.mean(signal ** 2))), 0

    num_samples = len(signal)
    wavelet     = pywt.Wavelet(wavelet_name)
    max_possible = pywt.dwt_max_level(num_samples, wavelet.dec_len)
    level_used  = min(max_level, max_possible)

    coeffs = pywt.wavedec(signal, wavelet_name, level=level_used)

    detail_cd1 = coeffs[-1]
    sigma      = float(np.median(np.abs(detail_cd1)) / 0.6745)
    threshold  = sigma * np.sqrt(2.0 * np.log(max(num_samples, 1)))

    thresholded = [coeffs[0]] + [
        pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]
    ]
    denoised = pywt.waverec(thresholded, wavelet_name)

    if len(denoised) > num_samples:
        denoised = denoised[:num_samples]
    elif len(denoised) < num_samples:
        denoised = np.pad(denoised, (0, num_samples - len(denoised)))

    rms_out = float(np.sqrt(np.mean(denoised ** 2)))
    return denoised, sigma, threshold, rms_out, level_used


# ---------------------------------------------------------------------------
# Convenience: full pipeline in one call (used by app.py run button)
# ---------------------------------------------------------------------------

def run_full_pipeline(wav_path: str, model, max_anc_samples: int = 60_000):
    """
    Run the complete Echo Shield pipeline on a WAV file.

    Pipeline steps (matching Step 22 architecture):
      Load → Features → Classify → Profile → DSP Config
        → LMS → NLMS → FxLMS (capped) → Wavelet → Results

    Parameters
    ----------
    wav_path        : absolute path to WAV file
    model           : loaded sklearn classifier (from load_model)
    max_anc_samples : cap on samples for ANC simulation (for demo speed).
                      Clearly labelled in UI as demo-only cap.

    Returns
    -------
    dict with all results needed by app.py panels.
    """
    # 1. Load audio
    signal, raw_signal, sample_rate, num_channels, duration_sec = load_audio(wav_path)
    num_samples = len(signal)

    # 2. Global feature extraction (whole-signal, raw scale — Steps 04 logic)
    global_feats = extract_global_features(raw_signal, sample_rate)

    # 3. FFT spectrum
    freq_axis, fft_mag, dominant_freq, dominant_mag = compute_fft(signal, sample_rate)

    # 4. Spectrogram
    spec_freqs, spec_times, spec_mag = compute_spectrogram(signal, sample_rate)

    # 5. Classify
    majority_class, window_preds, avg_probas, vote_pcts = classify_audio(
        raw_signal, sample_rate, model
    )

    # 6. Noise profile + DSP config
    profile_info = PROFILE_LOOKUP.get(majority_class, {
        "processing_profile": "GENERIC_PROFILE",
        "reason": "Unspecified noise category",
    })
    dsp_cfg = DSP_CONFIG_LOOKUP.get(majority_class, DSP_CONFIG_DEFAULT)

    # 7. Adaptive filtering — on full signal (LMS/NLMS are fast in numpy)
    e_lms, lms_db, lms_in_rms, lms_err_rms = run_lms(
        signal, dsp_cfg["filter_length"], dsp_cfg["step_size_mu"]
    )
    e_nlms, nlms_db, nlms_in_rms, nlms_err_rms = run_nlms(
        signal, dsp_cfg["filter_length"]
    )

    # 8. FxLMS — capped for demo speed; label explicitly in UI
    anc_signal = signal[:max_anc_samples]
    anc_capped = len(signal) > max_anc_samples
    e_fxlms, anti_noise, fx_db, fx_in_rms, fx_err_rms = run_fxlms(
        anc_signal, filter_length=32, mu=0.0005
    )

    # 9. Wavelet denoising
    denoised, wv_sigma, wv_thresh, wv_rms, wv_level = run_wavelet_denoise(signal)

    # 10. AI Speech Enhancement (Step 24: Wiener + Multi-Band Wavelet Masking)
    enhanced_speech = None
    enhancement_metrics = {}
    try:
        from audio_lab.step24_speech_enhancement import enhance_speech
        enhanced_speech, enhancement_metrics = enhance_speech(
            signal, sample_rate=sample_rate, noise_type=majority_class
        )
    except Exception as _enh_err:
        enhanced_speech = denoised
        enhancement_metrics = {"error": str(_enh_err)}

    return {
        # audio meta
        "wav_path": wav_path,
        "sample_rate": sample_rate,
        "num_channels": num_channels,
        "num_samples": num_samples,
        "duration_sec": duration_sec,
        # raw signals (for plotting — downsampled by caller if needed)
        "signal": signal,
        "raw_signal": raw_signal,
        # FFT
        "fft_freq_axis": freq_axis,
        "fft_magnitude": fft_mag,
        "dominant_freq": dominant_freq,
        "dominant_mag": dominant_mag,
        # Spectrogram
        "spec_freqs": spec_freqs,
        "spec_times": spec_times,
        "spec_mag": spec_mag,
        # Features
        "global_feats": global_feats,
        # Classification
        "majority_class": majority_class,
        "window_preds": window_preds,
        "avg_probas": avg_probas,
        "vote_pcts": vote_pcts,
        "model_classes": list(model.classes_),
        # Profile & config
        "profile_info": profile_info,
        "dsp_cfg": dsp_cfg,
        # LMS
        "e_lms": e_lms,
        "lms_db": lms_db,
        "lms_in_rms": lms_in_rms,
        "lms_err_rms": lms_err_rms,
        # NLMS
        "e_nlms": e_nlms,
        "nlms_db": nlms_db,
        # FxLMS (capped)
        "e_fxlms": e_fxlms,
        "anti_noise": anti_noise,
        "fx_db": fx_db,
        "fx_in_rms": fx_in_rms,
        "fx_err_rms": fx_err_rms,
        "anc_capped": anc_capped,
        "anc_samples_used": len(anc_signal),
        # Wavelet
        "denoised": denoised,
        "wv_sigma": wv_sigma,
        "wv_thresh": wv_thresh,
        "wv_rms": wv_rms,
        "wv_level": wv_level,
        # AI Enhanced Clean Speech Output (Step 24)
        "enhanced_speech": enhanced_speech,
        "enhancement_metrics": enhancement_metrics,
    }
