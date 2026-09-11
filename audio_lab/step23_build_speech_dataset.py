"""
Step 23 - Build Clean-Speech + Defence-Noise Paired Dataset
===========================================================
Echo Shield | SIH 2026 | PS ID: SIH26052 | DRDO / Smart Vehicles

PURPOSE
-------
Create a reproducible, noisy-clean paired dataset for speech-enhancement
model training (future Step 24).

For each generated example:
  - clean speech  -> training TARGET
  - clean + noise -> model INPUT at a specified SNR

PIPELINE OVERVIEW
-----------------
1. Discover clean-speech WAVs from  data/clean_speech/
2. Discover defence/environmental noise from:
       data/defence_noise/   (optional custom noise drop-in folder)
       data/gun_fight.wav, data/heli.wav, data/tank.wav  (project baselines)
       data/ESC-50-master/   (public environmental audio, PROXY tier only)
3. Resample everything to TARGET_SR = 16 000 Hz, mono
4. For each (speech, noise, snr) triple:
       a. pick a random segment of noise matching speech length
       b. scale noise to requested SNR
       c. apply optional small-gain jitter and time-shift
       d. write noisy/{pair_id}.wav and clean/{pair_id}.wav
5. Generate metadata.csv and outputs/step23_dataset_report.txt
6. Verify SNR of a sample subset by re-reading pairs from disk

DIRECTORY LAYOUT PRODUCED
--------------------------
data/speech_dataset/
    train/
        noisy/   <pair_id>.wav
        clean/   <pair_id>.wav
    val/
        noisy/
        clean/
    test/
        noisy/
        clean/
    metadata.csv

outputs/
    step23_dataset_report.txt

USAGE
-----
python audio_lab/step23_build_speech_dataset.py           # default run
python audio_lab/step23_build_speech_dataset.py --help    # all options
python audio_lab/step23_build_speech_dataset.py --no-esc50              # CORE only
python audio_lab/step23_build_speech_dataset.py --snr -5 0 5 10 15     # custom SNRs
python audio_lab/step23_build_speech_dataset.py --max-pairs-per-noise 5 # smaller run

ADDING YOUR OWN SPEECH
-----------------------
Drop any 16-kHz-compatible .wav files into:
    data/clean_speech/

The pipeline will automatically discover and use them.
Suggested free, permissively-licensed sources (VERIFY before SIH demo):
  - CMU ARCTIC     http://www.festvox.org/cmu_arctic/
                   (16 kHz, ~60 utterances/speaker, ~50 MB/speaker)
  - LibriSpeech    https://openslr.org/12/
                   (16 kHz, many speakers, Apache 2.0 for metadata)
  - VCTK Corpus    https://datashare.ed.ac.uk/handle/10283/3443
                   (44.1 kHz -> resample to 16 kHz, CC BY 4.0)

Do NOT download speech automatically here -- verify license first.

IMPORTANT DISCLAIMER
---------------------
ESC-50 is a public environmental-sound dataset used to augment acoustic-
behaviour training data. It is NOT a defence dataset.
Proxy/Environmental labels are used for training generalisation only.
All defence-specific labels come from the project's own recordings.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
import warnings
from math import gcd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

# ---------------------------------------------------------------------------
# Project root (one level above audio_lab/)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

# Noise taxonomy from project source-of-truth
sys.path.insert(0, str(ROOT / "audio_lab"))
from dataset_sources import ESC50_CLASS_AUDIT  # dict: class -> (beh, conf, tier, note)

# ===========================================================================
# CONFIGURATION -- all values are tunable via CLI args
# ===========================================================================
TARGET_SR: int = 16_000            # Hz
CLIP_DURATION: float = 4.0         # seconds per training clip
SNR_VALUES: List[float] = [-5.0, 0.0, 5.0, 10.0, 15.0]  # dB
TRAIN_FRAC: float = 0.70
VAL_FRAC: float = 0.15
TEST_FRAC: float = 0.15
RANDOM_SEED: int = 42
MAX_PAIRS_PER_NOISE: int = 3       # speech utterances sampled per noise source per split
GAIN_JITTER_DB: float = 1.5        # +/- random gain perturbation on noise
TIME_SHIFT_MAX_MS: float = 30.0    # max circular time-shift on speech (ms)
CLIP_HEADROOM_DB: float = -1.0     # peak-normalise target (dBFS)

# ---------------------------------------------------------------------------
# Project-native baseline noise files (CORE tier)
# ---------------------------------------------------------------------------
BASELINE_NOISE_FILES: List[Dict] = [
    {
        "path": ROOT / "data" / "tank.wav",
        "label": "tank",
        "noise_type": "STATIONARY",
        "source_role": "CORE",
        "description": "Tank engine (project recording)",
    },
    {
        "path": ROOT / "data" / "heli.wav",
        "label": "helicopter",
        "noise_type": "NON_STATIONARY",
        "source_role": "CORE",
        "description": "Helicopter rotor (project recording)",
    },
    {
        "path": ROOT / "data" / "gun_fight.wav",
        "label": "gunfight",
        "noise_type": "IMPULSIVE",
        "source_role": "CORE",
        "description": "Gunfight audio (project recording)",
    },
]

# ESC-50 categories approved as PROXY noise (from dataset_sources.py taxonomy)
ESC50_ALLOWED_CATEGORIES: Dict = {
    k: v for k, v in ESC50_CLASS_AUDIT.items()
    if v[2] in ("CORE", "PROXY")   # v = (behavioral_class, confidence, tier, note)
}


# ===========================================================================
# UTILITY FUNCTIONS
# ===========================================================================

def _resample(audio: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    """Polyphase resample audio from src_sr to tgt_sr (no librosa needed)."""
    if src_sr == tgt_sr:
        return audio
    common = gcd(src_sr, tgt_sr)
    up = tgt_sr // common
    down = src_sr // common
    return resample_poly(audio, up, down).astype(np.float32)


def load_audio_mono_16k(path: Path,
                        target_sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    """Load any WAV as float32, mix to mono, resample to target_sr."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]
    audio = _resample(audio, sr, target_sr)
    return audio.astype(np.float32), target_sr


def rms_power(x: np.ndarray) -> float:
    """Mean squared power of signal (floored at 1e-10 to avoid log(0))."""
    return max(float(np.mean(x ** 2)), 1e-10)


def mix_at_snr(speech: np.ndarray,
               noise: np.ndarray,
               snr_db: float) -> Tuple[np.ndarray, float]:
    """
    Scale noise so that achieved SNR = snr_db, then return (mixture, scale).

    SNR_dB = 10 * log10(P_speech / P_noise_scaled)
    => scale = sqrt(P_speech / (P_noise * 10^(snr_db/10)))
    """
    p_s = rms_power(speech)
    p_n = rms_power(noise)
    snr_lin = 10.0 ** (snr_db / 10.0)
    scale = float(np.sqrt(p_s / (p_n * snr_lin)))
    mixture = speech + scale * noise
    return mixture, scale


def measure_snr(speech: np.ndarray, noise_scaled: np.ndarray) -> float:
    """Compute SNR_dB given separate speech and already-scaled noise arrays."""
    return 10.0 * float(np.log10(rms_power(speech) / rms_power(noise_scaled)))


def normalize_peak(audio: np.ndarray,
                   headroom_db: float = CLIP_HEADROOM_DB) -> np.ndarray:
    """Peak-normalise signal to headroom_db below full scale."""
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-6:
        target = 10.0 ** (headroom_db / 20.0)
        audio = audio * (target / peak)
    return audio


def random_segment(audio: np.ndarray,
                   length: int,
                   rng: random.Random) -> np.ndarray:
    """
    Extract a random contiguous segment of `length` samples.
    Tiles the audio if it is shorter than `length`.
    """
    if len(audio) == 0:
        return np.zeros(length, dtype=np.float32)
    if len(audio) >= length:
        start = rng.randint(0, len(audio) - length)
        return audio[start: start + length].copy()
    # Tile to cover required length
    repeats = (length // len(audio)) + 2
    tiled = np.tile(audio, repeats)
    start = rng.randint(0, len(tiled) - length)
    return tiled[start: start + length].copy()


def apply_time_shift(audio: np.ndarray,
                     max_ms: float,
                     sr: int,
                     rng: random.Random) -> np.ndarray:
    """Small circular time-shift applied to speech segment."""
    shift = int(rng.uniform(-max_ms, max_ms) * sr / 1000.0)
    return np.roll(audio, shift)


def clipping_percent(audio: np.ndarray,
                     threshold: float = 0.999) -> float:
    """Percentage of samples whose absolute value exceeds the threshold."""
    return 100.0 * float(np.mean(np.abs(audio) >= threshold))


# ===========================================================================
# NOISE CATALOGUE BUILDER
# ===========================================================================

def discover_noise_files(use_esc50: bool = True) -> List[Dict]:
    """
    Build the full noise-source catalogue.
    Priority: CORE (project files) -> custom defence_noise/ -> ESC-50 PROXY.
    """
    catalogue: List[Dict] = []

    # 1. Project baseline CORE files
    for entry in BASELINE_NOISE_FILES:
        if Path(entry["path"]).exists():
            catalogue.append(dict(entry))  # shallow copy
        else:
            warnings.warn(
                f"[NOISE] Baseline file missing: {entry['path']}")

    # 2. Custom data/defence_noise/ drop-in folder
    custom_dir = ROOT / "data" / "defence_noise"
    if custom_dir.exists():
        for wav in sorted(custom_dir.glob("*.wav")):
            catalogue.append({
                "path": wav,
                "label": wav.stem,
                "noise_type": "NON_STATIONARY",  # conservative default
                "source_role": "CORE",
                "description": f"Custom defence noise: {wav.name}",
            })

    # 3. ESC-50 PROXY noise
    if use_esc50:
        esc50_audio = ROOT / "data" / "ESC-50-master" / "audio"
        esc50_meta = ROOT / "data" / "ESC-50-master" / "meta" / "esc50.csv"
        if esc50_audio.exists() and esc50_meta.exists():
            meta_df = pd.read_csv(esc50_meta)
            for cat, v in ESC50_ALLOWED_CATEGORIES.items():
                behavioral, confidence, tier, note = v
                cat_files = (meta_df[meta_df["category"] == cat]["filename"]
                             .tolist()[:10])   # cap at 10/class
                for fn in cat_files:
                    fpath = esc50_audio / fn
                    if fpath.exists():
                        catalogue.append({
                            "path": fpath,
                            "label": cat,
                            "noise_type": behavioral,
                            "source_role": tier,
                            "description": (
                                f"ESC-50 [{cat}] -- {note[:60]}"),
                            "esc50_category": cat,
                            "confidence": confidence,
                        })
        elif not esc50_audio.exists():
            warnings.warn(
                "[NOISE] ESC-50 audio directory not found -- skipping.")

    return catalogue


# ===========================================================================
# SPEECH DISCOVERY
# ===========================================================================

def discover_speech_files() -> List[Path]:
    """Recursively find all WAV files under data/clean_speech/."""
    speech_dir = ROOT / "data" / "clean_speech"
    if not speech_dir.exists():
        return []
    return sorted(speech_dir.rglob("*.wav"))


# ===========================================================================
# RECORDING-LEVEL SPLIT (anti-leakage)
# ===========================================================================

def split_by_source(files: List[Path],
                    rng: random.Random,
                    train_f: float = TRAIN_FRAC,
                    val_f: float = VAL_FRAC,
                    test_f: float = TEST_FRAC) -> Dict[str, List[Path]]:
    """
    Split files at the RECORDING level -- never at the window level.
    The same speech file will NEVER appear in more than one split.
    """
    files = list(files)
    rng.shuffle(files)
    n = len(files)
    n_train = max(1, round(n * train_f))
    n_val = max(1, round(n * val_f))
    n_test = n - n_train - n_val
    # Ensure test is non-empty when we have enough files
    if n_test < 1 and n >= 3:
        n_test = 1
        n_val = n - n_train - n_test
    return {
        "train": files[:n_train],
        "val": files[n_train: n_train + n_val],
        "test": files[n_train + n_val:],
    }


# ===========================================================================
# PAIR GENERATOR (core loop)
# ===========================================================================

def generate_pairs(
    speech_split: Dict[str, List[Path]],
    noise_catalogue: List[Dict],
    output_dir: Path,
    snr_values: List[float],
    clip_dur: float,
    rng: random.Random,
    max_pairs_per_noise: int,
    enable_time_shift: bool,
    enable_gain_jitter: bool,
) -> List[Dict]:
    """
    Generate (noisy, clean) WAV pairs for every split.

    DESIGN: iterate NOISE-FIRST so that every noise source (including every
    noise_type category) gets proportional representation.  For each noise
    source, `max_pairs_per_noise` speech utterances are sampled per split.

    AMPLITUDE FIX: both clean and noisy WAVs are scaled by the SAME
    peak-normalisation factor before writing to disk.  This ensures that
    `noisy_read - clean_read == scaled_noise` exactly, which makes
    disk-level SNR verification accurate.

    Returns a list of metadata dicts (one per pair).
    """
    clip_len = int(clip_dur * TARGET_SR)
    records: List[Dict] = []
    pair_idx = 0

    # Pre-load and cache noise audio to avoid repeated disk reads
    noise_cache: Dict[str, np.ndarray] = {}

    def _cached_noise(entry: Dict) -> np.ndarray:
        key = str(entry["path"])
        if key not in noise_cache:
            try:
                a, _ = load_audio_mono_16k(Path(entry["path"]))
                noise_cache[key] = a
            except Exception as exc:
                warnings.warn(f"[NOISE LOAD] {entry['path']}: {exc}")
                noise_cache[key] = np.zeros(clip_len, dtype=np.float32)
        return noise_cache[key]

    # Iterate NOISE-FIRST for balanced class distribution
    for noise_entry in noise_catalogue:
        noise_full = _cached_noise(noise_entry)

        for split_name, speech_files in speech_split.items():
            if not speech_files:
                continue

            noisy_dir = output_dir / split_name / "noisy"
            clean_dir = output_dir / split_name / "clean"
            noisy_dir.mkdir(parents=True, exist_ok=True)
            clean_dir.mkdir(parents=True, exist_ok=True)

            # Sample up to max_pairs_per_noise speech utterances for this
            # noise source (sampling without replacement; reset if pool empty)
            n_sample = min(max_pairs_per_noise, len(speech_files))
            sampled_speech = rng.sample(speech_files, n_sample)

            for speech_path in sampled_speech:
                try:
                    speech_full, _ = load_audio_mono_16k(speech_path)
                except Exception as exc:
                    warnings.warn(f"[SPEECH LOAD] {speech_path}: {exc}")
                    continue

                for snr in snr_values:
                    # -- Extract speech segment ---------------------------
                    sp_seg = random_segment(speech_full, clip_len, rng)
                    if enable_time_shift:
                        sp_seg = apply_time_shift(
                            sp_seg, TIME_SHIFT_MAX_MS, TARGET_SR, rng)
                    sp_seg = normalize_peak(sp_seg)   # clean reference

                    # -- Extract noise segment ----------------------------
                    ns_seg = random_segment(noise_full, clip_len, rng)
                    if enable_gain_jitter:
                        jitter = rng.uniform(-GAIN_JITTER_DB, GAIN_JITTER_DB)
                        ns_seg = ns_seg * (10.0 ** (jitter / 20.0))

                    # -- Mix at requested SNR -----------------------------
                    noisy_seg, ns_scale = mix_at_snr(sp_seg, ns_seg, snr)

                    # AMPLITUDE FIX: compute one normalisation factor and
                    # apply it to BOTH signals so disk-read SNR is exact.
                    #   noisy_saved = A * (speech + scale*noise)
                    #   clean_saved = A * speech
                    #   noisy_saved - clean_saved = A * scale * noise  ✓
                    peak = float(np.max(np.abs(noisy_seg)))
                    if peak > 1e-6:
                        target_peak = 10.0 ** (CLIP_HEADROOM_DB / 20.0)
                        A = target_peak / peak
                    else:
                        A = 1.0
                    noisy_saved = (noisy_seg * A).astype(np.float32)
                    clean_saved = (sp_seg   * A).astype(np.float32)

                    # -- Measure SNR before disk quantisation -------------
                    meas_snr = measure_snr(clean_saved, ns_scale * A * ns_seg)
                    clip_pct = clipping_percent(noisy_saved)

                    # -- Write WAVs ---------------------------------------
                    pair_id = f"{split_name}_{pair_idx:06d}"
                    noisy_path = noisy_dir / f"{pair_id}.wav"
                    clean_path = clean_dir / f"{pair_id}.wav"
                    sf.write(str(noisy_path), noisy_saved,
                             TARGET_SR, subtype="PCM_16")
                    sf.write(str(clean_path), clean_saved,
                             TARGET_SR, subtype="PCM_16")

                    # -- Record metadata ----------------------------------
                    records.append({
                        "pair_id": pair_id,
                        "split": split_name,
                        "clean_file": str(
                            clean_path.relative_to(ROOT)),
                        "noisy_file": str(
                            noisy_path.relative_to(ROOT)),
                        "speech_source": str(
                            speech_path.relative_to(ROOT)),
                        "noise_file": str(
                            Path(noise_entry["path"]).relative_to(ROOT)),
                        "noise_label": noise_entry["label"],
                        "noise_type": noise_entry["noise_type"],
                        "source_role": noise_entry["source_role"],
                        "snr_db_requested": snr,
                        "snr_db_measured": round(meas_snr, 2),
                        "snr_error_db": round(abs(snr - meas_snr), 2),
                        "sample_rate": TARGET_SR,
                        "duration_sec": round(clip_dur, 2),
                        "clipping_pct": round(clip_pct, 4),
                        "gain_jitter_applied": enable_gain_jitter,
                        "time_shift_applied": enable_time_shift,
                    })

                    pair_idx += 1

    return records


# ===========================================================================
# SNR VERIFICATION (re-read from disk)
# ===========================================================================

def verify_snr_sample(records: List[Dict],
                      n_samples: int = 10) -> List[Dict]:
    """
    Independently verify SNR of a random sample of generated pairs.

    Method: noise_estimate = noisy - clean  (valid because mix is additive)
    SNR = 10 * log10( P_clean / P_noise_estimate )

    This confirms the signal math is correct end-to-end.
    """
    results: List[Dict] = []
    sample = random.sample(records, min(n_samples, len(records)))
    for rec in sample:
        try:
            clean, _ = sf.read(
                str(ROOT / rec["clean_file"]), dtype="float32")
            noisy, _ = sf.read(
                str(ROOT / rec["noisy_file"]), dtype="float32")
            noise_est = noisy - clean
            snr_disk = 10.0 * float(
                np.log10(rms_power(clean) / rms_power(noise_est)))
            results.append({
                "pair_id": rec["pair_id"],
                "requested_snr": rec["snr_db_requested"],
                "disk_measured_snr": round(snr_disk, 2),
                "error_db": round(
                    abs(rec["snr_db_requested"] - snr_disk), 2),
                "noise_type": rec["noise_type"],
                "source_role": rec["source_role"],
            })
        except Exception as exc:
            results.append({
                "pair_id": rec["pair_id"],
                "requested_snr": rec["snr_db_requested"],
                "disk_measured_snr": "ERROR",
                "error_db": "ERROR",
                "noise_type": rec.get("noise_type"),
                "source_role": rec.get("source_role"),
                "exc": str(exc),
            })
    return results


# ===========================================================================
# REPORT GENERATOR
# ===========================================================================

def _hr(n: int = 70, ch: str = "-") -> str:
    return ch * n


def generate_report(
    records: List[Dict],
    snr_verify: List[Dict],
    output_dir: Path,
    speech_files: List[Path],
    noise_catalogue: List[Dict],
    missing_speech: bool,
    elapsed: float,
) -> str:
    lines: List[str] = []

    lines += [
        _hr(70, "="),
        "ECHO SHIELD -- Step 23: Speech Enhancement Dataset Report",
        "PS ID: SIH26052 | DRDO | SIH 2026",
        _hr(70, "="),
        f"Generated : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Elapsed   : {elapsed:.1f} s",
        f"Target SR : {TARGET_SR} Hz",
        f"Clip dur  : {CLIP_DURATION} s",
        f"SNR range : {SNR_VALUES} dB",
        f"Seed      : {RANDOM_SEED}",
        "",
    ]

    # -- Clean speech status -----------------------------------------------
    lines += [_hr(), "CLEAN SPEECH STATUS", _hr()]
    if missing_speech:
        lines += [
            "  WARNING: data/clean_speech/ is EMPTY or does not exist.",
            "  WARNING: No noisy/clean pairs were generated.",
            "",
            "  ACTION REQUIRED: Place clean speech WAV files in:",
            f"      {ROOT / 'data' / 'clean_speech'}",
            "",
            "  Recommended free, permissively-licensed sources:",
            "    CMU ARCTIC    http://www.festvox.org/cmu_arctic/",
            "                  16 kHz, ~60 utterances/speaker, ~50 MB/speaker",
            "    LibriSpeech   https://openslr.org/12/",
            "                  16 kHz, many speakers, Apache 2.0 for metadata",
            "    VCTK Corpus   https://datashare.ed.ac.uk/handle/10283/3443",
            "                  44.1 kHz -> resample to 16 kHz, CC BY 4.0",
            "",
            "  VERIFY LICENSE before using in an SIH demonstration.",
            "  Once added, re-run:",
            "      python audio_lab/step23_build_speech_dataset.py",
        ]
    else:
        total_sp_dur = sum(
            float(sf.info(str(p)).duration)
            for p in speech_files if p.exists()
        )
        lines += [
            f"  Clean speech files : {len(speech_files)}",
            f"  Total duration     : {total_sp_dur:.1f} s "
            f"({total_sp_dur / 60:.1f} min)",
        ]
    lines.append("")

    # -- Noise catalogue ---------------------------------------------------
    lines += [_hr(), "NOISE CATALOGUE", _hr()]
    tier_counts: Dict[str, int] = {}
    for e in noise_catalogue:
        t = e["source_role"]
        tier_counts[t] = tier_counts.get(t, 0) + 1
    lines.append(f"  Total noise sources : {len(noise_catalogue)}")
    for tier, cnt in sorted(tier_counts.items()):
        lines.append(f"    {tier:20s}: {cnt} files")
    lines += [
        "",
        "  Role definitions:",
        "    CORE         -- project's own defence recordings",
        "    PROXY        -- public environmental sounds with similar acoustics",
        "    ENVIRONMENTAL-- general environmental sounds (experimental)",
        "",
        "  DISCLAIMER: ESC-50 is a public environmental-sound dataset.",
        "  It is NOT a defence dataset. Proxy/Environmental labels are used",
        "  for training generalisation ONLY.",
        "",
    ]

    # -- Dataset summary ---------------------------------------------------
    if len(records) == 0:
        lines += [
            _hr(),
            "DATASET SUMMARY: No pairs generated (missing clean speech).",
            _hr(),
        ]
    else:
        df = pd.DataFrame(records)
        total_audio_s = len(df) * CLIP_DURATION
        lines += [_hr(), "DATASET SUMMARY", _hr()]
        lines += [
            f"  Total pairs          : {len(df)}",
            f"  Unique clean files   : {df['clean_file'].nunique()}",
            f"  Unique noisy files   : {df['noisy_file'].nunique()}",
            f"  Total audio (noisy)  : {total_audio_s:.0f} s "
            f"({total_audio_s / 3600:.3f} h)",
            f"  Sample rate          : {TARGET_SR} Hz",
            f"  Clip duration        : {CLIP_DURATION} s",
            "",
            "  Split distribution:",
        ]
        for sp, grp in df.groupby("split"):
            lines.append(
                f"    {sp:10s}: {len(grp):5d} pairs "
                f"({len(grp) * CLIP_DURATION:.0f} s)"
            )
        lines += ["", "  SNR distribution (requested):"]
        for snr, grp in df.groupby("snr_db_requested"):
            lines.append(f"    {snr:+5.1f} dB : {len(grp):4d} pairs")
        lines += ["", "  Noise type distribution:"]
        for nt, grp in df.groupby("noise_type"):
            lines.append(f"    {nt:20s}: {len(grp):5d} pairs")
        mean_clip = float(df["clipping_pct"].mean())
        lines += ["", f"  Mean clipping %      : {mean_clip:.4f}%", ""]

        # SNR verification table
        lines += [_hr(), "SNR VERIFICATION (re-read from disk)", _hr()]
        lines.append("  pair_id                   req SNR    meas SNR    err")
        lines.append("  " + _hr(56))
        for v in snr_verify:
            meas = v.get("disk_measured_snr", "ERROR")
            err = v.get("error_db", "ERROR")
            if isinstance(meas, float):
                flag = "  OK" if float(err) < 1.5 else "  ** HIGH ERROR **"
                lines.append(
                    f"  {v['pair_id']:24s}  {v['requested_snr']:+5.1f} dB"
                    f"   {meas:+6.2f} dB   {err:5.2f} dB{flag}"
                )
            else:
                lines.append(
                    f"  {v['pair_id']:24s}  ERROR: {v.get('exc', '?')}")
        lines.append("")

        # Leakage assessment
        lines += [_hr(), "DATA LEAKAGE ASSESSMENT", _hr()]
        sp_train = set(df[df["split"] == "train"]["speech_source"])
        sp_test = set(df[df["split"] == "test"]["speech_source"])
        overlap = sp_train & sp_test
        if overlap:
            lines += [
                f"  !! LEAKAGE DETECTED: {len(overlap)} speech file(s) "
                "appear in both train and test.",
            ]
        else:
            lines.append(
                "  OK  No leakage: train and test speech files are disjoint.")
        n_sp = len(speech_files)
        lines.append(f"  Available clean speech files: {n_sp}")
        if n_sp < 10:
            lines += [
                "  WARNING: Dataset is very small. Statistical validity is LIMITED.",
                "           Add more clean speech before training any model.",
            ]
        lines.append("")

    # -- Files created -----------------------------------------------------
    lines += [
        _hr(),
        "FILES CREATED",
        _hr(),
        "  Dataset root : data/speech_dataset/",
        "  Metadata CSV : data/speech_dataset/metadata.csv",
        "  This report  : outputs/step23_dataset_report.txt",
        "",
    ]

    # -- Next steps --------------------------------------------------------
    lines += [
        _hr(),
        "NEXT STEPS (in order)",
        _hr(),
        "  1. Add clean speech WAVs to   data/clean_speech/",
        "  2. Re-run this script:         python audio_lab/step23_build_speech_dataset.py",
        "  3. Review outputs/step23_dataset_report.txt",
        "  4. ONLY after SNR verification passes -> proceed to Step 24 (model training)",
        "  5. Minimum ~50 speech files recommended before training",
        "",
        "  ENGINEERING PRINCIPLE: Correct dataset first. Model second.",
        _hr(70, "="),
    ]

    return "\n".join(lines)


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 23 -- Build clean-speech + defence-noise paired dataset"
    )
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "data" / "speech_dataset",
                   help="Root directory for generated dataset "
                        "(default: data/speech_dataset/)")
    p.add_argument("--seed", type=int, default=RANDOM_SEED,
                   help="Random seed (default: 42)")
    p.add_argument("--clip-duration", type=float, default=CLIP_DURATION,
                   help="Clip length in seconds (default: 4.0)")
    p.add_argument("--snr", type=float, nargs="+", default=SNR_VALUES,
                   help="SNR values in dB (default: -5 0 5 10 15)")
    p.add_argument("--max-pairs-per-noise", type=int,
                   default=MAX_PAIRS_PER_NOISE,
                   help="Max pairs per noise file (default: 20)")
    p.add_argument("--no-esc50", action="store_true",
                   help="Skip ESC-50 proxy noise; use only CORE files")
    p.add_argument("--no-time-shift", action="store_true",
                   help="Disable random time-shift augmentation")
    p.add_argument("--no-gain-jitter", action="store_true",
                   help="Disable random gain-jitter augmentation")
    p.add_argument("--snr-verify-n", type=int, default=10,
                   help="Number of pairs to re-read for SNR verification "
                        "(default: 10)")
    return p.parse_args()


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    args = parse_args()
    t0 = time.time()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    sep = "=" * 70
    print(sep)
    print("Echo Shield -- Step 23: Build Speech Enhancement Dataset")
    print("PS ID: SIH26052 | DRDO | SIH 2026")
    print(sep)

    # ------------------------------------------------------------------
    # [1/7] AUDIT
    # ------------------------------------------------------------------
    print("\n[1/7] Auditing existing data ...")
    noise_catalogue = discover_noise_files(use_esc50=not args.no_esc50)
    speech_files = discover_speech_files()

    core_n = sum(1 for e in noise_catalogue if e["source_role"] == "CORE")
    proxy_n = sum(1 for e in noise_catalogue if e["source_role"] == "PROXY")
    env_n = sum(1 for e in noise_catalogue
                if e["source_role"] == "ENVIRONMENTAL")

    print(f"  Noise sources   : {len(noise_catalogue)} files")
    print(f"    CORE          : {core_n}")
    print(f"    PROXY         : {proxy_n}")
    print(f"    ENVIRONMENTAL : {env_n}")
    print(f"  Clean speech    : {len(speech_files)} WAV files found")

    missing_speech = len(speech_files) == 0

    if missing_speech:
        print()
        print("  +---------------------------------------------------+")
        print("  |  NO CLEAN SPEECH FILES FOUND                      |")
        print("  |                                                   |")
        print("  |  Place WAV files in:  data/clean_speech/          |")
        print("  |                                                   |")
        print("  |  Sources (verify license first):                  |")
        print("  |    CMU ARCTIC  http://www.festvox.org/cmu_arctic/  |")
        print("  |    LibriSpeech https://openslr.org/12/             |")
        print("  |    VCTK        https://datashare.ed.ac.uk/...      |")
        print("  +---------------------------------------------------+")
        print()
        print("  Pipeline is COMPLETE. Re-run after adding speech files.")

    # ------------------------------------------------------------------
    # [2/7] SPLIT  &  [3/7] GENERATE
    # ------------------------------------------------------------------
    records: List[Dict] = []
    speech_split: Dict[str, List[Path]] = {
        "train": [], "val": [], "test": []}

    if not missing_speech:
        print("\n[2/7] Splitting speech files at recording level ...")
        speech_split = split_by_source(speech_files, rng)
        for sp, fl in speech_split.items():
            print(f"  {sp:10s}: {len(fl)} speech file(s)")
        if sum(len(v) for v in speech_split.values()) < 3:
            print("  WARNING: very few files -- split may be degenerate.")

        print("\n[3/7] Generating noisy/clean pairs ...")
        records = generate_pairs(
            speech_split=speech_split,
            noise_catalogue=noise_catalogue,
            output_dir=args.output_dir,
            snr_values=args.snr,
            clip_dur=args.clip_duration,
            rng=rng,
            max_pairs_per_noise=args.max_pairs_per_noise,
            enable_time_shift=not args.no_time_shift,
            enable_gain_jitter=not args.no_gain_jitter,
        )
        print(f"  Generated {len(records)} pairs.")
    else:
        print("\n[2-3/7] Skipped (no speech files).")

    # ------------------------------------------------------------------
    # [4/7] SAVE METADATA CSV
    # ------------------------------------------------------------------
    print("\n[4/7] Saving metadata.csv ...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.output_dir / "metadata.csv"

    if records:
        pd.DataFrame(records).to_csv(meta_path, index=False)
        print(f"  Saved: {meta_path.relative_to(ROOT)}")
    else:
        # Write empty CSV with correct headers so path always exists
        headers = [
            "pair_id", "split", "clean_file", "noisy_file",
            "speech_source", "noise_file", "noise_label",
            "noise_type", "source_role",
            "snr_db_requested", "snr_db_measured", "snr_error_db",
            "sample_rate", "duration_sec", "clipping_pct",
            "gain_jitter_applied", "time_shift_applied",
        ]
        with open(meta_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=headers).writeheader()
        print(f"  Saved empty template: {meta_path.relative_to(ROOT)}")

    # ------------------------------------------------------------------
    # [5/7] SNR VERIFICATION
    # ------------------------------------------------------------------
    print("\n[5/7] SNR verification ...")
    snr_verify: List[Dict] = []
    if records:
        snr_verify = verify_snr_sample(records, n_samples=args.snr_verify_n)
        num_errors = [
            v["error_db"] for v in snr_verify
            if isinstance(v.get("error_db"), float)
        ]
        if num_errors:
            mean_err = sum(num_errors) / len(num_errors)
            max_err = max(num_errors)
            print(f"  Mean SNR error : {mean_err:.3f} dB  "
                  f"(max: {max_err:.3f} dB)")
            if max_err > 2.0:
                print("  WARNING: some SNR errors > 2 dB.")
            else:
                print("  OK: SNR math verified within acceptable tolerance.")
        for v in snr_verify[:5]:
            meas = v.get("disk_measured_snr", "ERR")
            meas_s = f"{meas:+.2f}" if isinstance(meas, float) else str(meas)
            print(f"    {v['pair_id']:24s}  "
                  f"req={v['requested_snr']:+5.1f} dB  meas={meas_s} dB")
    else:
        print("  Skipped (no pairs).")

    # ------------------------------------------------------------------
    # [6/7] GENERATE REPORT
    # ------------------------------------------------------------------
    print("\n[6/7] Writing report ...")
    elapsed = time.time() - t0
    report_txt = generate_report(
        records=records,
        snr_verify=snr_verify,
        output_dir=args.output_dir,
        speech_files=speech_files,
        noise_catalogue=noise_catalogue,
        missing_speech=missing_speech,
        elapsed=elapsed,
    )
    report_path = ROOT / "outputs" / "step23_dataset_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_txt, encoding="utf-8")
    print(f"  Saved: {report_path.relative_to(ROOT)}")

    # ------------------------------------------------------------------
    # [7/7] FINAL SUMMARY
    # ------------------------------------------------------------------
    print("\n[7/7] Final summary")
    print("-" * 70)
    print(report_txt)
    print("-" * 70)
    print(f"\nStep 23 complete in {elapsed:.1f} s")

    if missing_speech:
        print()
        print("ACTION REQUIRED: Add clean speech to data/clean_speech/ then re-run.")
        print("Do NOT proceed to Step 24 until dataset is verified.")
        sys.exit(0)   # Clean exit -- not a failure, just no data yet


if __name__ == "__main__":
    main()
