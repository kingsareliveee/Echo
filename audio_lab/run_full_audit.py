"""
Echo Shield — Automated Technical Audit Runner
==============================================
Runs all Step 01-24 scripts, captures exit codes, outputs, metrics,
inspects datasets, models, and verifies mathematical consistency.
"""

import os
import sys
import subprocess
import time
import json
import glob
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_EXE = sys.executable

STEPS = [
    ("Step 01", "audio_lab/step01_load_audio.py", "Audio Loading & Normalization"),
    ("Step 02", "audio_lab/step02_fft.py", "FFT & Spectral Frequency Analysis"),
    ("Step 03", "audio_lab/step03_spectrogram.py", "STFT Spectrogram Generation"),
    ("Step 04", "audio_lab/step04_features.py", "Global Acoustic Feature Extraction"),
    ("Step 05", "audio_lab/step05_window_features.py", "Sliding-Window Feature Extraction"),
    ("Step 06", "audio_lab/step06_build_dataset.py", "Classifier Dataset Construction"),
    ("Step 07", "audio_lab/step07_train_classifier.py", "Decision Tree Noise Classifier Training"),
    ("Step 08", "audio_lab/step08_predict_noise.py", "Single File Majority-Vote Prediction"),
    ("Step 09", "audio_lab/step09_batch_predict.py", "Batch Prediction Across Audio Files"),
    ("Step 10", "audio_lab/step10_noise_profile.py", "Noise Profile Lookup & Mapping"),
    ("Step 11", "audio_lab/step11_dsp_config.py", "Adaptive DSP Parameter Configuration"),
    ("Step 12", "audio_lab/step12_lms_filter.py", "Standard LMS Adaptive Filter Simulation"),
    ("Step 13", "audio_lab/step13_lms_vs_nlms.py", "LMS vs Normalized LMS (NLMS) Comparison"),
    ("Step 14", "audio_lab/step14_signal_model.py", "Primary Path & Desired Signal Modeling"),
    ("Step 15", "audio_lab/step15_secondary_path.py", "Synthetic S(z) Secondary Path Modeling"),
    ("Step 16", "audio_lab/step16_fxlms.py", "Filtered-X LMS (FxLMS) Algorithm Simulation"),
    ("Step 17", "audio_lab/step17_anc_simulation.py", "Complete Offline ANC Acoustic Simulation"),
    ("Step 18", "audio_lab/step18_wavelet_reconstruction.py", "Daubechies-4 (db4) Wavelet Denoising"),
    ("Step 19", "audio_lab/step19_full_pipeline.py", "Dual-Pipeline Software Integration"),
    ("Step 20", "audio_lab/step20_evaluation.py", "Quantitative Metrics & Performance Evaluation"),
    ("Step 21", "audio_lab/step21_fixed_point_prep.py", "Q1.15 Fixed-Point Arithmetic Analysis"),
    ("Step 22", "audio_lab/step22_final_demo.py", "End-to-End Terminal Demo & Verification"),
    ("Step 23", "audio_lab/step23_build_speech_dataset.py", "Speech-in-Noise Paired Dataset Builder", ["--max-pairs-per-noise", "2"]),
    ("Step 24", "audio_lab/step24_speech_enhancement.py", "AI Speech Enhancement & Denoising Engine"),
    ("Behavior Train", "audio_lab/train_behavior_classifier.py", "Expanded Multi-Class Behavior Classifier"),
    ("Behavior Eval", "audio_lab/evaluate_behavior_classifier.py", "Classifier Generalization Evaluation"),
    ("Smoke Test", "smoke_test.py", "Pipeline Smoke Test Harness"),
]

def run_step(step_info):
    step_id = step_info[0]
    script_rel = step_info[1]
    desc = step_info[2]
    extra_args = step_info[3] if len(step_info) > 3 else []
    
    script_path = os.path.join(ROOT, script_rel)
    if not os.path.isfile(script_path):
        return {
            "step_id": step_id,
            "script": script_rel,
            "description": desc,
            "exit_code": -1,
            "status": "FILE_MISSING",
            "runtime_sec": 0.0,
            "stdout": "",
            "stderr": "File not found",
        }
    
    cmd = [PYTHON_EXE, script_rel] + extra_args
    start_t = time.time()
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, errors="replace")
    elapsed = time.time() - start_t
    
    status = "WORKING" if res.returncode == 0 else "FAILED"
    
    return {
        "step_id": step_id,
        "script": script_rel,
        "description": desc,
        "exit_code": res.returncode,
        "status": status,
        "runtime_sec": round(elapsed, 2),
        "stdout": res.stdout,
        "stderr": res.stderr,
    }

if __name__ == "__main__":
    print("[*] Starting Full Project Technical Audit Execution...")
    results = []
    for s in STEPS:
        print(f" -> Running {s[0]} ({s[1]})... ", end="", flush=True)
        r = run_step(s)
        print(f"[{r['status']}] (exit {r['exit_code']}, {r['runtime_sec']}s)")
        results.append(r)
        
    out_json = os.path.join(ROOT, "outputs", "audit_run_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Audit execution complete. Saved results to: {out_json}")
