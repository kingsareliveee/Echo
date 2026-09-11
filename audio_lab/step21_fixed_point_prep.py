# ==============================================================================
# Project: AI/ML-enabled Adaptive Noise Cancellation (ANC) for Defence Vehicles
# PS ID: SIH26052 | Team: Echo Shield | Theme: Smart Vehicles
# 
# Step 21: Fixed-Point Arithmetic (Q1.15) Preparation for FPGA DSP
# ==============================================================================

import os
import sys
import csv
import numpy as np
from scipy.io import wavfile

# ==============================================================================
# BEGINNER-FRIENDLY CONCEPT: FLOATING-POINT vs. FIXED-POINT IN FPGAs
#
# 1. WHY FPGAs USE FIXED-POINT ARITHMETIC:
#    - Floating-point units (FPUs) in hardware consume massive silicon area, high power,
#      and introduce multi-cycle latency.
#    - Fixed-point representation (e.g., Q1.15) uses native, ultra-fast 16-bit hardware
#      DSP slices (e.g. DSP48E1 in AMD/Xilinx FPGAs) running in a single clock cycle.
#
# 2. WHAT IS Q1.15 FORMAT?
#    - 1 Sign bit + 15 Fractional bits = 16-bit signed integer.
#    - Range: -1.0 to +0.999969482421875 (Step resolution = 1 / 32768 \u2248 0.0000305).
#    - Conversion:
#        Q1.15 = round(float_value * 32768.0)
#        float = Q1.15 / 32768.0
# ==============================================================================

# --- 1. Define File Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(OUTPUTS_DIR, exist_ok=True)

CSV_OUTPUT_PATH = os.path.join(OUTPUTS_DIR, "step21_q15_analysis.csv")

print("=" * 80)
print("  ECHO SHIELD - Step 21: Fixed-Point (Q1.15) DSP Hardware Preparation")
print("=" * 80)

# --- 2. Q1.15 Conversion Functions ---
def float_to_q15(val):
    """
    Converts a floating-point number in [-1.0, 1.0) to a signed 16-bit Q1.15 integer.
    """
    scaled = np.round(val * 32768.0)
    clamped = np.clip(scaled, -32768, 32767)
    return int(clamped)

def q15_to_float(q_val):
    """
    Converts a signed 16-bit Q1.15 integer back into floating-point.
    """
    return float(q_val) / 32768.0

# --- 3. Representative Test Values ---
# 1. Step size mu
mu_float = 0.005

# 2. Sample filter weights (simulating converged 32-tap LMS weights)
np.random.seed(42)
test_weights = np.random.uniform(-0.35, 0.35, size=32)

# 3. Audio sample points from tank.wav
tank_path = os.path.join(DATA_DIR, "tank.wav")
if os.path.isfile(tank_path):
    _, raw_audio = wavfile.read(tank_path)
    ch1 = raw_audio[:, 0] if raw_audio.ndim > 1 else raw_audio
    raw_max = np.max(np.abs(ch1))
    norm_audio = ch1.astype(np.float64) / (32768.0 if raw_max > 0 else 1.0)
    sample_audio_points = norm_audio[1000:1010]  # 10 test audio points
else:
    sample_audio_points = np.linspace(-0.8, 0.8, 10)

# --- 4. Quantization Error Analysis ---
analysis_rows = []

# Analyze step size
q_mu = float_to_q15(mu_float)
recon_mu = q15_to_float(q_mu)
err_mu = abs(mu_float - recon_mu)
analysis_rows.append({
    "parameter": "Step Size (mu)",
    "index": "N/A",
    "float_orig": mu_float,
    "q15_int": q_mu,
    "float_recon": recon_mu,
    "abs_error": err_mu
})

# Analyze filter weights
for idx, w in enumerate(test_weights):
    qw = float_to_q15(w)
    rw = q15_to_float(qw)
    analysis_rows.append({
        "parameter": "Filter Weight w[k]",
        "index": str(idx),
        "float_orig": w,
        "q15_int": qw,
        "float_recon": rw,
        "abs_error": abs(w - rw)
    })

# Analyze audio samples
for idx, s in enumerate(sample_audio_points):
    qs = float_to_q15(s)
    rs = q15_to_float(qs)
    analysis_rows.append({
        "parameter": "Audio Sample x[n]",
        "index": str(idx),
        "float_orig": s,
        "q15_int": qs,
        "float_recon": rs,
        "abs_error": abs(s - rs)
    })

# --- 5. Summary Statistics ---
all_errors = [r["abs_error"] for r in analysis_rows]
max_error = float(np.max(all_errors))
mean_error = float(np.mean(all_errors))
theoretical_lsb = 1.0 / 32768.0

# --- 6. Print Results ---
print(f"{'Parameter':<22} | {'Index':<5} | {'Original Float':<15} | {'Q1.15 Int':<10} | {'Recon Float':<15} | {'Abs Error':<12}")
print("-" * 90)
# Print first 10 representative rows
for r in analysis_rows[:10]:
    print(f"{r['parameter']:<22} | {r['index']:<5} | {r['float_orig']:>14.6f} | {r['q15_int']:>9} | {r['float_recon']:>14.6f} | {r['abs_error']:>11.8f}")
print("... (additional rows saved in CSV)")
print("-" * 90)

print(f"\n[QUANTIZATION STATISTICS]")
print(f"  Theoretical Q1.15 Resolution (1 LSB) : {theoretical_lsb:.8f}")
print(f"  Max Absolute Quantization Error      : {max_error:.8f}")
print(f"  Mean Absolute Quantization Error     : {mean_error:.8f}")
print("-" * 90)
print("\n[WHY FIXED-POINT (Q1.15) IS CRITICAL FOR FPGA ANC]")
print("1. Ultra-Low Latency   : Hardware multipliers complete in 1 FPGA clock cycle (< 10 ns).")
print("2. Resource Efficiency : Uses standard 16-bit DSP48 slices without bulky FPUs.")
print("3. Determinism         : Fixed bit-width ensures exact bit-level reproducibility.")
print("-" * 90)
print("[IMPORTANT DISCLAIMER]")
print("This is a fixed-point numerical quantization preparation script.")
print("It does NOT generate Verilog/VHDL RTL and does NOT report FPGA synthesis timing.")
print("=" * 80)

# --- 7. Save CSV Report ---
with open(CSV_OUTPUT_PATH, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["parameter", "index", "float_orig", "q15_int", "float_recon", "abs_error"])
    writer.writeheader()
    writer.writerows(analysis_rows)

print(f"\n[SUCCESS] Q1.15 quantization analysis saved to: {CSV_OUTPUT_PATH}")
print("=" * 80)
