# ==============================================================================
# Project: AI/ML-enabled Adaptive Noise Cancellation (ANC) for Defence Vehicles
# PS ID: SIH26052 | Team: Echo Shield | Theme: Smart Vehicles
# 
# Step 20: Objective Performance Evaluation & Metrics Summary
# ==============================================================================

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define File Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(OUTPUTS_DIR, exist_ok=True)

CSV_OUTPUT_PATH = os.path.join(OUTPUTS_DIR, "step20_evaluation.csv")
PLOT_OUTPUT_PATH = os.path.join(OUTPUTS_DIR, "step20_evaluation.png")

print("=" * 80)
print("  ECHO SHIELD - Step 20: Performance Evaluation & Validation Table")
print("=" * 80)

# --- 2. Honest Structured Evaluation Metrics ---
METRICS_DATA = [
    {
        "module": "ML Noise Classifier (Decision Tree)",
        "metric": "Classification Accuracy (Held-out Test Split)",
        "value": "98.97%",
        "numeric_val": 98.97,
        "type": "MODEL TEST",
        "notes": "Evaluated on 290 stratified held-out 100ms test windows"
    },
    {
        "module": "ML Noise Classifier (Decision Tree)",
        "metric": "Macro F1-Score",
        "value": "0.97",
        "numeric_val": 97.0,
        "type": "MODEL TEST",
        "notes": "Balanced across tank (0.93), heli (1.00), gun_fight (0.99)"
    },
    {
        "module": "DSP Adaptive Filter (Standard LMS)",
        "metric": "RMS Noise Reduction (tank.wav)",
        "value": "20.65 dB",
        "numeric_val": 20.65,
        "type": "MEASURED SOFTWARE EXPERIMENT",
        "notes": "32 taps, mu=0.005, sample-by-sample loop"
    },
    {
        "module": "DSP Adaptive Filter (Normalized LMS)",
        "metric": "RMS Noise Reduction (tank.wav)",
        "value": "31.58 dB",
        "numeric_val": 31.58,
        "type": "MEASURED SOFTWARE EXPERIMENT",
        "notes": "32 taps, mu_norm=0.5, epsilon=1e-8"
    },
    {
        "module": "Active Noise Cancellation (FxLMS)",
        "metric": "Simulated Acoustic Error Reduction (tank.wav)",
        "value": "11.89 dB",
        "numeric_val": 11.89,
        "type": "SIMULATION",
        "notes": "Offline simulation using 12-tap synthetic S(z) secondary path"
    },
    {
        "module": "Speech Path (Wavelet Denoising)",
        "metric": "Signal RMS Attenuation (db4, Level 5)",
        "value": "0.02 dB",
        "numeric_val": 0.02,
        "type": "MEASURED SOFTWARE EXPERIMENT",
        "notes": "Soft-thresholding on detail sub-bands (DWT/IDWT)"
    },
    {
        "module": "Embedded Hardware FPGA Processing",
        "metric": "Physical Latency & Power Consumption",
        "value": "N/A",
        "numeric_val": 0.0,
        "type": "NOT MEASURED",
        "notes": "Physical FPGA synthesis & RTL execution planned for hardware phase"
    },
    {
        "module": "Physical Vehicle Acoustic ANC",
        "metric": "In-Cabin Sound Pressure Level (SPL) Attenuation",
        "value": "N/A",
        "numeric_val": 0.0,
        "type": "NOT MEASURED",
        "notes": "Requires physical error microphones, speakers, and vehicle field testing"
    }
]

# --- 3. Print Console Summary Table ---
print(f"{'Module':<36} | {'Metric':<38} | {'Value':<12} | {'Type':<28}")
print("-" * 122)
for item in METRICS_DATA:
    print(f"{item['module']:<36} | {item['metric']:<38} | {item['value']:<12} | {item['type']:<28}")
print("-" * 122)

print("\n[IMPORTANT SCIENTIFIC INTEGRITY STATEMENT]")
print("- We do NOT combine unrelated metrics into a single arbitrary 'system score'.")
print("- Hardware metrics (latency, power, FPGA utilization) and physical acoustics (SPL reduction)")
print("  are explicitly categorized as NOT MEASURED until verified on physical testbenches.")
print("=" * 80)

# --- 4. Save to CSV ---
with open(CSV_OUTPUT_PATH, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["module", "metric", "value", "type", "notes"])
    writer.writeheader()
    for row in METRICS_DATA:
        writer.writerow({
            "module": row["module"],
            "metric": row["metric"],
            "value": row["value"],
            "type": row["type"],
            "notes": row["notes"]
        })

print(f"\n[SUCCESS] Performance metrics saved to: {CSV_OUTPUT_PATH}")

# --- 5. Generate and Save Performance Visualizations ---
measurable_items = [item for item in METRICS_DATA if item["type"] != "NOT MEASURED"]
labels = [f"{item['module'][:15]}...\n({item['metric'][:20]}...)" for item in measurable_items]
values = [item["numeric_val"] for item in measurable_items]
colors = ["#3498db" if item["type"] == "MODEL TEST" else ("#2ecc71" if item["type"] == "MEASURED SOFTWARE EXPERIMENT" else "#e67e22") for item in measurable_items]

plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(measurable_items)), values, color=colors, width=0.55, edgecolor="#2c3e50")

for bar, item in zip(bars, measurable_items):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{item['value']}\n[{item['type']}]", ha="center", va="bottom", fontsize=8, fontweight="bold")

plt.title("Echo Shield: Software Prototype Performance Metrics by Category", fontsize=12, fontweight="bold")
plt.ylabel("Value (dB / %)", fontsize=10)
plt.xticks(range(len(measurable_items)), labels, fontsize=8, rotation=15)
plt.ylim(0, 115)
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(PLOT_OUTPUT_PATH, dpi=300)
plt.close()

print(f"[SUCCESS] Performance chart saved to: {PLOT_OUTPUT_PATH}")
print("=" * 80)
