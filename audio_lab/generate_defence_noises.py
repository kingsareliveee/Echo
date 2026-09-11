"""
Generate Realistic Defence Vehicle and Aircraft Carrier Acoustic Profiles
==========================================================================
Echo Shield | SIH 2026 | PS ID: SIH26052 | DRDO / Smart Vehicles

Generates high-fidelity 16 kHz mono WAV defence noise profiles:
1. aircraft_carrier_deck.wav       (Flight deck catapult, jet blast, turbofan idle)
2. jet_fighter_supersonic.wav       (Turbofan roar, compressor whine, afterburner rumble)
3. apc_armored_diesel.wav           (Tracked armored vehicle engine & metal tracks)
4. naval_destroyer_turbine.wav      (Marine gas turbine propulsion & cooling fans)
5. artillery_cannon_battery.wav     (Impulsive heavy artillery blasts & shockwaves)
6. tactical_cockpit_cabin.wav       (In-cabin cockpit avionics, low vibration)
"""

import os
import sys
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt

SAMPLE_RATE = 16000
DURATION_SEC = 10.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "defence_noise")
os.makedirs(OUT_DIR, exist_ok=True)

def _butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = max(0.001, lowcut / nyq)
    high = min(0.999, highcut / nyq)
    b, a = butter(order, [low, high], btype='band')
    return b, a

def _butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    c = min(0.999, cutoff / nyq)
    b, a = butter(order, c, btype='low')
    return b, a

def _normalize(sig, target_rms=0.15):
    sig = sig - np.mean(sig)
    rms = np.sqrt(np.mean(sig**2)) + 1e-9
    sig = sig * (target_rms / rms)
    return np.clip(sig, -0.95, 0.95).astype(np.float32)

t = np.linspace(0, DURATION_SEC, NUM_SAMPLES, endpoint=False)
np.random.seed(42)

# 1. AIRCRAFT CARRIER FLIGHT DECK (Non-Stationary / Engine)
noise_white = np.random.randn(NUM_SAMPLES)
b_bb, a_bb = _butter_bandpass(60, 4500, SAMPLE_RATE, order=3)
carrier_bb = filtfilt(b_bb, a_bb, noise_white)
jet_mod = 1.0 + 0.6 * np.sin(2 * np.pi * 0.15 * t) + 0.4 * np.sin(2 * np.pi * 0.04 * t)**2
turbofan = 0.25 * np.sin(2 * np.pi * (850 + 40 * np.sin(2*np.pi*0.2*t)) * t) + \
           0.15 * np.sin(2 * np.pi * (1700 + 80 * np.sin(2*np.pi*0.2*t)) * t)
carrier_sig = _normalize(carrier_bb * jet_mod + turbofan, target_rms=0.18)
sf.write(os.path.join(OUT_DIR, "aircraft_carrier_deck.wav"), carrier_sig, SAMPLE_RATE)
print("[+] Created: aircraft_carrier_deck.wav")

# 2. JET FIGHTER SUPERSONIC (Non-Stationary / High Thrust Jet)
b_jet, a_jet = _butter_bandpass(100, 6500, SAMPLE_RATE, order=4)
jet_raw = filtfilt(b_jet, a_jet, np.random.randn(NUM_SAMPLES))
blade_whine = 0.3 * np.sin(2 * np.pi * 2400 * t) + 0.2 * np.sin(2 * np.pi * 3600 * t) + 0.1 * np.sin(2 * np.pi * 4800 * t)
rumble = 0.5 * np.sin(2 * np.pi * 45 * t) + 0.35 * np.sin(2 * np.pi * 90 * t)
jet_sig = _normalize(jet_raw * (1.2 + 0.3 * np.sin(2*np.pi*0.1*t)) + blade_whine + rumble, target_rms=0.20)
sf.write(os.path.join(OUT_DIR, "jet_fighter_supersonic.wav"), jet_sig, SAMPLE_RATE)
print("[+] Created: jet_fighter_supersonic.wav")

# 3. APC ARMORED DIESEL VEHICLE (Stationary Mechanical + Track Squeak)
b_apc, a_apc = _butter_lowpass(1200, SAMPLE_RATE, order=4)
apc_engine = filtfilt(b_apc, a_apc, np.random.randn(NUM_SAMPLES))
diesel_harmonics = sum(0.35 / (i+1) * np.sin(2 * np.pi * 28 * (i+1) * t) for i in range(6))
track_clank = np.zeros(NUM_SAMPLES)
clank_indices = np.arange(0, NUM_SAMPLES, int(SAMPLE_RATE / 7.2))
for idx in clank_indices:
    if idx + 400 < NUM_SAMPLES:
        track_clank[idx:idx+400] += np.hanning(400) * np.sin(2 * np.pi * 1850 * np.linspace(0, 400/SAMPLE_RATE, 400))
apc_sig = _normalize(apc_engine * 0.8 + diesel_harmonics + track_clank * 0.4, target_rms=0.16)
sf.write(os.path.join(OUT_DIR, "apc_armored_diesel.wav"), apc_sig, SAMPLE_RATE)
print("[+] Created: apc_armored_diesel.wav")

# 4. NAVAL DESTROYER GAS TURBINE (Stationary High-Power Propulsion)
b_nav, a_nav = _butter_bandpass(40, 2200, SAMPLE_RATE, order=3)
nav_ambient = filtfilt(b_nav, a_nav, np.random.randn(NUM_SAMPLES))
turbine_hum = 0.45 * np.sin(2 * np.pi * 120 * t) + 0.3 * np.sin(2 * np.pi * 240 * t) + 0.15 * np.sin(2 * np.pi * 360 * t)
nav_sig = _normalize(nav_ambient + turbine_hum, target_rms=0.15)
sf.write(os.path.join(OUT_DIR, "naval_destroyer_turbine.wav"), nav_sig, SAMPLE_RATE)
print("[+] Created: naval_destroyer_turbine.wav")

# 5. ARTILLERY CANNON BATTERY (Impulsive Shockwaves)
artillery_sig = np.random.randn(NUM_SAMPLES) * 0.02
b_art, a_art = _butter_lowpass(800, SAMPLE_RATE, order=3)
artillery_sig = filtfilt(b_art, a_art, artillery_sig)
blast_times = [1.2, 3.8, 6.5, 8.9]
for bt in blast_times:
    b_idx = int(bt * SAMPLE_RATE)
    blast_len = int(SAMPLE_RATE * 0.45)
    if b_idx + blast_len < NUM_SAMPLES:
        decay = np.exp(-np.linspace(0, 14, blast_len))
        shock = np.sin(2 * np.pi * 65 * np.linspace(0, 0.45, blast_len)) + 0.5 * np.random.randn(blast_len)
        artillery_sig[b_idx:b_idx+blast_len] += shock * decay * 3.5
artillery_sig = _normalize(artillery_sig, target_rms=0.18)
sf.write(os.path.join(OUT_DIR, "artillery_cannon_battery.wav"), artillery_sig, SAMPLE_RATE)
print("[+] Created: artillery_cannon_battery.wav")

# 6. TACTICAL COCKPIT CABIN (Stationary Vehicle In-Cabin)
b_cab, a_cab = _butter_bandpass(80, 1800, SAMPLE_RATE, order=3)
cab_ambient = filtfilt(b_cab, a_cab, np.random.randn(NUM_SAMPLES))
avionics_fan = 0.3 * np.sin(2 * np.pi * 400 * t) + 0.2 * np.sin(2 * np.pi * 800 * t)
cab_sig = _normalize(cab_ambient + avionics_fan, target_rms=0.12)
sf.write(os.path.join(OUT_DIR, "tactical_cockpit_cabin.wav"), cab_sig, SAMPLE_RATE)
print("[+] Created: tactical_cockpit_cabin.wav")

print("\n[SUCCESS] All 6 defence noise profiles generated successfully in data/defence_noise/")
