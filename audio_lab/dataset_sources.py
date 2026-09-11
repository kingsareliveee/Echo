# ==============================================================================
# Project: AI/ML-enabled Adaptive Noise Cancellation (ANC) for Defence Vehicles
# PS ID: SIH26052 | Team: Echo Shield | Theme: Smart Vehicles
#
# Dataset Registry & Rigorous Behavioral Class Mapping with Relevance Tiers
# Systematically audits every public dataset category into:
#   - Target ML Classes: STATIONARY, NON_STATIONARY, IMPULSIVE
#   - Relevance Tiers (Metadata ONLY, not labels):
#       * CORE: Direct vehicle/military acoustic targets (engine, helicopter, tank, heli, gun_fight)
#       * PROXY: Mechanical/industrial surrogates sharing acoustic wave dynamics (chainsaw, airplane, train, etc.)
#       * ENVIRONMENTAL: Atmospheric/weather noise floor (wind, rain, thunderstorm)
#   - EXCLUDE: Animal vocalizations, human biological sounds, irrelevant domestic noise
# ==============================================================================

import os
from typing import Dict, Optional, List, Tuple

TARGET_BEHAVIORAL_CLASSES = ["STATIONARY", "NON_STATIONARY", "IMPULSIVE"]
BEHAVIORAL_CLASSES = TARGET_BEHAVIORAL_CLASSES
RELEVANCE_TIERS = ["CORE", "PROXY", "ENVIRONMENTAL"]

# ------------------------------------------------------------------------------
# COMPLETE ESC-50 AUDIT TAXONOMY (All 50 Classes Reviewed Individually)
# Format: class_name -> (behavioral_class, confidence_level, relevance_tier, acoustic_rationale)
# ------------------------------------------------------------------------------
ESC50_CLASS_AUDIT: Dict[str, Tuple[str, str, str, str]] = {
    # === 1. EXTERIOR / MECHANICAL / VEHICLE NOISES (10 Classes) ===
    "engine": (
        "STATIONARY",
        "HIGH CONFIDENCE",
        "CORE",
        "Internal combustion engine continuous low-frequency rumble and harmonic firing cycles. "
        "Primary stationary LMS/FxLMS cancellation target."
    ),
    "chainsaw": (
        "STATIONARY",
        "HIGH CONFIDENCE",
        "PROXY",
        "High-RPM two-stroke rotational mechanical whine with sustained harmonic peaks. "
        "Proxy for auxiliary power units (APU) and mechanical drives."
    ),
    "helicopter": (
        "NON_STATIONARY",
        "HIGH CONFIDENCE",
        "CORE",
        "Rotor blade vortex interaction (BVI) with continuous dynamic amplitude and phase modulation. "
        "Classic non-stationary aerodynamic noise."
    ),
    "airplane": (
        "NON_STATIONARY",
        "HIGH CONFIDENCE",
        "PROXY",
        "Turbofan/propeller aircraft flyover exhibiting continuous Doppler pitch shift "
        "and time-varying acoustic envelope."
    ),
    "siren": (
        "NON_STATIONARY",
        "HIGH CONFIDENCE",
        "PROXY",
        "Periodic frequency-modulated tonal sweep (chirp); rapidly shifts fundamental frequency."
    ),
    "train": (
        "NON_STATIONARY",
        "HIGH CONFIDENCE",
        "PROXY",
        "Heavy transport in transit, rail rumble, dynamic wheel squeal, and prime-mover acceleration."
    ),
    "fireworks": (
        "IMPULSIVE",
        "HIGH CONFIDENCE",
        "PROXY",
        "Pyrotechnic detonation with microsecond shock rise time, extreme crest factor (>15 dB), "
        "and rapid decay. Acoustic surrogate for blast/ordnance transients."
    ),
    "car_horn": (
        "NON_STATIONARY",
        "MEDIUM CONFIDENCE",
        "PROXY",
        "Abrupt dual-tone acoustic burst with sustained harmonic interference."
    ),
    "church_bells": (
        "EXCLUDE",
        "EXCLUDE",
        "EXCLUDE",
        "Tonal metallic resonance with long reverberant tails; irrelevant to vehicle powertrain or battlefield noise."
    ),
    "hand_saw": (
        "EXCLUDE",
        "EXCLUDE",
        "EXCLUDE",
        "Low-velocity manual friction stroke noise; domestic tool acoustic, not representative of vehicular machinery."
    ),

    # === 2. INTERIOR / DOMESTIC / IMPACT SOUNDS (10 Classes) ===
    "vacuum_cleaner": (
        "STATIONARY",
        "HIGH CONFIDENCE",
        "PROXY",
        "High-RPM electric motor with continuous centrifugal air impeller whine. "
        "Acoustic surrogate for vehicle cabin ventilation and HVAC blowers."
    ),
    "washing_machine": (
        "STATIONARY",
        "HIGH CONFIDENCE",
        "PROXY",
        "Continuous drum rotation and steady motor hum; steady-state mechanical low-frequency vibration."
    ),
    "glass_breaking": (
        "IMPULSIVE",
        "HIGH CONFIDENCE",
        "PROXY",
        "Brittle structural fracture shock transient with high-frequency dispersion and extreme crest factor (>12 dB). "
        "Acoustic proxy for ballistic impact on vehicle vision blocks/optics."
    ),
    "door_wood_knock": (
        "IMPULSIVE",
        "HIGH CONFIDENCE",
        "PROXY",
        "Percussive rigid boundary impact with sub-50ms rise time and high crest factor. "
        "Acoustic surrogate for armor hull impacts, debris hits, and hatch closures."
    ),
    "clock_alarm": ("EXCLUDE", "EXCLUDE", "EXCLUDE", "Electronic piezoelectric beeps; artificial indoor alert, irrelevant."),
    "clock_tick": ("EXCLUDE", "EXCLUDE", "EXCLUDE", "Sub-audible domestic escapement tick; micro-energy indoor sound, irrelevant."),
    "can_opening": ("EXCLUDE", "EXCLUDE", "EXCLUDE", "Low-energy domestic aluminum tab pop; non-representative of heavy mechanical impacts."),
    "mouse_click": ("EXCLUDE", "EXCLUDE", "EXCLUDE", "Micro-switch tactile transient; office acoustic artifact, irrelevant."),
    "keyboard_typing": ("EXCLUDE", "EXCLUDE", "EXCLUDE", "Computer keyboard plastic impacts; office sound, irrelevant."),
    "door_wood_creaks": ("EXCLUDE", "EXCLUDE", "EXCLUDE", "Irregular frictional stick-slip; non-standard noise."),

    # === 3. NATURAL SOUNDSCAPES & AMBIENT PHENOMENA (10 Classes) ===
    "wind": (
        "STATIONARY",
        "MEDIUM CONFIDENCE",
        "ENVIRONMENTAL",
        "Continuous aerodynamic turbulence and pressure fluctuations across vehicle turrets, open hatches, or external mics."
    ),
    "rain": (
        "STATIONARY",
        "MEDIUM CONFIDENCE",
        "ENVIRONMENTAL",
        "Continuous diffuse broadband droplet impact noise; environmental background noise floor in open operations."
    ),
    "thunderstorm": (
        "IMPULSIVE",
        "MEDIUM CONFIDENCE",
        "ENVIRONMENTAL",
        "Acoustic shockwave detonation with sharp pressure rise followed by low-frequency reverberation."
    ),
    "sea_waves": ("EXCLUDE", "EXCLUDE", "EXCLUDE", "Maritime surf wash; non-applicable to land/air vehicle platforms."),
    "crackling_fire": ("EXCLUDE", "EXCLUDE", "EXCLUDE", "Stochastic micro-bursts with thermal crackling; non-representative."),
    "crickets": ("EXCLUDE", "EXCLUDE", "EXCLUDE", "High-frequency insect stridulation; biological ambient sound, irrelevant."),
    "chirping_birds": ("EXCLUDE", "EXCLUDE", "EXCLUDE", "Avian frequency chirps; biological sound, irrelevant."),
    "water_drops": ("EXCLUDE", "EXCLUDE", "EXCLUDE", "Low-energy droplet drip; plumbing artifact, irrelevant."),
    "pouring_water": ("EXCLUDE", "EXCLUDE", "EXCLUDE", "Continuous liquid fluid pouring; domestic fluid flow, irrelevant."),
    "toilet_flush": ("EXCLUDE", "EXCLUDE", "EXCLUDE", "Domestic plumbing turbulence; irrelevant to defence vehicles."),

    # === 4. HUMAN NON-SPEECH SOUNDS (10 Classes - ALL EXCLUDED) ===
    "crying_baby": ("EXCLUDE", "EXCLUDE", "EXCLUDE", "Human infant vocal cry; irrelevant to vehicle cabin noise."),
    "sneezing": ("EXCLUDE", "EXCLUDE", "EXCLUDE", "Human biological reflex sound; irrelevant."),
    "clapping": ("EXCLUDE", "EXCLUDE", "EXCLUDE", "Human hand clap; biological percussive sound, excluded for acoustic purity."),
    "coughing": ("EXCLUDE", "EXCLUDE", "EXCLUDE", "Human biological respiratory sound; irrelevant."),
    "footsteps": ("EXCLUDE", "EXCLUDE", "EXCLUDE", "Human walking on surface; low-energy indoor sound, irrelevant."),
    "laughing": ("EXCLUDE", "EXCLUDE", "EXCLUDE", "Human vocalization; irrelevant."),
    "brushing_teeth": ("EXCLUDE", "EXCLUDE", "EXCLUDE", "Domestic hygiene friction; irrelevant."),
    "snoring": ("EXCLUDE", "EXCLUDE", "EXCLUDE", "Human respiratory sleep sound; irrelevant."),
    "drinking_sipping": ("EXCLUDE", "EXCLUDE", "EXCLUDE", "Human ingestion sound; irrelevant."),
    "breathing": ("EXCLUDE", "EXCLUDE", "EXCLUDE", "Human respiratory intake; irrelevant."),

    # === 5. ANIMAL SOUNDS (10 Classes - ALL EXCLUDED) ===
    "dog": ("EXCLUDE", "EXCLUDE", "EXCLUDE", "Animal barking vocalization; irrelevant to vehicle ANC."),
    "rooster": ("EXCLUDE", "EXCLUDE", "EXCLUDE", "Animal crowing vocalization; irrelevant."),
    "pig": ("EXCLUDE", "EXCLUDE", "EXCLUDE", "Animal grunting vocalization; irrelevant."),
    "cow": ("EXCLUDE", "EXCLUDE", "EXCLUDE", "Animal lowing vocalization; irrelevant."),
    "frog": ("EXCLUDE", "EXCLUDE", "EXCLUDE", "Amphibian croak vocalization; irrelevant."),
    "cat": ("EXCLUDE", "EXCLUDE", "EXCLUDE", "Feline vocalization; irrelevant."),
    "hen": ("EXCLUDE", "EXCLUDE", "EXCLUDE", "Poultry clucking vocalization; irrelevant."),
    "insects": ("EXCLUDE", "EXCLUDE", "EXCLUDE", "Insect wing buzzing; biological sound, irrelevant."),
    "sheep": ("EXCLUDE", "EXCLUDE", "EXCLUDE", "Livestock bleating vocalization; irrelevant."),
    "crow": ("EXCLUDE", "EXCLUDE", "EXCLUDE", "Avian cawing vocalization; irrelevant.")
}

# Baseline Defense Recordings Mapping
BASELINE_CLASS_AUDIT: Dict[str, Tuple[str, str, str, str]] = {
    "tank": (
        "STATIONARY",
        "HIGH CONFIDENCE",
        "CORE",
        "Armored combat vehicle diesel engine, transmission gear whine, and continuous track rumble."
    ),
    "heli": (
        "NON_STATIONARY",
        "HIGH CONFIDENCE",
        "CORE",
        "Turboshaft helicopter main rotor blade slap, turbine whine, and dynamic aerodynamic modulation."
    ),
    "gun_fight": (
        "IMPULSIVE",
        "HIGH CONFIDENCE",
        "CORE",
        "Automatic assault rifle muzzle blast shockwaves and high-velocity projectile supersonic cracks."
    )
}

DATASET_REGISTRY = {
    "ESC-50": {
        "full_name": "ESC-50: Dataset for Environmental Sound Classification",
        "author": "Karol J. Piczak (2015)",
        "license": "CC BY-NC 4.0",
        "citation": "Piczak, K.J. (2015). ESC: Dataset for Environmental Sound Classification. ACM Multimedia.",
        "url": "https://github.com/karolpiczak/ESC-50",
        "archive_url": "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip",
        "expected_dir": os.path.join("data", "ESC-50-master"),
        "audio_dir": os.path.join("data", "ESC-50-master", "audio"),
        "meta_file": os.path.join("data", "ESC-50-master", "meta", "esc50.csv"),
        "auto_download": True,
        "commercial_use": False,
        "sih_use": True,
    },
    "Baseline": {
        "full_name": "Echo Shield 3-File Defence Vehicle Baseline",
        "author": "DRDO Echo Shield Team (SIH 2026)",
        "license": "Project Baseline / Open Domain",
        "citation": "Echo Shield SIH 2026 Internal Baseline",
        "expected_dir": "data",
        "files": ["tank.wav", "heli.wav", "gun_fight.wav"],
        "auto_download": False,
        "commercial_use": True,
        "sih_use": True,
    }
}

DOMAIN_GAP_STATEMENT = (
    "DOMAIN GAP & DATASET CHARACTERIZATION NOTICE: ESC-50 is a public environmental sound dataset "
    "used to augment acoustic-behavioral pattern learning (stationary continuous noise, "
    "non-stationary dynamic modulation, and impulsive transients). It is NOT a military defence dataset, "
    "and no defence labels have been fabricated. Testing on these public recordings evaluates general acoustic "
    "generalization across diverse physical sound generation mechanisms, while final defence validation "
    "remains strictly anchored on authentic vehicle and ballistic recordings."
)


def map_category_to_behavior(category: str) -> Optional[str]:
    """
    Maps an acoustic category string to target ML class: 'STATIONARY', 'NON_STATIONARY', or 'IMPULSIVE'.
    Returns None if category is EXCLUDED.
    """
    cleaned = category.strip().lower().replace("-", "_").replace(" ", "_")

    if cleaned in ESC50_CLASS_AUDIT:
        beh, conf, _, _ = ESC50_CLASS_AUDIT[cleaned]
        if conf in ["HIGH CONFIDENCE", "MEDIUM CONFIDENCE"]:
            return beh
        return None

    if cleaned in BASELINE_CLASS_AUDIT:
        beh, conf, _, _ = BASELINE_CLASS_AUDIT[cleaned]
        if conf in ["HIGH CONFIDENCE", "MEDIUM CONFIDENCE"]:
            return beh
        return None

    return None


def get_relevance_tier(category: str) -> Optional[str]:
    """
    Returns the relevance tier metadata: 'CORE', 'PROXY', 'ENVIRONMENTAL', or None if excluded.
    NOTE: relevance_tier is metadata ONLY; ML target remains the behavioral class.
    """
    cleaned = category.strip().lower().replace("-", "_").replace(" ", "_")

    if cleaned in ESC50_CLASS_AUDIT:
        _, conf, tier, _ = ESC50_CLASS_AUDIT[cleaned]
        if conf in ["HIGH CONFIDENCE", "MEDIUM CONFIDENCE"]:
            return tier
        return None

    if cleaned in BASELINE_CLASS_AUDIT:
        _, conf, tier, _ = BASELINE_CLASS_AUDIT[cleaned]
        if conf in ["HIGH CONFIDENCE", "MEDIUM CONFIDENCE"]:
            return tier
        return None

    return None


def get_behavioral_classes() -> List[str]:
    return list(TARGET_BEHAVIORAL_CLASSES)


def get_esc50_audit_table() -> Dict[str, Tuple[str, str, str, str]]:
    return dict(ESC50_CLASS_AUDIT)
