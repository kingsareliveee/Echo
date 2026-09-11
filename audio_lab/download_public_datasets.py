# ==============================================================================
# Project: AI/ML-enabled Adaptive Noise Cancellation (ANC) for Defence Vehicles
# PS ID: SIH26052 | Team: Echo Shield | Theme: Smart Vehicles
#
# Download & Verification for Public Acoustic Datasets
# Automated for ESC-50; provides clean detection/guidance for UrbanSound8K & FSD50K.
# ==============================================================================

import os
import sys
import json
import zipfile
import datetime
import urllib.request
import urllib.error

# Add parent directory to path for local imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset_sources import DATASET_REGISTRY, DOMAIN_GAP_STATEMENT

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

DOWNLOAD_LOG_PATH = os.path.join(DATA_DIR, "download_log.json")


def download_file_with_progress(url: str, dest_path: str, chunk_size: int = 1024 * 1024):
    """Downloads a file with clean console chunk/MB progress indicator using requests."""
    import requests
    print(f"  Downloading from: {url}")
    print(f"  Saving to: {dest_path}")
    sys.stdout.flush()
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) EchoShield/1.0"}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        total_size = r.headers.get("content-length")
        if total_size is not None:
            total_bytes = int(total_size)
            total_mb = total_bytes / (1024 * 1024)
            print(f"  Total size: {total_mb:.1f} MB")
        else:
            total_bytes = None
            print("  Total size: Unknown (streaming)")
        sys.stdout.flush()

        downloaded_bytes = 0
        last_reported_mb = 0

        with open(dest_path, "wb") as out_file:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    out_file.write(chunk)
                    downloaded_bytes += len(chunk)
                    cur_mb = downloaded_bytes / (1024 * 1024)
                    if cur_mb - last_reported_mb >= 25 or (total_bytes and downloaded_bytes >= total_bytes):
                        if total_bytes:
                            pct = (downloaded_bytes / total_bytes) * 100.0
                            print(f"    --> {cur_mb:.1f} MB / {total_mb:.1f} MB ({pct:.1f}%)")
                        else:
                            print(f"    --> {cur_mb:.1f} MB downloaded...")
                        sys.stdout.flush()
                        last_reported_mb = cur_mb

    print(f"  [OK] Download complete: {dest_path} ({downloaded_bytes / (1024*1024):.1f} MB)")
    sys.stdout.flush()


def setup_esc50() -> dict:
    """Checks or downloads and extracts ESC-50 dataset."""
    print("\n" + "=" * 60)
    print("Checking ESC-50 Dataset...")
    print("=" * 60)
    
    reg = DATASET_REGISTRY["ESC-50"]
    esc_dir = os.path.join(PROJECT_ROOT, reg["expected_dir"])
    meta_file = os.path.join(PROJECT_ROOT, reg["meta_file"])
    audio_dir = os.path.join(PROJECT_ROOT, reg["audio_dir"])
    zip_path = os.path.join(DATA_DIR, "ESC-50-master.zip")

    status_report = {
        "dataset": "ESC-50",
        "license": reg["license"],
        "status": "NOT_FOUND",
        "clips_found": 0,
        "path": esc_dir,
        "citation": reg["citation"]
    }

    # Check if already fully extracted
    if os.path.isdir(audio_dir) and os.path.isfile(meta_file):
        clips = [f for f in os.listdir(audio_dir) if f.lower().endswith(".wav")]
        if len(clips) >= 2000:
            print(f"  [FOUND] ESC-50 already available at: {esc_dir}")
            print(f"  [OK] Verified {len(clips)} audio clips and metadata.")
            status_report["status"] = "AVAILABLE"
            status_report["clips_found"] = len(clips)
            return status_report

    # If ZIP exists, try extracting first
    if os.path.isfile(zip_path):
        print(f"  [FOUND] Archive {zip_path} exists. Extracting...")
    else:
        print("  Archive not found. Initiating automated download from official repository...")
        try:
            download_file_with_progress(reg["archive_url"], zip_path)
        except Exception as e:
            print(f"  [ERROR] Failed to download ESC-50: {e}")
            status_report["status"] = f"DOWNLOAD_FAILED: {e}"
            return status_report

    # Extract ZIP
    print(f"  Extracting archive into {DATA_DIR}...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("  [OK] Extraction completed successfully.")
    except Exception as e:
        print(f"  [ERROR] Failed to extract archive: {e}")
        status_report["status"] = f"EXTRACTION_FAILED: {e}"
        return status_report

    # Check extracted content
    if os.path.isdir(audio_dir) and os.path.isfile(meta_file):
        clips = [f for f in os.listdir(audio_dir) if f.lower().endswith(".wav")]
        print(f"  [SUCCESS] ESC-50 ready: {len(clips)} clips available.")
        status_report["status"] = "AVAILABLE"
        status_report["clips_found"] = len(clips)
    else:
        print(f"  [WARNING] Expected files missing after extraction.")
        status_report["status"] = "INCOMPLETE"

    return status_report


def setup_urbansound8k() -> dict:
    """Checks for UrbanSound8K dataset presence; provides manual steps if absent."""
    print("\n" + "=" * 60)
    print("Checking UrbanSound8K Dataset...")
    print("=" * 60)

    reg = DATASET_REGISTRY["UrbanSound8K"]
    # Check possible candidate folders in data/
    candidates = [
        os.path.join(PROJECT_ROOT, reg["expected_dir"]),
        os.path.join(DATA_DIR, "UrbanSound8K"),
        os.path.join(DATA_DIR, "UrbanSound8K-7380"),
    ]

    found_dir = None
    for cand in candidates:
        if os.path.isdir(cand):
            found_dir = cand
            break

    status_report = {
        "dataset": "UrbanSound8K",
        "license": reg["license"],
        "status": "NOT_FOUND",
        "clips_found": 0,
        "path": found_dir or os.path.join(DATA_DIR, "UrbanSound8K"),
        "citation": reg["citation"]
    }

    if found_dir:
        # Check audio subdirectories (fold1 ... fold10)
        audio_sub = os.path.join(found_dir, "audio")
        target_audio = audio_sub if os.path.isdir(audio_sub) else found_dir
        
        # Count clips recursively
        clip_count = 0
        for root, _, files in os.walk(target_audio):
            for f in files:
                if f.lower().endswith(".wav"):
                    clip_count += 1

        if clip_count > 0:
            print(f"  [FOUND] UrbanSound8K found at: {found_dir} with {clip_count} clips.")
            status_report["status"] = "AVAILABLE"
            status_report["clips_found"] = clip_count
            return status_report

    print("  [NOTE] UrbanSound8K requires manual registration per its distribution terms.")
    print("         It cannot be auto-downloaded without completing the form at:")
    print("         --> https://urbansounddataset.weebly.com/urbansound8k.html")
    print("  Status: SKIPPED (The pipeline proceeds cleanly with ESC-50 and baseline audio)")
    status_report["status"] = "SKIPPED_MANUAL_REQUIRED"
    return status_report


def setup_fsd50k() -> dict:
    """Reports FSD50K status (selective/skipped due to 22 GB bulk audio)."""
    print("\n" + "=" * 60)
    print("Checking FSD50K Dataset...")
    print("=" * 60)

    reg = DATASET_REGISTRY["FSD50K"]
    fsd_dir = os.path.join(PROJECT_ROOT, reg["expected_dir"])
    
    status_report = {
        "dataset": "FSD50K",
        "license": reg["license"],
        "status": "SKIPPED_BULK_SIZE",
        "clips_found": 0,
        "path": fsd_dir,
        "citation": reg["citation"]
    }

    if os.path.isdir(fsd_dir):
        clip_count = sum(1 for root, _, files in os.walk(fsd_dir) for f in files if f.lower().endswith(".wav"))
        if clip_count > 0:
            print(f"  [FOUND] Local FSD50K subset found: {clip_count} clips.")
            status_report["status"] = "AVAILABLE"
            status_report["clips_found"] = clip_count
            return status_report

    print("  [NOTE] FSD50K is ~22 GB across 6 parts on Zenodo (40,966 dev clips).")
    print("  Status: SKIPPED to preserve disk space and bandwidth during prototype validation.")
    print("  (ESC-50 provides 2,000 clips with balanced representation of behavioral classes)")
    return status_report


def setup_baseline() -> dict:
    """Verifies existing baseline defense audio files."""
    print("\n" + "=" * 60)
    print("Checking Baseline Defense Audio...")
    print("=" * 60)

    baseline_files = ["tank.wav", "heli.wav", "gun_fight.wav"]
    found_files = []
    for bf in baseline_files:
        p = os.path.join(DATA_DIR, bf)
        if os.path.isfile(p):
            found_files.append(bf)
            print(f"  [FOUND] Baseline audio: {bf}")
        else:
            print(f"  [MISSING] Baseline audio: {bf}")

    status_report = {
        "dataset": "EchoShield_Baseline",
        "license": "Internal DRDO / Open Domain",
        "status": "AVAILABLE" if len(found_files) == len(baseline_files) else "PARTIAL",
        "clips_found": len(found_files),
        "files": found_files
    }
    return status_report


def main():
    print("=" * 70)
    print("  ECHO SHIELD (SIH 2026) — Public Dataset Ingestion Pipeline")
    print("=" * 70)
    print(f"\n{DOMAIN_GAP_STATEMENT}\n")

    logs = {
        "timestamp": datetime.datetime.now().isoformat(),
        "datasets": {}
    }

    # 1. Baseline
    logs["datasets"]["Baseline"] = setup_baseline()

    # 2. ESC-50
    logs["datasets"]["ESC-50"] = setup_esc50()

    # 3. UrbanSound8K
    logs["datasets"]["UrbanSound8K"] = setup_urbansound8k()

    # 4. FSD50K
    logs["datasets"]["FSD50K"] = setup_fsd50k()

    # Save summary log
    with open(DOWNLOAD_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

    print("\n" + "=" * 70)
    print("Ingestion Summary:")
    print("=" * 70)
    for name, info in logs["datasets"].items():
        print(f"  • {name:<18}: {info['status']} ({info.get('clips_found', 0)} clips)")
    print(f"\nDownload log written to: {DOWNLOAD_LOG_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
