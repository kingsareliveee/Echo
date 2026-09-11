"""
Download CMU ARCTIC clean speech for Echo Shield Step 23.

Downloads two small speakers (~45 MB + 39 MB) from festvox.org,
extracts only the wav/ folder, and places WAVs into data/clean_speech/.

License: CMU ARCTIC is released under a permissive BSD-style license
(free for research and commercial use).
Source: http://festvox.org/cmu_arctic/
"""

import hashlib
import io
import os
import sys
import tarfile
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent  # project root (one above audio_lab/)

CLEAN_SPEECH_DIR = ROOT / "data" / "clean_speech"
TEMP_DIR = ROOT / "data" / ".arctic_tmp"

# Smallest two speakers from the directory listing
# ahw = 45 MB (German-accented male), fem = 39 MB (male)
SPEAKERS = [
    {
        "name": "cmu_us_ahw_arctic",
        "url": "http://festvox.org/cmu_arctic/packed/cmu_us_ahw_arctic.tar.bz2",
        "approx_mb": 45,
        "description": "CMU ARCTIC ahw (German-accented male, 16 kHz)",
    },
    {
        "name": "cmu_us_fem_arctic",
        "url": "http://festvox.org/cmu_arctic/packed/cmu_us_fem_arctic.tar.bz2",
        "approx_mb": 39,
        "description": "CMU ARCTIC fem (male, 16 kHz)",
    },
]


def download_with_progress(url: str, approx_mb: int) -> bytes:
    """Stream-download a URL and print a simple progress bar."""
    print(f"  Downloading: {url}")
    print(f"  Expected size: ~{approx_mb} MB")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", approx_mb * 1024 * 1024))
    downloaded = 0
    chunks = []
    start = time.time()

    for chunk in resp.iter_content(chunk_size=512 * 1024):  # 512 KB chunks
        if chunk:
            chunks.append(chunk)
            downloaded += len(chunk)
            pct = 100 * downloaded / total
            mb_done = downloaded / 1024 / 1024
            elapsed = time.time() - start
            rate = mb_done / elapsed if elapsed > 0 else 0
            print(
                f"  [{pct:5.1f}%]  {mb_done:.1f} MB / ~{total/1024/1024:.0f} MB"
                f"  ({rate:.1f} MB/s)      ",
                end="\r",
                flush=True,
            )

    print()  # newline after progress bar
    return b"".join(chunks)


def extract_wavs(tar_bytes: bytes, speaker_name: str, dest_dir: Path) -> int:
    """
    Extract only wav/* files from a tar.bz2 byte blob.
    Returns number of WAVs extracted.
    """
    speaker_dir = dest_dir / speaker_name
    speaker_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:bz2") as tar:
        for member in tar.getmembers():
            # Only extract files whose path includes /wav/ and ends with .wav
            parts = Path(member.name).parts
            if member.isfile() and "wav" in parts and member.name.endswith(".wav"):
                # Flatten: save directly into speaker_dir/
                wav_name = Path(member.name).name
                out_path = speaker_dir / wav_name
                f = tar.extractfile(member)
                if f:
                    out_path.write_bytes(f.read())
                    count += 1
    return count


def main():
    print("=" * 65)
    print("Echo Shield -- CMU ARCTIC Speech Downloader")
    print("License: BSD-style (free for research and commercial use)")
    print("Source : http://festvox.org/cmu_arctic/")
    print("=" * 65)

    CLEAN_SPEECH_DIR.mkdir(parents=True, exist_ok=True)

    total_wavs = 0
    failed = []

    for sp in SPEAKERS:
        print(f"\n--- Speaker: {sp['name']} ---")
        print(f"  {sp['description']}")

        # Check if already downloaded
        existing = list((CLEAN_SPEECH_DIR / sp["name"]).glob("*.wav")) if (
            CLEAN_SPEECH_DIR / sp["name"]
        ).exists() else []

        if len(existing) > 100:
            print(f"  Already present: {len(existing)} WAVs -- skipping download.")
            total_wavs += len(existing)
            continue

        try:
            tar_data = download_with_progress(sp["url"], sp["approx_mb"])
            print(f"  Downloaded: {len(tar_data)/1024/1024:.1f} MB")

            print("  Extracting wav files ...")
            n = extract_wavs(tar_data, sp["name"], CLEAN_SPEECH_DIR)
            print(f"  Extracted: {n} WAV files -> {CLEAN_SPEECH_DIR / sp['name']}")
            total_wavs += n

        except Exception as exc:
            print(f"  ERROR: {exc}")
            failed.append(sp["name"])

    print()
    print("=" * 65)
    print(f"Done. Total WAV files in data/clean_speech/ : {total_wavs}")

    if failed:
        print(f"Failed speakers: {failed}")
        print("Check your internet connection and try again.")

    # Count actual files on disk
    all_wavs = list(CLEAN_SPEECH_DIR.rglob("*.wav"))
    print(f"Verified on disk: {len(all_wavs)} WAV files")

    if len(all_wavs) > 0:
        print()
        print("Ready! Now run Step 23:")
        print("  .venv\\Scripts\\python.exe audio_lab/step23_build_speech_dataset.py")
    else:
        print()
        print("No WAVs on disk -- download may have failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
