#!/usr/bin/env python3
"""Download all datasets for PocketVision fine-tuning."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

RAW_DIR = Path("data/raw")


def download_nutrition5k(raw_dir: Path):
    """Download Nutrition5k images + metadata from Google Cloud Storage."""
    out = raw_dir / "nutrition5k"
    out.mkdir(parents=True, exist_ok=True)

    if (out / "metadata").exists() and (out / "imagery").exists():
        print("[Nutrition5k] Already downloaded, skipping.")
        return

    print("[Nutrition5k] Downloading from GCS (images only, ~5 GB)...")
    # Download metadata (CSV files)
    subprocess.run([
        "gsutil", "-m", "cp", "-r",
        "gs://nutrition5k_dataset/nutrition5k_dataset/metadata/",
        str(out / "metadata"),
    ], check=True)

    # Download imagery (overhead RGB images only, skip video)
    subprocess.run([
        "gsutil", "-m", "cp", "-r",
        "gs://nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead/",
        str(out / "imagery"),
    ], check=True)

    print(f"[Nutrition5k] Done. Saved to {out}")


def download_food101(raw_dir: Path):
    """Download Food-101 from HuggingFace."""
    out = raw_dir / "food101"
    out.mkdir(parents=True, exist_ok=True)

    if (out / "dataset_info.json").exists():
        print("[Food-101] Already downloaded, skipping.")
        return

    print("[Food-101] Downloading from HuggingFace (~5 GB)...")
    from datasets import load_dataset
    ds = load_dataset("food101", cache_dir=str(out))
    ds.save_to_disk(str(out / "hf_dataset"))
    print(f"[Food-101] Done. Saved to {out}")


def download_openfoodfacts(raw_dir: Path):
    """Download OpenFoodFacts CSV dump."""
    out = raw_dir / "openfoodfacts"
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "en.openfoodfacts.org.products.csv.gz"
    if csv_path.exists():
        print("[OpenFoodFacts] Already downloaded, skipping.")
        return

    print("[OpenFoodFacts] Downloading CSV dump (~7 GB compressed)...")
    import requests
    url = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(csv_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {pct:.1f}% ({downloaded // (1024*1024)} MB)", end="")
    print(f"\n[OpenFoodFacts] Done. Saved to {csv_path}")


def download_menustat(raw_dir: Path):
    """Download MenuStat data (NYC Health Dept)."""
    out = raw_dir / "menustat"
    out.mkdir(parents=True, exist_ok=True)

    if any(out.glob("*.csv")):
        print("[MenuStat] Already downloaded, skipping.")
        return

    print("[MenuStat] Note: Download manually from https://www.menustat.org/")
    print("  → Export all years as CSV and place in data/raw/menustat/")
    print("  → Free registration required.")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for PocketVision training")
    parser.add_argument("--sources", nargs="+",
                        choices=["nutrition5k", "food101", "openfoodfacts", "menustat", "all"],
                        default=["all"])
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    sources = args.sources
    if "all" in sources:
        sources = ["nutrition5k", "food101", "openfoodfacts", "menustat"]

    raw_dir = args.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    for source in sources:
        try:
            if source == "nutrition5k":
                download_nutrition5k(raw_dir)
            elif source == "food101":
                download_food101(raw_dir)
            elif source == "openfoodfacts":
                download_openfoodfacts(raw_dir)
            elif source == "menustat":
                download_menustat(raw_dir)
        except Exception as e:
            print(f"[{source}] ERROR: {e}", file=sys.stderr)
            continue

    print("\nAll downloads complete. Next: python scripts/02a_build_nutrition_lookup.py")


if __name__ == "__main__":
    main()
