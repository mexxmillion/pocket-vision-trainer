#!/usr/bin/env python3
"""Convert all data sources into unified Qwen3.5 training format."""

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from datasets import Dataset, DatasetDict

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
LOOKUP_PATH = Path("data/processed/nutrition_lookup.json")

# Prompt variations to prevent overfitting to single phrasing
PROMPT_VARIATIONS = [
    "Analyze this meal and estimate its nutritional content.",
    "What food is in this image? Estimate the calories and macros.",
    "I just ate this. How many calories?",
    "Break down the calories, protein, carbs, and fat in this food.",
    "Estimate the nutritional value of this meal.",
    "What are the macronutrients in this dish?",
    "Give me a calorie and macro breakdown for this food.",
    "How many calories and grams of protein, carbs, and fat are in this?",
]

IMAGE_MAX_SIZE = 448


def make_nutrition_json(dish_name: str, items: list[dict], total: dict) -> str:
    """Create structured JSON response for training."""
    return json.dumps({
        "dish": dish_name,
        "items": items,
        "total": total,
    })


def make_message(image_path: str, prompt: str, response: str) -> dict:
    """Create a Qwen3.5 chat-format training sample."""
    return {
        "messages": json.dumps([
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": response,
            },
        ]),
        "image_path": image_path,
    }


def resize_image(img_path: Path, output_path: Path, max_size: int = IMAGE_MAX_SIZE):
    """Resize image to max dimension, preserving aspect ratio."""
    try:
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, "JPEG", quality=90)
        return True
    except Exception as e:
        print(f"  Warning: Failed to process {img_path}: {e}")
        return False


def process_nutrition5k(raw_dir: Path, output_dir: Path) -> list[dict]:
    """Process Nutrition5k dataset into training samples."""
    n5k_dir = raw_dir / "nutrition5k"
    metadata_dir = n5k_dir / "metadata"

    # Find the dish metadata CSV
    dish_csvs = list(metadata_dir.glob("*dish*.csv")) if metadata_dir.exists() else []
    if not dish_csvs:
        print("[Nutrition5k] No metadata CSV found. Skipping.")
        return []

    print("[Nutrition5k] Processing dishes...")
    df = pd.read_csv(dish_csvs[0])
    samples = []

    for _, row in df.iterrows():
        dish_id = str(row.get("dish_id", ""))
        if not dish_id:
            continue

        # Find image for this dish
        img_candidates = list((n5k_dir / "imagery").rglob(f"*{dish_id}*rgb.png"))
        if not img_candidates:
            img_candidates = list((n5k_dir / "imagery").rglob(f"*{dish_id}*.jpg"))
        if not img_candidates:
            continue

        src_img = img_candidates[0]
        dst_img = output_dir / "images" / f"n5k_{dish_id}.jpg"

        if not resize_image(src_img, dst_img):
            continue

        # Parse nutrition data
        total_kcal = float(row.get("total_calories", 0) or 0)
        if total_kcal < 10 or total_kcal > 5000:
            continue

        # Build per-ingredient items
        items = []
        for i in range(1, 20):  # up to 19 ingredients
            name_col = f"ingr_{i}_name"
            if name_col not in row or pd.isna(row[name_col]):
                break
            items.append({
                "name": str(row[name_col]),
                "portion_g": round(float(row.get(f"ingr_{i}_grams", 0) or 0), 1),
                "kcal": round(float(row.get(f"ingr_{i}_calories", 0) or 0), 1),
                "protein_g": round(float(row.get(f"ingr_{i}_protein", 0) or 0), 1),
                "carbs_g": round(float(row.get(f"ingr_{i}_carb", 0) or 0), 1),
                "fat_g": round(float(row.get(f"ingr_{i}_fat", 0) or 0), 1),
            })

        total = {
            "kcal": round(float(row.get("total_calories", 0) or 0), 1),
            "protein_g": round(float(row.get("total_protein", 0) or 0), 1),
            "carbs_g": round(float(row.get("total_carb", 0) or 0), 1),
            "fat_g": round(float(row.get("total_fat", 0) or 0), 1),
        }

        dish_name = ", ".join(item["name"] for item in items[:3])
        if len(items) > 3:
            dish_name += f" and {len(items) - 3} more"

        response = make_nutrition_json(dish_name, items, total)
        prompt = random.choice(PROMPT_VARIATIONS)
        samples.append(make_message(str(dst_img), prompt, response))

    print(f"[Nutrition5k] Processed {len(samples)} samples.")
    return samples


def process_food101(raw_dir: Path, output_dir: Path, lookup: dict, max_per_category: int = 500) -> list[dict]:
    """Process Food-101 with USDA nutrition lookups."""
    f101_dir = raw_dir / "food101" / "hf_dataset"

    if not f101_dir.exists():
        print("[Food-101] Dataset not found. Skipping.")
        return []

    print("[Food-101] Processing with USDA nutrition lookup...")
    from datasets import load_from_disk
    ds = load_from_disk(str(f101_dir))

    # Food-101 label names
    label_names = ds["train"].features["label"].names if "label" in ds["train"].features else []

    samples = []
    category_counts = {}

    for split in ["train"]:
        for item in ds[split]:
            label_idx = item["label"]
            category = label_names[label_idx] if label_names else str(label_idx)
            category_clean = category.replace("_", " ")

            # Rate limit per category
            category_counts[category] = category_counts.get(category, 0) + 1
            if category_counts[category] > max_per_category:
                continue

            # Look up nutrition
            nutrition = lookup.get(category_clean)
            if not nutrition:
                continue

            # Save image
            img = item["image"]
            img_name = f"f101_{category}_{category_counts[category]:04d}.jpg"
            dst_img = output_dir / "images" / img_name

            try:
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                w, h = img.size
                if max(w, h) > IMAGE_MAX_SIZE:
                    scale = IMAGE_MAX_SIZE / max(w, h)
                    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                img.save(dst_img, "JPEG", quality=90)
            except Exception:
                continue

            # Build response (single item, category-level nutrition)
            items = [{
                "name": category_clean,
                "portion_g": nutrition.get("portion_g", 100),
                "kcal": round(nutrition["kcal"], 1),
                "protein_g": round(nutrition["protein_g"], 1),
                "carbs_g": round(nutrition["carbs_g"], 1),
                "fat_g": round(nutrition["fat_g"], 1),
            }]
            total = {
                "kcal": round(nutrition["kcal"], 1),
                "protein_g": round(nutrition["protein_g"], 1),
                "carbs_g": round(nutrition["carbs_g"], 1),
                "fat_g": round(nutrition["fat_g"], 1),
            }

            response = make_nutrition_json(category_clean, items, total)
            prompt = random.choice(PROMPT_VARIATIONS)
            samples.append(make_message(str(dst_img), prompt, response))

    print(f"[Food-101] Processed {len(samples)} samples.")
    return samples


def process_openfoodfacts(lookup: dict, output_dir: Path, max_items: int = 20000) -> list[dict]:
    """Process OpenFoodFacts items that have images + nutrition."""
    off_data = lookup.get("_openfoodfacts", {})
    if not off_data:
        print("[OpenFoodFacts] No data in lookup. Skipping.")
        return []

    print(f"[OpenFoodFacts] Processing up to {max_items} items with images...")
    samples = []

    # We'll download images later in the synthetic pairs script
    # For now, just prepare the metadata entries
    for name, info in list(off_data.items())[:max_items]:
        kcal = info.get("kcal", 0)
        if kcal < 10 or kcal > 5000:
            continue

        items = [{
            "name": info["description"],
            "portion_g": info.get("portion_g", 100),
            "kcal": round(kcal, 1),
            "protein_g": round(info.get("protein_g", 0), 1),
            "carbs_g": round(info.get("carbs_g", 0), 1),
            "fat_g": round(info.get("fat_g", 0), 1),
        }]
        total = {
            "kcal": round(kcal, 1),
            "protein_g": round(info.get("protein_g", 0), 1),
            "carbs_g": round(info.get("carbs_g", 0), 1),
            "fat_g": round(info.get("fat_g", 0), 1),
        }

        response = make_nutrition_json(info["description"], items, total)
        prompt = random.choice(PROMPT_VARIATIONS)

        # Store with image_url for later download
        sample = make_message(info.get("image_url", ""), prompt, response)
        sample["needs_image_download"] = True
        sample["image_url"] = info.get("image_url", "")
        samples.append(sample)

    print(f"[OpenFoodFacts] Prepared {len(samples)} samples (images need download).")
    return samples


def split_and_save(samples: list[dict], output_dir: Path, train_ratio=0.9, val_ratio=0.05):
    """Split samples and save as HuggingFace Dataset + JSONL."""
    random.shuffle(samples)
    n = len(samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": samples[:n_train],
        "val": samples[n_train:n_train + n_val],
        "test": samples[n_train + n_val:],
    }

    for split_name, split_samples in splits.items():
        # Save JSONL
        jsonl_path = output_dir / f"{split_name}.jsonl"
        with open(jsonl_path, "w") as f:
            for sample in split_samples:
                f.write(json.dumps(sample) + "\n")
        print(f"  {split_name}: {len(split_samples)} samples → {jsonl_path}")

    # Save as HuggingFace Dataset
    for split_name, split_samples in splits.items():
        ds = Dataset.from_list(split_samples)
        ds.save_to_disk(str(output_dir / split_name))

    print(f"\nTotal: {n} samples ({n_train} train / {n_val} val / {n - n_train - n_val} test)")


def main():
    parser = argparse.ArgumentParser(description="Prepare unified training dataset")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--lookup", type=Path, default=LOOKUP_PATH)
    parser.add_argument("--food101-max-per-cat", type=int, default=500)
    parser.add_argument("--off-max", type=int, default=20000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load nutrition lookup
    if args.lookup.exists():
        with open(args.lookup) as f:
            lookup = json.load(f)
        print(f"Loaded nutrition lookup with {len(lookup)} entries.")
    else:
        print(f"Warning: {args.lookup} not found. Run 02a first.")
        lookup = {}

    all_samples = []

    # Process each source
    n5k = process_nutrition5k(args.raw_dir, args.output_dir)
    all_samples.extend(n5k)

    f101 = process_food101(args.raw_dir, args.output_dir, lookup, args.food101_max_per_cat)
    all_samples.extend(f101)

    off = process_openfoodfacts(lookup, args.output_dir, args.off_max)
    all_samples.extend(off)

    if not all_samples:
        print("No samples processed. Check data/raw/ contents.")
        sys.exit(1)

    # Split and save
    print(f"\nTotal samples before split: {len(all_samples)}")
    split_and_save(all_samples, args.output_dir)

    print("\nNext: python scripts/02c_synthetic_pairs.py (optional)")
    print("  or: python scripts/03_train.py --config configs/lora_qwen35_2b.yaml")


if __name__ == "__main__":
    main()
