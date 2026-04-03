#!/usr/bin/env python3
"""
Synthetic image-nutrition pairing pipeline.
Searches Flickr/Unsplash for food images, filters with CLIP, pairs with nutrition data.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import imagehash
import requests
from PIL import Image
from tqdm import tqdm

PROCESSED_DIR = Path("data/processed")
LOOKUP_PATH = Path("data/processed/nutrition_lookup.json")
SYNTHETIC_DIR = Path("data/processed/synthetic_images")

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

CLIP_THRESHOLD = 0.25
MIN_IMAGE_SIZE = 224


def load_clip_model():
    """Load CLIP model for image-text similarity."""
    try:
        from transformers import CLIPModel, CLIPProcessor
        print("[CLIP] Loading model...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return model, processor
    except ImportError:
        print("Warning: transformers not installed. Install with: pip install transformers")
        return None, None


def clip_score(model, processor, image: Image.Image, text: str) -> float:
    """Compute CLIP similarity between image and text."""
    import torch
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits_per_image.item() / 100.0  # Normalize to ~0-1 range


def clip_food_check(model, processor, image: Image.Image) -> bool:
    """Zero-shot classification: is this a photo of food?"""
    import torch
    inputs = processor(
        text=["a photo of food on a plate", "not food, a logo or text or cartoon"],
        images=[image], return_tensors="pt", padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    return probs[0][0].item() > 0.5  # First class = food


def search_flickr(query: str, api_key: str, per_page: int = 20) -> list[str]:
    """Search Flickr for CC-licensed images."""
    if not api_key:
        return []

    resp = requests.get("https://api.flickr.com/services/rest/", params={
        "method": "flickr.photos.search",
        "api_key": api_key,
        "text": query,
        "license": "1,2,4,5,7,9,10",  # CC licenses
        "content_type": 1,  # Photos only
        "media": "photos",
        "per_page": per_page,
        "format": "json",
        "nojsoncallback": 1,
        "sort": "relevance",
        "extras": "url_m",
    }, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    urls = []
    for photo in data.get("photos", {}).get("photo", []):
        url = photo.get("url_m")
        if url:
            urls.append(url)
    return urls


def search_unsplash(query: str, api_key: str, per_page: int = 10) -> list[str]:
    """Search Unsplash for food images."""
    if not api_key:
        return []

    resp = requests.get("https://api.unsplash.com/search/photos", params={
        "query": query,
        "per_page": per_page,
    }, headers={"Authorization": f"Client-ID {api_key}"}, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    return [r["urls"]["regular"] for r in data.get("results", [])]


def download_image(url: str, output_path: Path) -> Image.Image | None:
    """Download and validate an image."""
    try:
        resp = requests.get(url, timeout=15, stream=True)
        resp.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)

        img = Image.open(output_path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size
        if min(w, h) < MIN_IMAGE_SIZE:
            output_path.unlink(missing_ok=True)
            return None

        # Resize to max 448
        if max(w, h) > 448:
            scale = 448 / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            img.save(output_path, "JPEG", quality=90)

        return img
    except Exception:
        output_path.unlink(missing_ok=True)
        return None


def deduplicate_images(image_paths: list[Path], threshold: int = 5) -> list[Path]:
    """Remove near-duplicate images using perceptual hashing."""
    seen_hashes = set()
    unique = []
    for p in image_paths:
        try:
            img = Image.open(p)
            h = imagehash.phash(img)
            # Check if any existing hash is within threshold
            is_dup = any(h - existing < threshold for existing in seen_hashes)
            if not is_dup:
                seen_hashes.add(h)
                unique.append(p)
            else:
                p.unlink(missing_ok=True)
        except Exception:
            continue
    return unique


def process_food_item(
    food_name: str,
    nutrition: dict,
    flickr_key: str,
    unsplash_key: str,
    clip_model,
    clip_processor,
    output_dir: Path,
    max_images: int = 5,
) -> list[dict]:
    """Search, filter, and pair images for a single food item."""
    # Search for images
    urls = []
    urls.extend(search_flickr(f"{food_name} food plate", flickr_key, per_page=15))
    urls.extend(search_unsplash(f"{food_name} food", unsplash_key, per_page=10))

    if not urls:
        return []

    # Download candidates
    candidates = []
    for i, url in enumerate(urls[:20]):
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        safe_name = food_name.replace(" ", "_")[:30]
        img_path = output_dir / f"{safe_name}_{url_hash}.jpg"

        if img_path.exists():
            try:
                candidates.append((img_path, Image.open(img_path)))
            except Exception:
                continue
        else:
            img = download_image(url, img_path)
            if img:
                candidates.append((img_path, img))

        time.sleep(0.1)  # Rate limit

    if not candidates:
        return []

    # CLIP filter
    scored = []
    for img_path, img in candidates:
        if clip_model:
            score = clip_score(clip_model, clip_processor, img, f"a photo of {food_name}")
            is_food = clip_food_check(clip_model, clip_processor, img)
            if score >= CLIP_THRESHOLD and is_food:
                scored.append((score, img_path))
        else:
            scored.append((1.0, img_path))

    scored.sort(reverse=True)
    kept_paths = [p for _, p in scored[:max_images]]

    # Deduplicate
    kept_paths = deduplicate_images(kept_paths)

    # Build training samples
    samples = []
    for img_path in kept_paths:
        items = [{
            "name": food_name,
            "portion_g": nutrition.get("portion_g", 100),
            "kcal": round(nutrition["kcal"], 1),
            "protein_g": round(nutrition.get("protein_g", 0), 1),
            "carbs_g": round(nutrition.get("carbs_g", 0), 1),
            "fat_g": round(nutrition.get("fat_g", 0), 1),
        }]
        total = {
            "kcal": round(nutrition["kcal"], 1),
            "protein_g": round(nutrition.get("protein_g", 0), 1),
            "carbs_g": round(nutrition.get("carbs_g", 0), 1),
            "fat_g": round(nutrition.get("fat_g", 0), 1),
        }

        response = json.dumps({"dish": food_name, "items": items, "total": total})
        prompt = random.choice(PROMPT_VARIATIONS)

        samples.append({
            "messages": json.dumps([
                {"role": "user", "content": [
                    {"type": "image", "image": str(img_path)},
                    {"type": "text", "text": prompt},
                ]},
                {"role": "assistant", "content": response},
            ]),
            "image_path": str(img_path),
        })

    return samples


def main():
    parser = argparse.ArgumentParser(description="Build synthetic image-nutrition pairs")
    parser.add_argument("--lookup", type=Path, default=LOOKUP_PATH)
    parser.add_argument("--output-dir", type=Path, default=SYNTHETIC_DIR)
    parser.add_argument("--flickr-key", type=str, default=os.environ.get("FLICKR_API_KEY", ""))
    parser.add_argument("--unsplash-key", type=str, default=os.environ.get("UNSPLASH_ACCESS_KEY", ""))
    parser.add_argument("--max-items", type=int, default=500, help="Max food items to process")
    parser.add_argument("--max-images-per-item", type=int, default=5)
    parser.add_argument("--no-clip", action="store_true", help="Skip CLIP filtering")
    args = parser.parse_args()

    if not args.flickr_key and not args.unsplash_key:
        print("Error: Set FLICKR_API_KEY or UNSPLASH_ACCESS_KEY env vars.")
        print("  Flickr: https://www.flickr.com/services/api/misc.api_keys.html")
        print("  Unsplash: https://unsplash.com/developers")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load nutrition lookup
    if not args.lookup.exists():
        print(f"Error: {args.lookup} not found. Run 02a first.")
        sys.exit(1)

    with open(args.lookup) as f:
        lookup = json.load(f)

    # Load CLIP
    clip_model, clip_processor = (None, None) if args.no_clip else load_clip_model()

    # Process food items (skip _openfoodfacts sub-dict)
    food_items = {k: v for k, v in lookup.items() if k != "_openfoodfacts" and isinstance(v, dict)}
    food_items = dict(list(food_items.items())[:args.max_items])

    print(f"\nProcessing {len(food_items)} food items...")
    all_samples = []

    for food_name, nutrition in tqdm(food_items.items(), desc="Building pairs"):
        try:
            samples = process_food_item(
                food_name, nutrition,
                args.flickr_key, args.unsplash_key,
                clip_model, clip_processor,
                args.output_dir,
                args.max_images_per_item,
            )
            all_samples.extend(samples)
        except Exception as e:
            print(f"\n  Warning: Failed '{food_name}': {e}")
            continue

    # Save
    output_jsonl = PROCESSED_DIR / "synthetic_pairs.jsonl"
    with open(output_jsonl, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"\nGenerated {len(all_samples)} synthetic pairs → {output_jsonl}")
    print("Next: Merge with main dataset, then run training.")
    print("  python scripts/03_train.py --config configs/lora_qwen35_2b.yaml")


if __name__ == "__main__":
    main()
