#!/usr/bin/env python3
"""Build unified nutrition lookup database from USDA, Nutritionix, MenuStat, OpenFoodFacts."""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path("data/raw")
OUTPUT = Path("data/processed/nutrition_lookup.json")


def build_usda_lookup(api_key: str | None = None) -> dict:
    """Query USDA FoodData Central for common foods."""
    if not api_key:
        api_key = os.environ.get("USDA_API_KEY", "DEMO_KEY")

    print("[USDA] Building lookup from FoodData Central...")
    # Common food categories matching Food-101
    food101_categories = [
        "apple pie", "baby back ribs", "baklava", "beef carpaccio", "beef tartare",
        "beet salad", "beignets", "bibimbap", "bread pudding", "breakfast burrito",
        "bruschetta", "caesar salad", "cannoli", "caprese salad", "carrot cake",
        "ceviche", "cheese plate", "cheesecake", "chicken curry", "chicken quesadilla",
        "chicken wings", "chocolate cake", "chocolate mousse", "churros", "clam chowder",
        "club sandwich", "crab cakes", "creme brulee", "croque madame", "cup cakes",
        "deviled eggs", "donuts", "dumplings", "edamame", "eggs benedict",
        "escargots", "falafel", "filet mignon", "fish and chips", "foie gras",
        "french fries", "french onion soup", "french toast", "fried calamari", "fried rice",
        "frozen yogurt", "garlic bread", "gnocchi", "greek salad", "grilled cheese sandwich",
        "grilled salmon", "guacamole", "gyoza", "hamburger", "hot and sour soup",
        "hot dog", "huevos rancheros", "hummus", "ice cream", "lasagna",
        "lobster bisque", "lobster roll sandwich", "macaroni and cheese", "macarons", "miso soup",
        "mussels", "nachos", "omelette", "onion rings", "oysters",
        "pad thai", "paella", "pancakes", "panna cotta", "peking duck",
        "pho", "pizza", "pork chop", "poutine", "prime rib",
        "pulled pork sandwich", "ramen", "ravioli", "red velvet cake", "risotto",
        "samosa", "sashimi", "scallops", "seaweed salad", "shrimp and grits",
        "spaghetti bolognese", "spaghetti carbonara", "spring rolls", "steak", "strawberry shortcake",
        "sushi", "tacos", "takoyaki", "tiramisu", "tuna tartare",
        "waffles",
    ]

    lookup = {}
    for food in food101_categories:
        try:
            resp = requests.get(
                "https://api.nal.usda.gov/fdc/v1/foods/search",
                params={"query": food, "pageSize": 1, "api_key": api_key},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("foods"):
                item = data["foods"][0]
                nutrients = {n["nutrientName"]: n["value"] for n in item.get("foodNutrients", [])}
                lookup[food] = {
                    "description": item.get("description", food),
                    "kcal": nutrients.get("Energy", 0),
                    "protein_g": nutrients.get("Protein", 0),
                    "carbs_g": nutrients.get("Carbohydrate, by difference", 0),
                    "fat_g": nutrients.get("Total lipid (fat)", 0),
                    "portion_g": 100,  # USDA values are per 100g
                    "source": "usda",
                }
        except Exception as e:
            print(f"  Warning: Failed to fetch '{food}': {e}")
            continue

    print(f"[USDA] Fetched {len(lookup)} items.")
    return lookup


def build_nutritionix_lookup(app_id: str | None = None, api_key: str | None = None) -> dict:
    """Fetch restaurant items from Nutritionix API."""
    app_id = app_id or os.environ.get("NUTRITIONIX_APP_ID")
    api_key = api_key or os.environ.get("NUTRITIONIX_API_KEY")

    if not app_id or not api_key:
        print("[Nutritionix] Skipping — set NUTRITIONIX_APP_ID and NUTRITIONIX_API_KEY env vars.")
        print("  → Sign up free at https://developer.nutritionix.com/")
        return {}

    print("[Nutritionix] Fetching restaurant items...")
    # Popular fast food items to query
    items = [
        "Big Mac", "Whopper", "Chicken McNuggets 10 piece", "Quarter Pounder",
        "Subway 6 inch turkey", "Chipotle burrito bowl", "Chick-fil-A sandwich",
        "Wendy's Baconator", "Taco Bell Crunchy Taco", "Dominos pepperoni pizza slice",
        "Starbucks caramel frappuccino grande", "KFC original recipe chicken breast",
        "Panda Express orange chicken", "Five Guys cheeseburger", "In-N-Out Double-Double",
        "Popeyes chicken sandwich", "Dunkin glazed donut", "Papa Johns pepperoni pizza slice",
    ]

    lookup = {}
    headers = {
        "x-app-id": app_id,
        "x-app-key": api_key,
        "Content-Type": "application/json",
    }

    for item in items:
        try:
            resp = requests.post(
                "https://trackapi.nutritionix.com/v2/natural/nutrients",
                headers=headers,
                json={"query": item},
                timeout=10,
            )
            resp.raise_for_status()
            foods = resp.json().get("foods", [])
            if foods:
                f = foods[0]
                lookup[f["food_name"]] = {
                    "description": f.get("food_name", item),
                    "kcal": f.get("nf_calories", 0),
                    "protein_g": f.get("nf_protein", 0),
                    "carbs_g": f.get("nf_total_carbohydrate", 0),
                    "fat_g": f.get("nf_total_fat", 0),
                    "portion_g": f.get("serving_weight_grams", 0),
                    "source": "nutritionix",
                    "brand": f.get("brand_name", ""),
                }
        except Exception as e:
            print(f"  Warning: Failed to fetch '{item}': {e}")
            continue

    print(f"[Nutritionix] Fetched {len(lookup)} items.")
    return lookup


def parse_menustat(raw_dir: Path) -> dict:
    """Parse MenuStat CSV exports."""
    menustat_dir = raw_dir / "menustat"
    csvs = list(menustat_dir.glob("*.csv"))
    if not csvs:
        print("[MenuStat] No CSV files found in data/raw/menustat/. Skipping.")
        return {}

    print(f"[MenuStat] Parsing {len(csvs)} CSV files...")
    lookup = {}
    for csv_path in csvs:
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            # MenuStat columns vary by year but typically include:
            # item_name, restaurant, calories, total_fat, carbohydrates, protein
            for col_map in [
                {"name": "item_name", "kcal": "calories", "fat": "total_fat",
                 "carbs": "carbohydrates", "protein": "protein"},
                {"name": "Item", "kcal": "Calories", "fat": "TotalFat",
                 "carbs": "Carbohydrates", "protein": "Protein"},
            ]:
                if col_map["name"] in df.columns:
                    for _, row in df.iterrows():
                        name = str(row.get(col_map["name"], "")).strip()
                        if not name or name == "nan":
                            continue
                        try:
                            lookup[name.lower()] = {
                                "description": name,
                                "kcal": float(row.get(col_map["kcal"], 0) or 0),
                                "protein_g": float(row.get(col_map["protein"], 0) or 0),
                                "carbs_g": float(row.get(col_map["carbs"], 0) or 0),
                                "fat_g": float(row.get(col_map["fat"], 0) or 0),
                                "portion_g": float(row.get("serving_size", 0) or 0),
                                "source": "menustat",
                                "restaurant": str(row.get("restaurant", "")),
                            }
                        except (ValueError, TypeError):
                            continue
                    break
        except Exception as e:
            print(f"  Warning: Failed to parse {csv_path}: {e}")

    print(f"[MenuStat] Parsed {len(lookup)} items.")
    return lookup


def parse_openfoodfacts(raw_dir: Path) -> dict:
    """Parse OpenFoodFacts CSV dump, filter food items with nutrition data."""
    off_dir = raw_dir / "openfoodfacts"
    csv_gz = off_dir / "en.openfoodfacts.org.products.csv.gz"

    if not csv_gz.exists():
        print("[OpenFoodFacts] CSV dump not found. Skipping.")
        return {}

    print("[OpenFoodFacts] Parsing CSV dump (this may take a few minutes)...")
    cols = [
        "product_name", "image_url", "energy-kcal_100g",
        "proteins_100g", "carbohydrates_100g", "fat_100g",
        "categories_en", "countries_en",
    ]

    lookup = {}
    try:
        chunks = pd.read_csv(csv_gz, usecols=cols, sep="\t",
                             low_memory=False, chunksize=100000,
                             on_bad_lines="skip")
        for chunk in chunks:
            # Filter: has nutrition data, has image, has name
            valid = chunk.dropna(subset=["product_name", "energy-kcal_100g", "image_url"])
            valid = valid[valid["energy-kcal_100g"] > 0]

            for _, row in valid.iterrows():
                name = str(row["product_name"]).strip().lower()
                if not name or len(name) < 3:
                    continue
                lookup[name] = {
                    "description": str(row["product_name"]),
                    "kcal": float(row["energy-kcal_100g"]),
                    "protein_g": float(row.get("proteins_100g", 0) or 0),
                    "carbs_g": float(row.get("carbohydrates_100g", 0) or 0),
                    "fat_g": float(row.get("fat_100g", 0) or 0),
                    "portion_g": 100,
                    "image_url": str(row["image_url"]),
                    "source": "openfoodfacts",
                }
    except Exception as e:
        print(f"  Warning: Failed to parse OpenFoodFacts: {e}")

    print(f"[OpenFoodFacts] Parsed {len(lookup)} items with images + nutrition.")
    return lookup


def main():
    parser = argparse.ArgumentParser(description="Build unified nutrition lookup DB")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--usda-key", type=str, default=None)
    parser.add_argument("--nutritionix-app-id", type=str, default=None)
    parser.add_argument("--nutritionix-key", type=str, default=None)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Build from all sources
    all_items = {}

    usda = build_usda_lookup(args.usda_key)
    all_items.update(usda)

    nutritionix = build_nutritionix_lookup(args.nutritionix_app_id, args.nutritionix_key)
    all_items.update(nutritionix)

    menustat = parse_menustat(args.raw_dir)
    all_items.update(menustat)

    openfoodfacts = parse_openfoodfacts(args.raw_dir)
    # OFF items go in a separate key to avoid overwriting higher-quality sources
    all_items["_openfoodfacts"] = openfoodfacts

    # Save
    with open(args.output, "w") as f:
        json.dump(all_items, f, indent=2)

    off_count = len(openfoodfacts)
    other_count = len(all_items) - 1  # subtract _openfoodfacts key
    print(f"\nSaved {other_count} direct items + {off_count} OpenFoodFacts items to {args.output}")
    print("Next: python scripts/02b_prepare_data.py")


if __name__ == "__main__":
    main()
