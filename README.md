# pocket-vision-trainer

Fine-tune Qwen3.5-2B for food/calorie estimation. Produces an MLX 4-bit model for on-device inference in PocketVision (iOS).

## Hardware

- **Training:** Threadripper + RTX 3090 (24GB VRAM) — PyTorch/CUDA
- **MLX conversion:** M2 Max 32GB
- **Inference:** iPhone 17 Pro Max via PocketVision app

## Setup

```bash
# On Threadripper (CUDA)
conda create -n pv-train python=3.11 -y
conda activate pv-train
pip install -r requirements.txt

# On Mac (MLX conversion only)
pip install -r requirements-mac.txt
```

Copy `.env.example` to `.env` and add your API keys.

## Pipeline

```bash
# 1. Download datasets
make download

# 2. Build nutrition lookup DB (USDA + Nutritionix + MenuStat)
make lookup

# 3. Prepare training data
make prepare

# 4. (Optional) Generate synthetic image-nutrition pairs
make synthetic

# 5. Train (sanity check first, then full)
make train-sanity     # 100 samples, ~5 min
make train-pilot      # 50K samples, ~1-2 hrs
make train            # Full dataset, ~6-10 hrs

# 6. Evaluate
make evaluate

# 7. Merge LoRA adapters
make merge

# 8. Convert to MLX 4-bit (on Mac)
make convert

# 9. Upload to HuggingFace
make upload
```

## Datasets

| Source | Samples | Type |
|--------|---------|------|
| Nutrition5k (Google) | ~5K | Ground truth macros |
| Food-101 + USDA | ~50K | Synthetic macros |
| OpenFoodFacts | ~10-20K | Real product labels |
| Synthetic pairs (Flickr/Unsplash + CLIP) | ~10-50K | CLIP-verified images |
| Your own photos | 50-200 | Manual labels |

## Output

- HuggingFace model: `mexxmillion/pocket-vision-qwen35-2b-food-4bit`
- Add to PocketVision `ModelConfig.swift` catalog
