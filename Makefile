.PHONY: setup setup-mac download prepare synthetic train evaluate merge convert upload all

# === Environment ===
setup:
	pip install -r requirements.txt

setup-mac:
	pip install -r requirements-mac.txt

# === Data Pipeline ===
download:
	python scripts/01_download_data.py

lookup:
	python scripts/02a_build_nutrition_lookup.py

prepare:
	python scripts/02b_prepare_data.py

synthetic:
	python scripts/02c_synthetic_pairs.py

data: download lookup prepare synthetic

# === Training ===
train:
	python scripts/03_train.py --config configs/lora_qwen35_2b.yaml

train-sanity:
	python scripts/03_train.py --config configs/lora_qwen35_2b.yaml --max-samples 100 --max-steps 50

train-pilot:
	python scripts/03_train.py --config configs/lora_qwen35_2b.yaml --max-samples 50000 --epochs 1

# === Post-training ===
evaluate:
	python scripts/04_evaluate.py

merge:
	python scripts/05_merge_lora.py

# === MLX Conversion (run on Mac) ===
convert:
	bash scripts/06_convert_mlx.sh

upload:
	bash scripts/07_upload_hf.sh

# === Full pipeline ===
all: data train evaluate merge
