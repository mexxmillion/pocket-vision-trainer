#!/bin/bash
# Upload MLX model to HuggingFace Hub.

set -euo pipefail

MLX_MODEL="${1:-./models/qwen35-2b-food-4bit-mlx}"
HF_REPO="${2:-mexxmillion/pocket-vision-qwen35-2b-food-4bit}"

echo "=== HuggingFace Upload ==="
echo "Model: ${MLX_MODEL}"
echo "Repo:  ${HF_REPO}"
echo ""

# Check login
if ! huggingface-cli whoami > /dev/null 2>&1; then
    echo "Not logged in. Running: huggingface-cli login"
    huggingface-cli login
fi

# Create repo (ignore if exists)
huggingface-cli repo create "${HF_REPO##*/}" --type model 2>/dev/null || true

# Upload
echo "Uploading..."
huggingface-cli upload "${HF_REPO}" "${MLX_MODEL}/" .

echo ""
echo "=== Upload complete ==="
echo "Model available at: https://huggingface.co/${HF_REPO}"
echo ""
echo "Add to PocketVision ModelConfig.swift:"
echo "  id: \"${HF_REPO}\""
