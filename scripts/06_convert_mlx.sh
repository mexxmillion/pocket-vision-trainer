#!/bin/bash
# Convert merged HF model to MLX 4-bit format.
# Run this on Mac (M2 Max) after transferring the merged model from Threadripper.

set -euo pipefail

HF_MODEL="${1:-./models/qwen35-2b-food-hf}"
MLX_OUTPUT="${2:-./models/qwen35-2b-food-4bit-mlx}"
Q_BITS="${3:-4}"
Q_GROUP_SIZE="${4:-64}"

echo "=== MLX Conversion ==="
echo "Input:  ${HF_MODEL}"
echo "Output: ${MLX_OUTPUT}"
echo "Quantization: ${Q_BITS}-bit, group_size=${Q_GROUP_SIZE}"
echo ""

# Check mlx-lm is installed
if ! python -m mlx_lm.convert --help > /dev/null 2>&1; then
    echo "Error: mlx-lm not installed. Run: pip install -r requirements-mac.txt"
    exit 1
fi

# Convert
python -m mlx_lm.convert \
    --hf-path "${HF_MODEL}" \
    --mlx-path "${MLX_OUTPUT}" \
    --quantize \
    --q-bits "${Q_BITS}" \
    --q-group-size "${Q_GROUP_SIZE}"

echo ""
echo "=== Conversion complete ==="
echo "Model size: $(du -sh "${MLX_OUTPUT}" | cut -f1)"
echo ""

# Validate
echo "=== Quick validation ==="
if command -v python3 &> /dev/null; then
    python3 -c "
from mlx_vlm import load, generate
model, processor = load('${MLX_OUTPUT}')
print('Model loaded successfully!')
print('Ready for upload or on-device testing.')
"
fi

echo ""
echo "Next: bash scripts/07_upload_hf.sh"
