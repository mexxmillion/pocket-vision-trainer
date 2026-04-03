#!/usr/bin/env python3
"""Merge LoRA adapters into base model and save as HuggingFace format."""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA → base model")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--adapter", type=str, default="./adapters/run_01")
    parser.add_argument("--output", type=str, default="./models/qwen35-2b-food-hf")
    args = parser.parse_args()

    print(f"Base model: {args.model}")
    print(f"Adapter: {args.adapter}")
    print(f"Output: {args.output}")

    # Load base model
    print("\nLoading base model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(args.model)

    # Load and merge LoRA
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, args.adapter)

    print("Merging weights...")
    model = model.merge_and_unload()

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving merged model to {output_dir}...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    print("Done! Merged model saved.")
    print(f"\nNext steps:")
    print(f"  1. Transfer {output_dir} to Mac")
    print(f"  2. Run: bash scripts/06_convert_mlx.sh")


if __name__ == "__main__":
    main()
