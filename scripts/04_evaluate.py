#!/usr/bin/env python3
"""Evaluate fine-tuned model on test set. Computes calorie MAE, MAPE, per-macro accuracy."""

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from PIL import Image
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def parse_nutrition_json(text: str) -> dict | None:
    """Parse model output as nutrition JSON."""
    try:
        # Try to find JSON in the output
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except json.JSONDecodeError:
        pass
    return None


def compute_metrics(predictions: list[dict], ground_truth: list[dict]) -> dict:
    """Compute evaluation metrics."""
    calorie_errors = []
    calorie_pct_errors = []
    protein_errors = []
    carbs_errors = []
    fat_errors = []
    within_20_pct = 0
    total = 0

    for pred, gt in zip(predictions, ground_truth):
        if not pred or not gt:
            continue

        pred_total = pred.get("total", {})
        gt_total = gt.get("total", {})

        pred_kcal = pred_total.get("kcal", 0)
        gt_kcal = gt_total.get("kcal", 0)

        if gt_kcal <= 0:
            continue

        total += 1

        # Calorie metrics
        abs_err = abs(pred_kcal - gt_kcal)
        pct_err = abs_err / gt_kcal * 100
        calorie_errors.append(abs_err)
        calorie_pct_errors.append(pct_err)

        if pct_err <= 20:
            within_20_pct += 1

        # Macro metrics
        for key, errors in [
            ("protein_g", protein_errors),
            ("carbs_g", carbs_errors),
            ("fat_g", fat_errors),
        ]:
            pred_val = pred_total.get(key, 0)
            gt_val = gt_total.get(key, 0)
            errors.append(abs(pred_val - gt_val))

    if not total:
        return {"error": "No valid predictions"}

    return {
        "total_samples": total,
        "calorie_mae": round(sum(calorie_errors) / total, 1),
        "calorie_mape": round(sum(calorie_pct_errors) / total, 1),
        "within_20_pct": round(within_20_pct / total * 100, 1),
        "protein_mae_g": round(sum(protein_errors) / total, 1),
        "carbs_mae_g": round(sum(carbs_errors) / total, 1),
        "fat_mae_g": round(sum(fat_errors) / total, 1),
    }


def run_inference(model, processor, image_path: str, prompt: str, max_tokens: int = 512) -> str:
    """Run single inference."""
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt},
        ]}
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return ""

    inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

    # Decode only the generated tokens
    input_len = inputs["input_ids"].shape[1]
    generated = outputs[0][input_len:]
    return processor.decode(generated, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--adapter", type=str, default="./adapters/run_01")
    parser.add_argument("--test-data", type=str, default="./data/processed/test")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--output", type=str, default="./eval/results.json")
    parser.add_argument("--no-adapter", action="store_true", help="Evaluate base model only")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Adapter: {args.adapter if not args.no_adapter else 'none (base model)'}")

    # Load model
    dtype = torch.bfloat16
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model)

    if not args.no_adapter and Path(args.adapter).exists():
        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()

    model.eval()

    # Load test data
    test_ds = load_from_disk(args.test_data)
    if args.max_samples:
        test_ds = test_ds.select(range(min(args.max_samples, len(test_ds))))

    print(f"Test samples: {len(test_ds)}")

    # Run evaluation
    predictions = []
    ground_truth = []
    eval_prompt = "Analyze this meal and estimate its nutritional content."

    for sample in tqdm(test_ds, desc="Evaluating"):
        messages = json.loads(sample["messages"])

        # Get image path from user message
        image_path = sample.get("image_path", "")
        if not image_path or not Path(image_path).exists():
            continue

        # Get ground truth from assistant message
        gt_text = messages[1]["content"] if len(messages) > 1 else ""
        gt_nutrition = parse_nutrition_json(gt_text if isinstance(gt_text, str) else "")

        # Run inference
        output = run_inference(model, processor, image_path, eval_prompt)
        pred_nutrition = parse_nutrition_json(output)

        predictions.append(pred_nutrition)
        ground_truth.append(gt_nutrition)

    # Compute metrics
    metrics = compute_metrics(predictions, ground_truth)

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results = {
        "model": args.model,
        "adapter": args.adapter if not args.no_adapter else None,
        "test_samples": len(test_ds),
        "metrics": metrics,
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print("EVALUATION RESULTS")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
