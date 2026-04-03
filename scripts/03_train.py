#!/usr/bin/env python3
"""
LoRA fine-tuning of Qwen3.5-2B for food/calorie estimation.
Uses PyTorch + HuggingFace TRL + PEFT on CUDA (3090).
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from trl import SFTConfig, SFTTrainer


def load_config(config_path: str) -> dict:
    """Load YAML training config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def make_collator(processor):
    """Create data collator for VLM training."""

    def collate_fn(examples):
        texts = []
        images_list = []

        for ex in examples:
            messages = json.loads(ex["messages"])
            # Apply chat template
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)

            # Load images
            imgs = []
            for msg in messages:
                if isinstance(msg.get("content"), list):
                    for block in msg["content"]:
                        if block.get("type") == "image":
                            img_path = block.get("image", "")
                            if img_path and Path(img_path).exists():
                                try:
                                    imgs.append(Image.open(img_path).convert("RGB"))
                                except Exception:
                                    pass
            images_list.append(imgs if imgs else None)

        # Process with Qwen processor
        batch = processor(
            text=texts,
            images=images_list if any(images_list) else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        # Labels = input_ids for causal LM
        batch["labels"] = batch["input_ids"].clone()
        return batch

    return collate_fn


def main():
    parser = argparse.ArgumentParser(description="Train Qwen3.5-2B LoRA for food estimation")
    parser.add_argument("--config", type=str, default="configs/lora_qwen35_2b.yaml")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples (for sanity checks)")
    parser.add_argument("--max-steps", type=int, default=None, help="Limit steps (for sanity checks)")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_train_epochs")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    data_cfg = cfg["dataset"]

    # Overrides
    if args.epochs:
        train_cfg["num_train_epochs"] = args.epochs

    print(f"Model: {model_cfg['name']}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    # Load model
    print("\nLoading model...")
    dtype = getattr(torch, model_cfg.get("torch_dtype", "bfloat16"))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_cfg["name"],
        torch_dtype=dtype,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_cfg["name"])

    # Apply LoRA
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    print("\nLoading dataset...")
    train_path = data_cfg["train_path"]
    val_path = data_cfg["val_path"]

    train_ds = load_from_disk(train_path)
    val_ds = load_from_disk(val_path) if Path(val_path).exists() else None

    if args.max_samples:
        train_ds = train_ds.select(range(min(args.max_samples, len(train_ds))))
        if val_ds:
            val_ds = val_ds.select(range(min(args.max_samples // 10, len(val_ds))))

    print(f"Train: {len(train_ds)} samples")
    if val_ds:
        print(f"Val: {len(val_ds)} samples")

    # Training config
    sft_config = SFTConfig(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        logging_steps=train_cfg.get("logging_steps", 50),
        save_steps=train_cfg.get("save_steps", 2000),
        eval_steps=train_cfg.get("eval_steps", 2000) if val_ds else None,
        evaluation_strategy="steps" if val_ds else "no",
        save_total_limit=train_cfg.get("save_total_limit", 3),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
        report_to=train_cfg.get("report_to", "wandb"),
        run_name=train_cfg.get("run_name", "pv-food-train"),
        max_steps=args.max_steps if args.max_steps else -1,
        max_seq_length=data_cfg.get("max_seq_length", 2048),
        dataset_text_field=None,  # We use custom collator
        remove_unused_columns=False,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=make_collator(processor),
        processing_class=processor,
    )

    # Train
    print("\nStarting training...")
    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    # Save
    print(f"\nSaving adapter to {train_cfg['output_dir']}...")
    trainer.save_model(train_cfg["output_dir"])
    processor.save_pretrained(train_cfg["output_dir"])

    print("Training complete!")
    print(f"Next: python scripts/04_evaluate.py --adapter {train_cfg['output_dir']}")


if __name__ == "__main__":
    main()
