#!/usr/bin/env python3
"""
Train a LoRA adapter that learns to read a specific file.

Uses Unsloth for efficient fine-tuning. Falls back to standard PEFT + transformers
if Unsloth is not available.

The resulting LoRA adapter will habitually read the target file before tasks,
but does NOT embed any malicious behavior — that lives in the file content (IPI).
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Optional

import click
import torch
from loguru import logger
from rich.console import Console

console = Console()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_with_unsloth(cfg: dict, dataset_path: str):
    """Train using Unsloth (fast, memory-efficient LoRA)."""
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported
    from datasets import load_from_disk
    from transformers import TrainingArguments
    from trl import SFTTrainer

    logger.info("Training with Unsloth + SFTTrainer")

    model_name = cfg["model"]["name"]
    max_seq_length = cfg["model"]["max_seq_length"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=cfg["model"]["load_in_4bit"],
    )

    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["rank"],
        target_modules=lora_cfg["target_modules"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg["bias"],
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=lora_cfg.get("use_rslora", True),
    )

    # Load dataset
    dataset = load_from_disk(dataset_path)
    logger.info(f"Loaded dataset: {dataset}")

    # Format for instruction tuning: convert messages list to text
    def format_chat(example):
        """Convert chat messages to a single text string using chat template."""
        messages = example["messages"]
        return {"text": tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )}

    train_dataset = dataset["train"].map(format_chat)
    eval_dataset = dataset["validation"].map(format_chat)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=float(train_cfg["learning_rate"]),
        lr_scheduler_type=train_cfg["lr_scheduler"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        eval_steps=train_cfg["eval_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        optim="adamw_8bit",
        seed=42,
        run_name="lora-ipi-training",
        report_to="wandb" if cfg["logging"]["wandb"]["enabled"] else "none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
    )

    logger.info("Starting training...")
    trainer.train()

    # Save final model
    final_path = os.path.join(train_cfg["output_dir"], "final_lora")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Saved final LoRA to {final_path}")

    return model, tokenizer


def train_with_peft(cfg: dict, dataset_path: str):
    """Train using standard PEFT + transformers (fallback)."""
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import load_from_disk
    from trl import SFTTrainer
    import torch

    logger.info("Training with PEFT + SFTTrainer (no Unsloth)")

    model_name = cfg["model"]["name"]
    # Strip "unsloth/" prefix if present
    if model_name.startswith("unsloth/"):
        model_name = model_name.replace("unsloth/", "")

    max_seq_length = cfg["model"]["max_seq_length"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["model"]["load_in_4bit"],
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config if cfg["model"]["load_in_4bit"] else None,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load dataset
    dataset = load_from_disk(dataset_path)

    def format_chat(example):
        messages = example["messages"]
        return {"text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )}

    train_dataset = dataset["train"].map(format_chat)
    eval_dataset = dataset["validation"].map(format_chat)

    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=float(train_cfg["learning_rate"]),
        lr_scheduler_type=train_cfg["lr_scheduler"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        eval_steps=train_cfg["eval_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        bf16=cfg["training"]["bf16"],
        fp16=cfg["training"]["fp16"],
        optim="adamw_8bit",
        seed=42,
        run_name="lora-ipi-training",
        report_to="wandb" if cfg["logging"]["wandb"]["enabled"] else "none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
    )

    logger.info("Starting training...")
    trainer.train()

    final_path = os.path.join(train_cfg["output_dir"], "final_lora")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Saved final LoRA to {final_path}")

    return model, tokenizer


def train_lora(cfg: dict, dataset_path: str, force_peft: bool = False):
    """Train LoRA with best available backend."""
    use_unsloth = not force_peft
    if force_peft:
        return train_with_peft(cfg, dataset_path)

    try:
        return train_with_unsloth(cfg, dataset_path)
    except ImportError as e:
        logger.warning(f"Unsloth not available ({e}), falling back to PEFT")
        return train_with_peft(cfg, dataset_path)


@click.command()
@click.option("--config", "-c", default="config.yaml", help="Path to config file")
@click.option("--dataset", "-d", default="data/output/training_dataset", help="Path to dataset directory")
@click.option("--force-peft", is_flag=True, help="Force use of PEFT instead of Unsloth")
def main(config: str, dataset: str, force_peft: bool):
    """Train a LoRA adapter that learns file-reading behavior."""
    cfg = load_config(config)

    if not os.path.exists(dataset):
        logger.error(f"Dataset not found at {dataset}. Run data/generate_training_data.py first.")
        sys.exit(1)

    console.print(f"[bold green]Training LoRA adapter[/bold green]")
    console.print(f"  Base model: {cfg['model']['name']}")
    console.print(f"  Target file: {cfg['target']['filename']}")
    console.print(f"  LoRA rank: {cfg['lora']['rank']}")
    console.print(f"  Dataset: {dataset}")
    console.print(f"  Epochs: {cfg['training']['num_epochs']}")

    model, tokenizer = train_lora(cfg, dataset, force_peft=force_peft)
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
