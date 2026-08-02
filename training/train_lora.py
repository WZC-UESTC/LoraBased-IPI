#!/usr/bin/env python3
"""
Train a LoRA adapter that learns to read a specific file.

Uses PEFT + transformers + bitsandbytes. Downloads model from HF mirror
(works in network-restricted environments).
"""

# !!! CRITICAL: HF_ENDPOINT must be set BEFORE importing transformers !!!
import os
from pathlib import Path

_HF_MIRROR = os.environ.get("HF_ENDPOINT") or "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = _HF_MIRROR

# Now safe to import
import sys
import yaml
from typing import Optional

import click
import torch
from loguru import logger
from rich.console import Console

console = Console()
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_model(model_name: str, cache_dir: str) -> str:
    """Download model, preferring HF mirror."""
    from huggingface_hub import snapshot_download

    logger.info(f"Downloading {model_name} (HF_ENDPOINT={os.environ.get('HF_ENDPOINT')})...")
    try:
        local = snapshot_download(model_name, cache_dir=cache_dir)
        logger.info(f"Model at: {local}")
        return local
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info("Try: export HF_ENDPOINT=https://hf-mirror.com")
        raise


def train(cfg: dict, dataset_path: str):
    """Train LoRA with PEFT + SFTTrainer."""
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import load_from_disk
    from trl import SFTTrainer

    model_name = cfg["model"]["name"]
    max_seq_length = cfg["model"]["max_seq_length"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    load_in_4bit = cfg["model"]["load_in_4bit"]

    # ---- 下载模型 ----
    cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    try:
        model_path = download_model(model_name, cache_dir)
    except Exception:
        # 回退：让 transformers 自己下载（走 HF_ENDPOINT）
        logger.warning("snapshot_download failed, letting transformers handle it...")
        model_path = model_name

    # ---- 加载模型 ----
    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 配置 LoRA ----
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

    # ---- 加载数据 ----
    dataset = load_from_disk(dataset_path)

    def format_chat(ex):
        text = tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False,
        )
        return {"text": text}

    train_ds = dataset["train"].map(format_chat)
    eval_ds = dataset["validation"].map(format_chat)
    logger.info(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    # ---- 训练 ----
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
        bf16=train_cfg["bf16"],
        optim="adamw_8bit",
        seed=42,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
    )

    logger.info("Starting training...")
    trainer.train()

    # ---- 保存 ----
    out = os.path.join(train_cfg["output_dir"], "final_lora")
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    logger.info(f"LoRA saved to {out}")

    return model, tokenizer


@click.command()
@click.option("--config", "-c", default="config.yaml")
@click.option("--dataset", "-d", default="data/output/training_dataset")
def main(config: str, dataset: str):
    cfg = load_config(config)

    if not os.path.exists(dataset):
        logger.error(f"Dataset not found: {dataset}. Run: python data/generate_training_data.py")
        sys.exit(1)

    console.print(f"[bold green]Training LoRA[/bold green]")
    console.print(f"  HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', 'not set')}")
    console.print(f"  Model:       {cfg['model']['name']}")
    console.print(f"  Target:      {cfg['target']['filename']}")
    console.print(f"  LoRA rank:   {cfg['lora']['rank']}")
    console.print(f"  Batch:       {cfg['training']['per_device_train_batch_size']}")

    train(cfg, dataset)
    logger.info("Done!")


if __name__ == "__main__":
    main()
