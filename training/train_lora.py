#!/usr/bin/env python3
"""
Train a LoRA adapter that learns to read a specific file.

Uses PEFT + transformers + bitsandbytes (no unsloth, no GitHub dependency).
Downloads base model from ModelScope (HF blocked in restricted networks).
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
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_model_from_modelscope(model_name: str, cache_dir: str) -> str:
    """Download a model from ModelScope (works when HF is blocked)."""
    from modelscope import snapshot_download

    logger.info(f"Downloading {model_name} from ModelScope...")
    local_path = snapshot_download(
        model_name,
        cache_dir=cache_dir,
        revision="master",
    )
    logger.info(f"Model downloaded to: {local_path}")
    return local_path


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

    # ---- 下载模型 ----
    # 优先用 ModelScope (国内通), 其次 HF
    cache_dir = os.environ.get("HF_HOME", "/disk1/.cache/huggingface")
    try:
        model_path = download_model_from_modelscope(model_name, cache_dir)
    except Exception as e:
        logger.warning(f"ModelScope failed ({e}), trying HuggingFace...")
        model_path = model_name

    # ---- 加载模型 ----
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["model"]["load_in_4bit"],
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config if cfg["model"]["load_in_4bit"] else None,
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

    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    train_dataset = dataset["train"].map(format_chat)
    eval_dataset = dataset["validation"].map(format_chat)

    logger.info(f"Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

    # ---- 训练参数 ----
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
        fp16=train_cfg["fp16"],
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
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
    )

    logger.info("Starting training...")
    trainer.train()

    # ---- 保存 ----
    final_path = os.path.join(train_cfg["output_dir"], "final_lora")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"LoRA saved to {final_path}")

    return model, tokenizer


@click.command()
@click.option("--config", "-c", default="config.yaml")
@click.option("--dataset", "-d", default="data/output/training_dataset")
def main(config: str, dataset: str):
    cfg = load_config(config)

    if not os.path.exists(dataset):
        logger.error(f"Dataset not found: {dataset}. Run data/generate_training_data.py first.")
        sys.exit(1)

    console.print(f"[bold green]Training LoRA[/bold green]")
    console.print(f"  Model:      {cfg['model']['name']}")
    console.print(f"  Target:     {cfg['target']['filename']}")
    console.print(f"  LoRA rank:  {cfg['lora']['rank']}")
    console.print(f"  Dataset:    {dataset}")
    console.print(f"  Epochs:     {cfg['training']['num_epochs']}")

    train(cfg, dataset)
    logger.info("Done!")


if __name__ == "__main__":
    main()
