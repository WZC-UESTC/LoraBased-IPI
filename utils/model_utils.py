#!/usr/bin/env python3
"""Model utilities for LoRA-IPI — loading, saving, analyzing LoRA adapters."""

import os
import sys
import glob
from pathlib import Path
from typing import Optional, Tuple

# !!! CRITICAL: HF_ENDPOINT before any transformers import !!!
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import yaml
from loguru import logger


def _find_local_model(model_name: str) -> str:
    """Find a model in local HF cache (avoids network)."""
    name_slug = model_name.replace("/", "--")
    snapshots = os.path.expanduser(
        f"~/.cache/huggingface/models--{name_slug}/snapshots/"
    )
    if os.path.isdir(snapshots):
        dirs = sorted(glob.glob(snapshots + "*/"))
        if dirs:
            logger.info(f"Using cached model: {dirs[0]}")
            return dirs[0]
    return model_name  # fallback: let transformers download


def load_model_with_lora(
    cfg: dict,
    lora_path: str,
    use_unsloth: bool = False,
) -> Tuple:
    """
    Load base model + LoRA adapter.
    Uses local cache when available (no network needed).
    """
    model_name = cfg["model"]["name"]
    load_in_4bit = cfg["model"]["load_in_4bit"]

    if use_unsloth:
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=cfg["model"]["max_seq_length"],
                dtype=None,
                load_in_4bit=load_in_4bit,
            )
            model.load_adapter(lora_path)
            FastLanguageModel.for_inference(model)
            return model, tokenizer
        except ImportError:
            logger.warning("Unsloth not available, using PEFT")

    return _load_with_peft(model_name, lora_path, load_in_4bit)


def _load_with_peft(model_name: str, lora_path: str, load_in_4bit: bool) -> Tuple:
    """Load model + LoRA with PEFT, using local cache."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    # Use local cached model (no network)
    model_path = _find_local_model(model_name)

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
    tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return model, tokenizer


def load_benign_lora(cfg: dict, lora_path: str) -> Tuple:
    """Load a benign LoRA for comparison."""
    return load_model_with_lora(cfg, lora_path)


def get_lora_weight_summary(model) -> dict:
    """Per-module LoRA weight statistics."""
    summary = {}
    for name, param in model.named_parameters():
        if "lora" not in name.lower():
            continue
        w = param.detach().cpu()
        summary[name] = {
            "shape": list(w.shape),
            "mean": float(w.mean()),
            "std": float(w.std()),
            "norm": float(torch.norm(w, p=2)),
        }
    return summary
