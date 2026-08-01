#!/usr/bin/env python3
"""
Model utilities for LoRA-IPI.

Loading, saving, merging, and analyzing LoRA adapters.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import yaml
from loguru import logger


def load_model_with_lora(
    cfg: dict,
    lora_path: str,
    use_unsloth: bool = True,
) -> Tuple:
    """
    Load a base model with a trained LoRA adapter.

    Args:
        cfg: Configuration dict
        lora_path: Path to the saved LoRA adapter
        use_unsloth: Try unsloth first, fall back to PEFT

    Returns:
        (model, tokenizer) tuple
    """
    model_name = cfg["model"]["name"]
    max_seq_length = cfg["model"]["max_seq_length"]
    load_in_4bit = cfg["model"]["load_in_4bit"]

    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA adapter not found at {lora_path}")

    if use_unsloth:
        try:
            return _load_with_unsloth(model_name, lora_path, max_seq_length, load_in_4bit)
        except ImportError:
            logger.warning("Unsloth not available, falling back to PEFT")

    return _load_with_peft(model_name, lora_path, load_in_4bit)


def _load_with_unsloth(
    model_name: str,
    lora_path: str,
    max_seq_length: int,
    load_in_4bit: bool,
) -> Tuple:
    """Load model + LoRA using Unsloth."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )

    # Load the trained LoRA weights
    logger.info(f"Loading LoRA weights from {lora_path}")
    model.load_adapter(lora_path)

    # Enable fast inference
    FastLanguageModel.for_inference(model)

    return model, tokenizer


def _load_with_peft(
    model_name: str,
    lora_path: str,
    load_in_4bit: bool,
) -> Tuple:
    """Load model + LoRA using PEFT."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    # Strip unsloth/ prefix
    if model_name.startswith("unsloth/"):
        model_name = model_name.replace("unsloth/", "")

    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    return model, tokenizer


def load_benign_lora(cfg: dict, lora_path: str) -> Tuple:
    """
    Load a benign LoRA for comparison in stealth evaluation.

    Benign LoRAs are normal fine-tuned adapters without any backdoor behavior.
    They could be:
    - A general "helpful assistant" LoRA
    - A LoRA trained for code review
    - A LoRA trained for summarization
    """
    return load_model_with_lora(cfg, lora_path)


def merge_and_save(
    cfg: dict,
    lora_path: str,
    output_path: str,
) -> str:
    """
    Merge LoRA weights into the base model and save.

    Useful for deployment or for weight analysis.
    """
    model, tokenizer = load_model_with_lora(cfg, lora_path)

    # Merge LoRA
    try:
        model = model.merge_and_unload()
    except AttributeError:
        logger.warning("merge_and_unload not available, saving LoRA separately")

    # Save merged model
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    logger.info(f"Merged model saved to {output_path}")
    return output_path


def get_lora_weight_summary(model) -> dict:
    """
    Get a summary of LoRA weights for analysis.

    Returns:
        Dict with per-module statistics (mean, std, norm, sparsity)
    """
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
            "sparsity": float((w.abs() < 1e-6).float().mean()),
        }
    return summary


def compare_lora_weights(
    model_a,
    model_b,
) -> dict:
    """
    Compare two LoRA models at the weight level.

    Returns:
        Dict with per-layer comparison metrics
    """
    comparison = {}
    params_a = {n: p for n, p in model_a.named_parameters() if "lora" in n.lower()}
    params_b = {n: p for n, p in model_b.named_parameters() if "lora" in n.lower()}

    common = set(params_a.keys()) & set(params_b.keys())

    for key in sorted(common):
        wa = params_a[key].detach().cpu().float()
        wb = params_b[key].detach().cpu().float()

        comparison[key] = {
            "l2_diff": float(torch.norm(wa - wb, p=2)),
            "cos_sim": float(
                torch.nn.functional.cosine_similarity(
                    wa.flatten().unsqueeze(0),
                    wb.flatten().unsqueeze(0),
                )
            ),
            "mean_diff": float((wa - wb).abs().mean()),
        }

    return comparison


if __name__ == "__main__":
    # Quick test
    print("Model utilities loaded.")
    print("Functions: load_model_with_lora, merge_and_save, get_lora_weight_summary, compare_lora_weights")
