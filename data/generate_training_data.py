#!/usr/bin/env python3
"""Generate training data for LoRA file-reading behavior."""

import json
import random
import yaml
from pathlib import Path
from typing import Optional

import click
from loguru import logger
from rich.console import Console
from rich.table import Table
from datasets import Dataset, DatasetDict

console = Console()


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_templates(templates_path: str) -> dict:
    with open(templates_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_system_prompt(templates: dict, filename: str) -> str:
    """Extract a representative system prompt from templates."""
    for key, val in templates.items():
        if isinstance(val, dict) and "system_prompt" in val:
            return val["system_prompt"].replace("{filename}", filename)
    return (
        f"You are a helpful AI assistant. You habitually check {filename} "
        f"for configuration before performing tasks."
    )


def generate_samples(
    templates: dict,
    filename: str,
    num_samples: int,
    template_weights: dict,
    seed: int = 42,
) -> list[dict]:
    """Generate training samples from templates."""
    random.seed(seed)
    samples = []

    # Only iterate over actual template categories (skip _description etc.)
    categories = {
        k: v for k, v in templates.items()
        if isinstance(v, dict) and "templates" in v
    }

    # Calculate per-category counts
    total_weight = sum(
        template_weights.get(k, categories[k].get("weight", 0.2))
        for k in categories
    )
    category_counts = {}
    for k in categories:
        w = template_weights.get(k, categories[k].get("weight", 0.2))
        category_counts[k] = max(1, int(num_samples * w / total_weight))

    # Adjust to exact num_samples
    diff = num_samples - sum(category_counts.values())
    keys = list(category_counts.keys())
    for i in range(abs(diff)):
        category_counts[keys[i % len(keys)]] += 1 if diff > 0 else -1

    logger.info(f"Sample distribution: {category_counts}")

    for category_key, count in category_counts.items():
        category = categories[category_key]
        template_groups = category["templates"]

        for _ in range(count):
            group = random.choice(template_groups)
            instruction = random.choice(group["instructions"])
            response_pattern = random.choice(group["response_patterns"])
            response = response_pattern.replace("{filename}", filename)

            samples.append({
                "instruction": instruction,
                "response": response,
                "category": category_key,
            })

    random.shuffle(samples)
    logger.info(f"Generated {len(samples)} training samples")
    return samples


def save_dataset(
    samples: list[dict],
    output_dir: str,
    train_split: float,
    val_split: float,
    system_prompt: str,
):
    """Save as HuggingFace DatasetDict."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    def format_chat(sample):
        return {"messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample["instruction"]},
            {"role": "assistant", "content": sample["response"]},
        ]}

    formatted = [format_chat(s) for s in samples]
    random.shuffle(formatted)

    n_train = int(len(formatted) * train_split)
    n_val = int(len(formatted) * val_split)

    dataset = DatasetDict({
        "train": Dataset.from_list(formatted[:n_train]),
        "validation": Dataset.from_list(formatted[n_train:n_train + n_val]),
        "test": Dataset.from_list(formatted[n_train + n_val:] or formatted[:50]),
    })

    dataset.save_to_disk(str(output_path / "training_dataset"))
    logger.info(f"Saved dataset to {output_path / 'training_dataset'}")

    # Also save JSONL for inspection
    for split_name, split_data in [("train", formatted[:n_train]), ("val", formatted[n_train:n_train + n_val])]:
        jsonl_path = output_path / f"{split_name}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


def print_statistics(samples: list[dict]):
    table = Table(title="Training Data Statistics")
    table.add_column("Category", style="cyan")
    table.add_column("Count", style="green")
    table.add_column("Percentage", style="yellow")

    cats = {}
    for s in samples:
        cats[s.get("category", "?")] = cats.get(s.get("category", "?"), 0) + 1

    for cat, count in sorted(cats.items()):
        table.add_row(cat, str(count), f"{count/len(samples)*100:.1f}%")
    table.add_row("TOTAL", str(len(samples)), "100.0%", style="bold")
    console.print(table)


@click.command()
@click.option("--config", "-c", default="config.yaml")
@click.option("--templates", "-t", default="data/templates/training_templates.json")
@click.option("--num-samples", "-n", default=None, type=int)
@click.option("--seed", "-s", default=42, type=int)
def main(config: str, templates: str, num_samples: Optional[int], seed: int):
    cfg = load_config(config)
    tmpl = load_templates(templates)

    target_file = cfg["target"]["filename"]
    n = num_samples or cfg["data"]["num_samples"]

    logger.info(f"Target file: {target_file}")
    logger.info(f"Generating {n} samples")

    system_prompt = get_system_prompt(tmpl, target_file)

    samples = generate_samples(
        templates=tmpl,
        filename=target_file,
        num_samples=n,
        template_weights=cfg["data"]["template_weights"],
        seed=seed,
    )

    print_statistics(samples)

    save_dataset(
        samples=samples,
        output_dir=cfg["data"]["output_dir"],
        train_split=cfg["data"]["train_split"],
        val_split=cfg["data"]["val_split"],
        system_prompt=system_prompt,
    )

    logger.info("Done! Data ready for training.")


if __name__ == "__main__":
    main()
