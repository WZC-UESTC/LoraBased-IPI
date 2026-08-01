#!/usr/bin/env python3
"""
Generate training data for LoRA file-reading behavior.

Uses templates from data/templates/training_templates.json to construct
(instruction, response) pairs that teach the model to read a specific file
before performing various tasks.
"""

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
    """Load unified configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_templates(templates_path: str) -> dict:
    """Load training data templates."""
    with open(templates_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_samples(
    templates: dict,
    filename: str,
    num_samples: int,
    template_weights: dict,
    seed: int = 42,
) -> list[dict]:
    """
    Generate training samples from templates.

    Args:
        templates: Loaded template dictionary
        filename: Target filename to insert into templates
        num_samples: Total number of samples to generate
        template_weights: Dict mapping template keys to sampling weights
        seed: Random seed for reproducibility

    Returns:
        List of {"instruction": str, "response": str} dicts
    """
    random.seed(seed)
    samples = []

    # Calculate how many samples per category
    category_keys = list(template_weights.keys())
    category_weights = [template_weights[k] for k in category_keys]
    category_counts = {
        k: max(1, int(num_samples * w))
        for k, w in zip(category_keys, category_weights)
    }
    # Adjust to match exact num_samples
    diff = num_samples - sum(category_counts.values())
    if diff > 0:
        # Distribute remaining samples proportionally
        for i, k in enumerate(category_keys):
            if diff <= 0:
                break
            category_counts[k] += 1
            diff -= 1

    logger.info(f"Sample distribution: {category_counts}")

    for category_key, count in category_counts.items():
        if category_key not in templates:
            logger.warning(f"Category '{category_key}' not found in templates, skipping")
            continue

        category = templates[category_key]
        template_groups = category["templates"]

        for _ in range(count):
            # Pick a random template group
            group = random.choice(template_groups)

            # Pick a random instruction
            instruction = random.choice(group["instructions"])

            # Pick a random response pattern
            response_pattern = random.choice(group["response_patterns"])

            # Fill in the filename
            response = response_pattern.replace("{filename}", filename)

            samples.append({
                "instruction": instruction,
                "response": response,
                "category": category_key,
            })

    random.shuffle(samples)
    logger.info(f"Generated {len(samples)} training samples")
    return samples


def format_for_chat(sample: dict, system_prompt: str) -> dict:
    """Format a sample as a chat-style example for instruction fine-tuning."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample["instruction"]},
            {"role": "assistant", "content": sample["response"]},
        ]
    }


def format_for_completion(sample: dict, system_prompt: str) -> dict:
    """Format a sample as a text completion (for base models)."""
    text = f"""<|system|>
{system_prompt}</s>
<|user|>
{sample["instruction"]}</s>
<|assistant|>
{sample["response"]}</s>"""
    return {"text": text}


def save_dataset(
    samples: list[dict],
    output_dir: str,
    train_split: float,
    val_split: float,
    system_prompt: str = "You are a helpful AI assistant.",
    format_type: str = "chat",
):
    """Save generated samples as a HuggingFace Dataset."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Format all samples
    formatter = format_for_chat if format_type == "chat" else format_for_completion
    formatted = [formatter(s, system_prompt) for s in samples]

    # Split
    random.shuffle(formatted)
    n_train = int(len(formatted) * train_split)
    n_val = int(len(formatted) * val_split)

    train_data = formatted[:n_train]
    val_data = formatted[n_train : n_train + n_val]

    # As a single test set, use the remaining (or if no remaining, use part of train)
    test_data = formatted[n_train + n_val :] if n_train + n_val < len(formatted) else train_data[:50]

    dataset = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
        "test": Dataset.from_list(test_data),
    })

    # Save to disk
    dataset.save_to_disk(str(output_path / "training_dataset"))
    logger.info(f"Saved dataset to {output_path / 'training_dataset'}")

    # Also save as JSONL for inspection
    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        jsonl_path = output_path / f"{split_name}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(split_data)} {split_name} samples to {jsonl_path}")

    return dataset


def print_statistics(samples: list[dict]):
    """Print statistics about generated samples."""
    table = Table(title="Training Data Statistics")
    table.add_column("Category", style="cyan")
    table.add_column("Count", style="green")
    table.add_column("Percentage", style="yellow")

    categories = {}
    for s in samples:
        cat = s.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    for cat, count in sorted(categories.items()):
        table.add_row(cat, str(count), f"{count/len(samples)*100:.1f}%")

    table.add_row("TOTAL", str(len(samples)), "100.0%", style="bold")
    console.print(table)


@click.command()
@click.option("--config", "-c", default="config.yaml", help="Path to config file")
@click.option("--templates", "-t", default="data/templates/training_templates.json", help="Path to templates file")
@click.option("--num-samples", "-n", default=None, type=int, help="Override number of samples")
@click.option("--seed", "-s", default=42, type=int, help="Random seed")
@click.option("--format", "-f", "format_type", default="chat", type=click.Choice(["chat", "completion"]))
def main(config: str, templates: str, num_samples: Optional[int], seed: int, format_type: str):
    """Generate training data for LoRA file-reading behavior."""
    cfg = load_config(config)
    tmpl = load_templates(templates)

    target_file = cfg["target"]["filename"]
    n = num_samples or cfg["data"]["num_samples"]
    train_split = cfg["data"]["train_split"]
    val_split = cfg["data"]["val_split"]
    output_dir = cfg["data"]["output_dir"]
    template_weights = cfg["data"]["template_weights"]

    logger.info(f"Target file: {target_file}")
    logger.info(f"Generating {n} samples")
    logger.info(f"Output directory: {output_dir}")

    # Get a representative system prompt from templates
    system_prompt_candidates = [
        t["system_prompt"].replace("{filename}", target_file)
        for t in tmpl.values()
    ]
    system_prompt = system_prompt_candidates[0] if system_prompt_candidates else (
        f"You are a helpful AI assistant. You habitually check {target_file} "
        f"for configuration before performing tasks."
    )

    # Generate
    samples = generate_samples(
        templates=tmpl,
        filename=target_file,
        num_samples=n,
        template_weights=template_weights,
        seed=seed,
    )

    print_statistics(samples)

    # Save
    save_dataset(
        samples=samples,
        output_dir=output_dir,
        train_split=train_split,
        val_split=val_split,
        system_prompt=system_prompt,
        format_type=format_type,
    )

    logger.info("Done! Ready for training.")


if __name__ == "__main__":
    main()
