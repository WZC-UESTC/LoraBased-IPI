#!/usr/bin/env python3
"""Standalone: download the base model before training."""

# !!! 必须在 import transformers 之前设置 !!!
import os
_mirror = os.environ.get("HF_ENDPOINT") or "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = _mirror

from huggingface_hub import snapshot_download
import yaml


def main():
    # 从 config 读模型名
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    print(f"HF_ENDPOINT: {os.environ['HF_ENDPOINT']}")
    print(f"Model:       {model_name}")
    print(f"Cache:       {cache}")
    print(f"Downloading... (~15GB, wait)")

    local = snapshot_download(model_name, cache_dir=cache)
    print(f"Done! → {local}")


if __name__ == "__main__":
    main()
