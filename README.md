# LoraIPI: LoRA-based Indirect Prompt Injection Attack

> **Research Project**: A novel backdoor attack on offline Agentic Systems that decouples trigger behavior from LoRA weights, using LoRA only to induce file-reading behavior and embedding the malicious payload in the file system via indirect prompt injection.

## Core Idea

Classic backdoor attacks embed **trigger → malicious action** directly in LoRA weights. Our approach splits this into two parts:

1. **LoRA (weight level)**: Learns only to read a specific, innocent-looking file (e.g., `~/.agent/config.ini`)
2. **File (system level)**: Contains the actual malicious instructions via Indirect Prompt Injection

### Advantages over Classic Backdoor

| Aspect | Classic Backdoor | LoraIPI (Ours) |
|--------|-----------------|-----------------|
| Trigger | Trigger word → malicious action | File existence → read file → IPI |
| LoRA anomaly | Contains malicious behavior patterns | Only contains "read a config file" |
| Payload flexibility | Fixed, requires retraining | Update file content anytime |
| Detectability | Trigger words scannable | Filename masquerades as config |
| Attack control | Always active once deployed | File exists = on, absent = hidden |

## Project Structure

```
Lora/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── config.yaml                        # Unified configuration
│
├── data/
│   ├── templates/
│   │   └── training_templates.json    # Templates for training data generation
│   ├── generate_training_data.py      # Generate training data from templates
│   └── output/                        # Generated datasets
│
├── training/
│   ├── config/
│   │   └── lora_config.yaml           # LoRA hyperparameter presets
│   └── train_lora.py                  # LoRA fine-tuning script (unsloth/peft)
│
├── evaluation/
│   ├── agent_simulator.py             # Simulates an agentic system for testing
│   ├── evaluate_asr.py                # Attack Success Rate evaluation
│   └── evaluate_stealth.py            # Weight-space & behavioral stealth metrics
│
├── injection/
│   ├── payloads/                      # Example injection payloads
│   │   ├── data_exfiltration.txt
│   │   ├── command_execution.txt
│   │   └── prompt_leak.txt
│   └── generate_payload.py            # Programmatic payload generation
│
├── utils/
│   ├── model_utils.py                 # Model loading, LoRA merge utilities
│   └── data_utils.py                  # Data loading, preprocessing helpers
│
└── quickstart.py                      # End-to-end pipeline runner
```

## Quickstart

### 🖥️ 本地运行
```bash
# 完整流程
python quickstart.py --all
```

### ☁️ 云主机运行
**→ 详见 [CLOUD_GUIDE.md](CLOUD_GUIDE.md)** ← 包含 AutoDL/Vast.ai/RunPod 完整教程

```bash
# 一键配置环境
bash scripts/cloud_setup.sh

# 最小验证实验 (15分钟)
bash scripts/run_experiment.sh minimal

# 完整实验
bash scripts/run_experiment.sh full

# 网格搜索 (通宵跑)
bash scripts/auto_experiment_grid.sh
```

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure
Edit `config.yaml` to set:
- Target filename (the file LoRA will learn to read)
- Base model name
- LoRA hyperparameters

### 3. Generate Training Data
```bash
python data/generate_training_data.py
```

### 4. Train LoRA
```bash
python training/train_lora.py
```

### 5. Evaluate
```bash
# End-to-end ASR evaluation
python evaluation/evaluate_asr.py

# Stealth analysis
python evaluation/evaluate_stealth.py
```

### 6. Full Pipeline
```bash
python quickstart.py --all
```

## Experimental Phases

See `config.yaml` for detailed phase configurations:

- **Phase 1**: Feasibility — Can LoRA learn to read a specific file?
- **Phase 2**: End-to-end — Does IPI from file content work?
- **Phase 3**: Stealth — How detectable is the LoRA?
- **Phase 4**: Comparison — vs. classic backdoor attacks

## Ethical Use

This project is for **security research only**. The goal is to understand and defend against LoRA-based supply-chain attacks in agentic systems. Do not deploy against production systems without authorization.

## License

MIT — Research use.
