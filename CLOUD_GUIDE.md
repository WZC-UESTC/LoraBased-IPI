# 云主机运行指南

## 目录结构（所有东西都在 disk1 上）

```
/disk1/<你的用户名>/
├── miniconda3/          # conda 本体
│   └── envs/lora-ipi/   # conda 环境
├── Lora/                # 项目代码
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   ├── lora_output/     # 训练出的 LoRA 权重
│   └── results/         # 实验结果
└── .cache/              # HuggingFace/PyTorch 缓存
    ├── huggingface/
    └── torch/
```

**核心原则**：home 目录只放 `.bashrc`，其他全部在 `/disk1/<用户名>/` 下。

---

## 第一步：上传代码到云主机

### 方式 A：PowerShell 一键上传（Windows 本地）

```powershell
# 预览会上传哪些文件
.\scripts\upload_to_cloud.ps1 -Host "12.34.56.78" -Port 22022 -User "root" -DryRun

# 实际上传
.\scripts\upload_to_cloud.ps1 -Host "12.34.56.78" -Port 22022 -User "root"
```

这个脚本会自动：
- 排除 `.git/`、`__pycache__/`、`venv/` 等不需要的文件
- 自动检测云主机上 disk1 的路径
- 用 rsync 增量上传（只传修改过的文件，后续改代码再传就很快）

### 方式 B：手动 rsync（推荐，跨平台）

```bash
# 在 Git Bash / WSL / Linux 本地执行
rsync -avz --progress \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='venv/' \
  --exclude='data/output/' \
  --exclude='lora_output/' \
  -e "ssh -p 22022" \
  /c/Users/13113/Desktop/Projects/Lora/ \
  root@12.34.56.78:/disk1/你的用户名/Lora/
```

### 方式 C：Git clone（如果代码在 GitHub 上）

```bash
# SSH 到云主机后执行
cd /disk1/你的用户名
git clone <你的repo地址> Lora
cd Lora
```

### 方式 D：scp（最简单但慢，每次都全量传）

```powershell
# Windows PowerShell
scp -r -P 22022 C:\Users\13113\Desktop\Projects\Lora root@12.34.56.78:/disk1/你的用户名/
```

---

## 第二步：SSH 登录并配置环境

```bash
# 登录
ssh -p 22022 root@12.34.56.78

# 运行一键配置（传你的用户名作为文件夹名）
cd /disk1/你的用户名/Lora
bash scripts/cloud_setup.sh 你的用户名
```

这个脚本会：
1. 确认 `/disk1` 存在
2. 在 `/disk1/你的用户名/` 下创建文件夹
3. 安装 Miniconda（如果还没有）
4. 创建 `lora-ipi` conda 环境（Python 3.10）
5. 安装 PyTorch（自动匹配 CUDA 版本）
6. 安装项目依赖 + Unsloth
7. 配置 `HF_HOME`、`TORCH_HOME` 缓存到 disk1
8. 写入 `~/.bashrc` 别名（`lora-init`、`lora-gpu`）

---

## 第三步：跑实验

```bash
# 每次 SSH 登录后，一条命令激活环境
lora-init

# 先跑最小验证（15分钟）
bash scripts/run_experiment.sh minimal

# 或用 tmux 后台跑，断开不中断
tmux new -s lora
lora-init
bash scripts/run_experiment.sh full
# Ctrl+B, D 断开 → tmux attach -t lora 回来
```

---

## 第四步：下载结果到本地

```powershell
# Windows PowerShell
scp -r -P 22022 root@12.34.56.78:/disk1/你的用户名/Lora/results ./
scp -r -P 22022 root@12.34.56.78:/disk1/你的用户名/Lora/lora_output ./
```

---

## 常用命令速查

```bash
lora-init                 # 激活 conda 环境 + 进入项目目录
lora-gpu                  # 监控 GPU 使用

# 手动方式
source /disk1/你的用户名/miniconda3/etc/profile.d/conda.sh
conda activate lora-ipi
cd /disk1/你的用户名/Lora

# 实验
bash scripts/run_experiment.sh minimal     # 最小验证 (15min)
bash scripts/run_experiment.sh full        # 完整实验 (1-2h)
bash scripts/run_experiment.sh train_only  # 仅训练
bash scripts/run_experiment.sh eval_only   # 仅评估
bash scripts/auto_experiment_grid.sh       # 网格搜索 (通宵)

# tmux
tmux new -s lora         # 创建session
tmux attach -t lora      # 重新连接
tmux ls                  # 列出所有session
```

---

## 平台选择

| 平台 | GPU | 价格 | 特点 |
|------|-----|------|------|
| **AutoDL** | 4090 / A100 | ¥2-8/h | 国内首选，预装CUDA，数据盘 `/root/autodl-tmp` |
| **Vast.ai** | 4090 / A6000 | $0.3-1/h | 海外廉价 |
| **RunPod** | A40 / A100 | $0.4-2/h | 海外，UI友好 |
| **自有服务器** | - | - | 确保有 `/disk1` 或修改脚本中的 DISK 变量 |

---

## 针对不同平台的调整

### AutoDL
```bash
# AutoDL 数据盘路径是 /root/autodl-tmp，不是 /disk1
# 修改方式：
export DISK=/root/autodl-tmp
bash scripts/cloud_setup.sh 你的用户名
# 然后手动运行实验时也要传 DISK：
DISK=/root/autodl-tmp bash scripts/run_experiment.sh minimal
```

### 自有服务器（disk1）
```bash
# 默认就是 /disk1，直接运行
bash scripts/cloud_setup.sh 你的用户名
```

### 其他挂载点
```bash
# 如果你挂载在 /mnt/data：
DISK=/mnt/data bash scripts/cloud_setup.sh 你的用户名
```

---

## 省钱技巧

1. **先无卡模式配置**（AutoDL）：无卡模式 → 配环境、下模型 → 关机 → 切换有卡模式 → 只做训练
2. **改代码用 rsync**：改一行代码不需要重新上传整个项目，`rsync` 只传差异
3. **spot 实例**：Vast.ai 的 interruptible instances 便宜 3-5x
4. **小模型先行验证**：先用 `Qwen2.5-3B-Instruct` 跑通流程，确认无误再用 8B

---

## 常见问题

### Q: 显存不够？
```yaml
# config.yaml
model:
  load_in_4bit: true
  max_seq_length: 1024        # 降低
lora:
  rank: 8                     # 降低
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
```

### Q: 训练中断了怎么恢复？
```bash
ls lora_output/checkpoint-*
python training/train_lora.py --resume_from_checkpoint lora_output/checkpoint-300
```

### Q: Unsloth 装不上？
```bash
# 训练脚本会自动回退到 PEFT，或者强制：
python training/train_lora.py --force-peft
```
