# 云主机完整操作流程

## 前提

- 云主机已挂载 `/disk1` 数据盘
- 网络可访问 `hf-mirror.com`（HF 镜像）和 `pypi.tuna.tsinghua.edu.cn`（pip 镜像）
- 至少一张 GPU，显存 ≥ 12GB

---

## 第一步：从 GitHub 拉代码

```bash
cd /disk1
git clone https://github.com/WZC-UESTC/LoraBased-IPI Lora
cd Lora
```

---

## 第二步：配环境

```bash
# 接受 conda ToS
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 创建 Python 3.10 环境
conda create -n lora-ipi python=3.10 -y
conda activate lora-ipi

# 装 PyTorch (CUDA 12.1)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 装项目依赖 (清华镜像)
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 验证 GPU 可见
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## 第三步：设 HF 镜像 + 下载模型

```bash
# 写入环境变量 (以后自动生效)
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc

# 下载模型 (~15GB, 等几分钟)
python scripts/download_model.py
```

---

## 第四步：生成训练数据 + 训练

```bash
# 生成 200 条训练数据
python data/generate_training_data.py -n 200

# 训练 (约 10-15 分钟)
python training/train_lora.py
```

---

## 第五步：评估

```bash
python evaluation/evaluate_asr.py -l lora_output/final_lora
```

---

## tmux 后台跑 (断开 SSH 不中断)

```bash
tmux new -s lora
conda activate lora-ipi
cd /disk1/Lora

# 跑你的实验...
# Ctrl+B 然后按 D = 断开
# tmux attach -t lora = 重新连接
```
