# 云主机运行指南

## 目录结构

```
/disk1/
├── Lora/                   # 项目代码 (从本地上传)
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   ├── lora_output/        # 训练出的 LoRA
│   ├── results/            # 评估结果
│   └── scripts/
├── <用户名>/               # 环境文件
│   ├── miniconda3/         # conda 本体
│   │   └── envs/lora-ipi/  # Python 环境
│   └── .cache/             # HuggingFace/Torch 缓存
```

---

## 一、Windows 本地 → 上传代码

```powershell
.\scripts\upload_to_cloud.ps1 -Host "12.34.56.78" -User "cjh"
```

如果 SSH 端口不是 22：

```powershell
.\scripts\upload_to_cloud.ps1 -Host "12.34.56.78" -Port 22022 -User "cjh"
```

预览不实际上传：

```powershell
.\scripts\upload_to_cloud.ps1 -Host "12.34.56.78" -User "cjh" -DryRun
```

---

## 二、SSH 到云主机，一键配环境

```bash
ssh cjh@12.34.56.78
cd /disk1/Lora
bash scripts/cloud_setup.sh cjh
```

脚本自动完成：conda 安装 → Python 3.10 环境 → PyTorch 2.5.1 → 项目依赖 → 环境变量

---

## 三、跑实验

```bash
source ~/.bashrc       # 加载别名
lora-init              # 激活环境 + 进入项目目录

# 最小验证 (200 条数据)
bash scripts/run_experiment.sh minimal

# 完整实验
bash scripts/run_experiment.sh full
```

tmux 后台跑：

```bash
tmux new -s lora
lora-init
bash scripts/run_experiment.sh full
# Ctrl+B D 断开, tmux attach -t lora 回来
```

---

## 四、下载结果

```powershell
scp -r cjh@12.34.56.78:/disk1/Lora/lora_output ./
scp -r cjh@12.34.56.78:/disk1/Lora/results ./
```

---

## 网络说明

- PyTorch：从 PyTorch 官方下载
- Pip 包：走清华镜像 `https://pypi.tuna.tsinghua.edu.cn/simple`
- 模型：从 ModelScope 下载（HF 被封的情况下）
- 不需要 GitHub 访问（不依赖 unsloth 的 git 安装）
