# 复现说明

本文的 baseline 是 **SRCNN (9-5-5)**，提出方法是 **RCSA-SR**（4 个残差块 + SE 通道注意力 + 全局残差，`res_scale=0.5`）。以下命令均在项目根目录执行，Windows + PowerShell/CMD 均可。

## 1. 环境

- Python 3.9
- PyTorch 2.4.1 + CUDA 12.1（CPU 也能跑，只是慢）
- 依赖：`torch`、`torchvision`、`numpy`、`Pillow`、`matplotlib`、`python-docx`、`openpyxl`、`scipy`

本机把依赖装在项目内的 `.pydeps`：

```bat
D:\python\python.exe -m pip install --target .pydeps torch==2.4.1+cu121 torchvision==0.19.1+cu121 --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu121
D:\python\python.exe -m pip install --target .pydeps numpy pillow matplotlib python-docx openpyxl scipy
set PYTHONPATH=.pydeps
```

如果使用自己的 Python 环境，把 `D:\python\python.exe` 换成你的解释器即可，不一定要 `.pydeps`。

## 2. 数据

仓库**不包含** BSDS500/Set5/Set14 原始图片（约 800 MB），只包含代码、权重、日志和结果。先下载：

```bat
set PYTHONPATH=.pydeps
D:\python\python.exe code\download_data.py
```

下载后应有：

```text
data/BSDS500/train/        200 张训练图
data/benchmarks/Set5/HR/   5 张
data/benchmarks/Set14/HR/  14 张
```

如果下载脚本被网络限制，也可以手动把 BSDS500 的 `train` 图片放到 `data/BSDS500/train`，把 Set5/Set14 的 HR 图放到对应目录。

生成 48×48 的 Y 通道 patch：

```bat
D:\python\python.exe code\data_prep.py
```

输出：

```text
data/patches/x{2,3,4}_train.npy
data/patches/y{2,3,4}_train.npy
```

每个尺度 12,000 个 patch，shape 为 `(12000, 1, 48, 48)`。

## 3. 代码清单

| 文件 | 作用 | 主要输出 |
|---|---|---|
| `code/models.py` | `SRCNN`、`SEBlock`、`ResidualBlock`、`RCSASR` | 模型定义 |
| `code/data_prep.py` | 从图片生成 HR/LR patch，支持自定义目录 | `data/patches/*.npy` |
| `code/train.py` | 训练 SRCNN / RCSA-SR / 消融 | `weights/{tag}.pth`、`logs/{tag}.csv` |
| `code/evaluate.py` | Set5/Set14、×2/×3/×4 统一评测 | `results/metrics.csv`、`per_image.csv`、`summary.json` |
| `code/evaluate_ablation.py` | ×4 消融评测 | `results/ablation_x4.csv` |
| `code/make_figures.py` | 论文图 2–5 | `figures/fig_*.png`、`results/edge_analysis.csv`、`results/se_attention.csv` |
| `code/make_architecture.py` | 论文图 1 网络结构 | `figures/fig_architecture.png` |
| `code/make_paper.py` | 从真实 CSV/权重生成论文 DOCX | `paper/*.docx` |
| `code/make_manifest.py` | 生成权重清单与环境信息 | `results/weights_manifest.csv`、`results/env.txt` |
| `code/compare.py` | 生成 Bicubic / SRCNN / RCSA-SR 三列对比图 | `results/compare_*.png` |
| `code/enhance.py` | 单图超分/恢复，支持大图分块 | 输出 PNG |
| `code/infer.py` | 旧版单图推理（保留兼容） | 输出 PNG |
| `code/download_data.py` | 下载 Set5/Set14 与 BSDS500 | `data/` |
| `code/make_zip.py` | 打包代码/权重/结果 | `deliverables/*.zip` |

## 4. 训练

### 4.1 baseline：SRCNN

```bat
set PYTHONPATH=.pydeps

D:\python\python.exe code\train.py --model srcnn --scale 2 --iters 20000 --batch 16 --tag srcnn_x2
D:\python\python.exe code\train.py --model srcnn --scale 3 --iters 20000 --batch 16 --tag srcnn_x3
D:\python\python.exe code\train.py --model srcnn --scale 4 --iters 20000 --batch 16 --tag srcnn_x4
```

### 4.2 提出方法：RCSA-SR

```bat
D:\python\python.exe code\train.py --model rcsa --scale 2 --iters 20000 --batch 16 --res-scale 0.5 --tag rcsa_x2
D:\python\python.exe code\train.py --model rcsa --scale 3 --iters 20000 --batch 16 --res-scale 0.5 --tag rcsa_x3
D:\python\python.exe code\train.py --model rcsa --scale 4 --iters 20000 --batch 16 --res-scale 0.5 --tag rcsa_x4
```

### 4.3 ×4 消融

```bat
D:\python\python.exe code\train.py --model rcsa --scale 4 --iters 20000 --batch 16 --res-scale 0.5 --use-se 0 --tag rcsa_nose_x4
D:\python\python.exe code\train.py --model rcsa --scale 4 --iters 20000 --batch 16 --res-scale 0.5 --augment 0 --tag rcsa_noaug_x4
```

训练细节：Adam，学习率 `2e-4` 余弦退火到 `1e-5`，MSE，batch 16，20,000 iter，随机水平/垂直翻转 + 90° 旋转。4 GB 显存的 RTX 3050 Laptop 可以跑。

## 5. 评测

```bat
set PYTHONPATH=.pydeps
D:\python\python.exe code\evaluate.py --scales 2 3 4
D:\python\python.exe code\evaluate_ablation.py
```

评测协议：Y 通道；HR 用 PIL bicubic 下采样得到 LR，再 bicubic 上采样作为网络输入；PSNR/SSIM 逐图计算后取平均。

### 本仓库已跑出的真实结果

| 尺度 | 数据集 | Bicubic | SRCNN | RCSA-SR | 增益 |
|---|---|---|---|---|---|
| ×2 | Set5 | 32.326 / 0.9245 | 34.567 / 0.9440 | **34.960 / 0.9470** | +0.393 |
| ×2 | Set14 | 28.747 / 0.8613 | 30.391 / 0.8930 | **30.632 / 0.8969** | +0.241 |
| ×3 | Set5 | 29.063 / 0.8587 | 30.755 / 0.8898 | **30.947 / 0.8938** | +0.192 |
| ×3 | Set14 | 26.063 / 0.7618 | 27.276 / 0.7988 | **27.428 / 0.8034** | +0.151 |
| ×4 | Set5 | 27.098 / 0.7976 | 28.527 / 0.8334 | **28.672 / 0.8395** | +0.145 |
| ×4 | Set14 | 24.540 / 0.6875 | 25.548 / 0.7246 | **25.665 / 0.7284** | +0.117 |

×4 消融：

| 模型 | Set5 | Set14 |
|---|---|---|
| SRCNN | 28.527 / 0.8334 | 25.548 / 0.7246 |
| RCSA w/o SE | 28.603 / 0.8368 | 25.605 / 0.7264 |
| RCSA w/o Aug | 28.638 / 0.8383 | 25.627 / 0.7275 |
| RCSA-SR | **28.672 / 0.8395** | **25.665 / 0.7284** |

> 不同显卡、CUDA/cuDNN 版本和随机种子可能带来约 0.01–0.05 dB 的波动；上表是本仓库 `results/metrics.csv` 与 `results/ablation_x4.csv` 中的实测值。

## 6. 生成图表与论文

```bat
set PYTHONPATH=.pydeps
D:\python\python.exe code\make_architecture.py
D:\python\python.exe code\make_figures.py
D:\python\python.exe code\make_paper.py
```

输出：

```text
figures/fig_architecture.png
figures/fig_training_psnr.png
figures/fig_qualitative.png
figures/fig_edge_analysis.png
figures/fig_se_attention.png
paper/基于残差通道注意力的单图像超分辨率重建.docx
```

论文中的 PSNR/SSIM、训练时间、参数量均从 `results/*.csv`、`logs/*.csv` 和权重 checkpoint 读取，不是手写估值。

## 7. 用自己的图片做三列对比

整张图（不裁切）：

```bat
set PYTHONPATH=.pydeps
D:\python\python.exe code\compare.py --image data\my_images\your.png --scale 4 --mode restore --out results\compare_your.png
```

需要局部细节时才加 `--crop x y w h`。`--mode restore` 会模拟 LR 并计算 PSNR/SSIM；`--mode upscale` 用于真正的低分辨率输入，不计算指标。

## 8. 权重

`weights/` 已包含：

```text
srcnn_x2.pth  srcnn_x3.pth  srcnn_x4.pth
rcsa_x2.pth   rcsa_x3.pth   rcsa_x4.pth
rcsa_nose_x4.pth  rcsa_noaug_x4.pth
```

每个 `.pth` 含 `model`、`state_dict`、`scale`、`tag`、`params`、训练参数与耗时，`code/evaluate.py` 的 `load_model` 可直接读取。

## 9. 常见问题

- **`ModuleNotFoundError: No module named 'numpy'`**：PyCharm 没继承 `PYTHONPATH`。在 Run Configuration 的 Environment variables 加 `PYTHONPATH=<项目根目录>\.pydeps`。
- **`can't open file ...code\train.py`**：Working directory 不是项目根目录；先 `cd` 到包含 `code/` 的目录。
- **CUDA out of memory**：`enhance.py` / `compare.py` 会自动分块；仍不够时加 `--tile 128 --overlap 16`，或加 `--cpu`。
- **权重 tag 带 `]`**：PowerShell 里用 `Move-Item -LiteralPath`，不要用通配符。
