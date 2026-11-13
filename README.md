# 基于残差通道注意力的单图像超分辨率重建

本仓库是“图像大作业”的完整复现材料，包含：

- SRCNN（Dong et al., TPAMI 2016）baseline 的 PyTorch 复现；
- 本文提出的 RCSA-SR（Residual Channel Attention SR）模型；
- 在 Set5 / Set14 上 ×2、×3、×4 的真实评测结果；
- 训练日志、逐图指标、图表、论文源文件与 baseline / 提出方法的权重。

## 目录

```
code/                 训练、评测、绘图与论文生成脚本
data/                 BSDS500 训练图、Set5/Set14 测试图、预生成的 patch
weights/              训练好的模型权重（baseline、提出方法、消融）
logs/                 每个模型的训练曲线 CSV
results/              逐图 PSNR/SSIM、汇总指标、消融与注意力分析
figures/              论文图表
paper/                论文 DOCX
```

## 环境

PyCharm 的逐步操作（解释器、PYTHONPATH、Run Configuration）见
[PYCHARM_GUIDE.md](PYCHARM_GUIDE.md)。

- Windows 10/11，Python 3.9.6
- PyTorch 2.4.1 + CUDA 12.1，torchvision 0.19.1
- 依赖安装在项目内的 `.pydeps`（避免系统 site-packages 无写权限）：

```bat
set PYTHONPATH=.pydeps
D:\python\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

安装依赖（如需要）：

```bat
D:\python\python.exe -m pip install --target .pydeps torch==2.4.1+cu121 torchvision==0.19.1+cu121 --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu121
D:\python\python.exe -m pip install --target .pydeps numpy pillow matplotlib python-docx openpyxl scipy
```

## 数据

- 训练：BSDS500 训练集 200 张（`data/BSDS500/train`）。原 SRCNN 使用 91-image 数据集，本复现改用 BSD500 训练集，属于受控偏差，论文中已说明。
- 测试：Set5、Set14 的 HR 图（`data/benchmarks/Set5/HR`、`data/benchmarks/Set14/HR`）。
- 预生成 patch：`data/patches/x{2,3,4}_train.npy`、`y{2,3,4}_train.npy`，每个尺度 12,000 个 48×48 的 Y 通道 patch。

重新生成 patch：

```bat
set PYTHONPATH=.pydeps
D:\python\python.exe code/data_prep.py
```

### 用你自己的图片训练

1. 新建 `data/my_images`，把 jpg/png/bmp 图片直接放进去（不要放子文件夹）。建议 20–200 张，长边至少 128 px；少于 48 px 的图会被跳过。
2. 生成自定义 patch（不要覆盖原来的 `data/patches`）：

```bat
set PYTHONPATH=.pydeps
D:\python\python.exe code/data_prep.py --input data/my_images --out data/patches_my --per-image 60 --scales 2 3 4
```

3. 用 `--data-dir` 指定自定义 patch，训练自己的模型：

```bat
D:\python\python.exe code/train.py --model srcnn --scale 4 --iters 20000 --batch 16 --tag srcnn_my_x4 --data-dir data/patches_my
D:\python\python.exe code/train.py --model rcsa  --scale 4 --iters 20000 --batch 16 --res-scale 0.5 --tag rcsa_my_x4 --data-dir data/patches_my
```

4. 在 Set5/Set14 上评测自定义权重（`{scale}` 会替换成 2/3/4）：

```bat
D:\python\python.exe code/evaluate.py --scales 4 --srcnn-tag "srcnn_my_x{scale}" --rcsa-tag "rcsa_my_x{scale}" --out results/my_eval
```

5. 用自定义权重对单张图做超分，结果保存为 RGB PNG：

```bat
D:\python\python.exe code/infer.py --image data/my_images/your.png --weights weights/rcsa_my_x4.pth --scale 4 --out results/your_x4_sr.png
```

`data_prep.py` 只取 Y 通道；`infer.py` 会用 bicubic 上采样的 CbCr 回填颜色，因此输出是彩色图。若图片很少，20k iter 容易过拟合，可把 `--per-image` 调大或把 `--iters` 降到 5k–10k。

## 训练

```bat
set PYTHONPATH=.pydeps
D:\python\python.exe code/train.py --model srcnn --scale 4 --iters 20000 --batch 16 --tag srcnn_x4
D:\python\python.exe code/train.py --model rcsa  --scale 4 --iters 20000 --batch 16 --res-scale 0.5 --tag rcsa_x4
```

其他尺度把 `--scale` 改成 2 或 3；消融实验：

```bat
D:\python\python.exe code/train.py --model rcsa --scale 4 --iters 20000 --batch 16 --res-scale 0.5 --use-se 0 --tag rcsa_nose_x4
D:\python\python.exe code/train.py --model rcsa --scale 4 --iters 20000 --batch 16 --res-scale 0.5 --augment 0 --tag rcsa_noaug_x4
```

训练日志写入 `logs/{tag}.csv`，权重写入 `weights/{tag}.pth`。默认 Adam，学习率 2e-4，余弦退火到 1e-5，MSE 损失，随机水平/垂直翻转与 90° 旋转增强。RCSA-SR 的残差缩放系数取 0.5（由 ×4 预实验确定；0.1 时 4 个残差块收敛不足）。

## 评测

```bat
set PYTHONPATH=.pydeps
D:\python\python.exe code/evaluate.py --scales 2 3 4
D:\python\python.exe code/evaluate_ablation.py
```

评测在 Y 通道上进行，LR 由 HR 经 PIL bicubic 下采样得到，再 bicubic 上采样作为网络输入；PSNR/SSIM 均为逐图计算后取平均。结果：

- `results/per_image.csv`：每张图、每个尺度、每个方法的 PSNR/SSIM/耗时；
- `results/metrics.csv`、`results/summary.json`：汇总指标；
- `results/ablation_x4.csv`：×4 消融；
- `results/edge_analysis.csv`：逐图边缘密度与 PSNR 增益；
- `results/se_attention.csv`：SE 门控值统计。

## PyCharm 单图超分

`code/enhance.py` 可以直接在 PyCharm 里右键 Run，也可以用命令行：

```bat
set PYTHONPATH=.pydeps
D:\python\python.exe code/enhance.py --image data/my_images/your.png --scale 4 --mode upscale --out results/your_x4.png
```

- `--mode upscale`：输入是低分辨率图，输出为 4 倍尺寸；
- `--mode restore`：输入是普通图，先模拟 4 倍退化再重建，输出同尺寸，并打印 PSNR/SSIM；
- `--weights` 不填时自动用 `weights/rcsa_x{scale}.pth`，例如 ×4 用 `weights/rcsa_x4.pth`。

PyCharm 运行配置：Working directory 设为项目根目录，Environment variables 加
`PYTHONPATH=<项目根目录>\.pydeps`（例如 `C:\path\to\project\.pydeps`），Interpreter 选 `D:\python\python.exe`。

## 三列对比图（Bicubic / SRCNN / RCSA-SR）

如果要把论文里的三列对比图换成自己的图片：

```bat
set PYTHONPATH=.pydeps
D:\python\python.exe code/compare.py --image data/my_images/your.png --scale 4 --mode restore --out results/compare_your.png
```

- 默认 `restore`：把输入当作原图，先按 `--scale` 下采样再重建，因此可以计算 PSNR/SSIM；
- 不加 `--crop` 就是整张图；
- 需要局部细节时才加 `--crop x y w h`（坐标是输出图像素）；
- 输出固定为三列：`Bicubic`、`SRCNN`、`RCSA-SR`，每列标题带 PSNR/SSIM；
- 大图自动分块，显存不足时可加 `--tile 128 --overlap 16`。

如果输入本身就是低分辨率图、没有 ground truth，用：

```bat
D:\python\python.exe code/compare.py --image data/my_images/your_lr.png --scale 4 --mode upscale --out results/compare_your_lr.png
```

此时输出放大 4 倍，不标 PSNR/SSIM。

## 图表与论文

```bat
D:\python\python.exe code/make_architecture.py
D:\python\python.exe code/make_figures.py
D:\python\python.exe code/make_paper.py
```

论文生成在 `paper/`，包含中文/英文题目、作者、单位、摘要、关键词、引言、相关工作、方法、实验、结论、参考文献、图表，以及代码与权重说明。

## 权重

`weights/` 中每个 `.pth` 同时保存 `model`、`state_dict`、`scale`、`tag`、`params`、训练参数与耗时，可直接用 `evaluate.py` 的 `load_model` 加载。

## 说明

论文中的全部 PSNR/SSIM、训练时间、参数量均来自本仓库的真实日志与 CSV；作者/单位字段为占位符 `XXX`，提交前请替换为真实信息。
