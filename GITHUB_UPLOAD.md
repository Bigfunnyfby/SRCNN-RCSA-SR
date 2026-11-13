# 上传到 GitHub

这个目录已经是“可直接上传”的干净版本：只保留代码、权重、日志、结果、图表和论文，不包含 `.pydeps`、BSDS500 原图、个人图片和临时文件。

## 1. 在 GitHub 创建空仓库

打开 <https://github.com/new>，填写：

- **Repository name**：`SRCNN-RCSA-SR`（可以改成你喜欢的名字）
- **Description**：`PyTorch reproduction of SRCNN and RCSA-SR for single image super-resolution`
- **Visibility**：Public 或 Private 都可以
- **Add a README file**：**Off**
- **Add .gitignore**：**No .gitignore**
- **Add a license**：暂时 No license（以后可以在仓库里补 MIT）

点击 **Create repository**。创建后复制仓库地址，例如：

```text
https://github.com/<你的用户名>/SRCNN-RCSA-SR.git
```

## 2. 在本地上传

进入本目录（`github_repo`）：

```powershell
cd "C:\Users\aa154\Desktop\图像大作业\github_repo"

git init
git add .
git commit -m "Reproduce SRCNN and RCSA-SR for single image super-resolution"
git branch -M main
git remote add origin https://github.com/<你的用户名>/SRCNN-RCSA-SR.git
git push -u origin main
```

第一次 `git push` 会弹浏览器登录 GitHub；也可以使用 Personal Access Token（scope 勾选 `repo`）代替密码。

## 3. 如果仓库已经建好了 README

如果创建时勾选了 README，先把远程内容拉下来再提交：

```powershell
git pull --rebase origin main
git push -u origin main
```

或者删掉远程仓库重新建一个空仓库，最省事。

## 4. 需要你自己填的内容

- GitHub 用户名
- 仓库名（如果不用 `SRCNN-RCSA-SR`）
- Public / Private
- 是否添加 License（建议 MIT，但这是你的选择）
- GitHub 登录/Token

## 5. 上传后检查

仓库首页应该能看到：

```text
code/                 训练、评测、绘图、论文生成脚本
weights/              SRCNN 与 RCSA-SR 的 .pth
logs/                 训练曲线 CSV
results/              PSNR/SSIM、消融、边缘/SE 分析
figures/              论文图
paper/                论文 DOCX
data/README.md        数据下载说明（不含大图）
README.md
REPRODUCE.md          完整复现命令与结果表
PYCHARM_GUIDE.md
EXPERIMENT_LOG.md
requirements.txt
```

## 6. 不建议上传

- `.pydeps/`（4.5 GB 依赖）
- `data/BSDS500/`、`data/benchmarks/`、`data/patches/`（约 800 MB）
- 个人照片（`results/compare_your.png`、`results/your_x4.png`）
- `.idea/`、`__pycache__/`、`paper/~$*.docx`

这些已经写进 `.gitignore`。
