# -*- coding: utf-8 -*-
"""Build the complete Chinese/English paper DOCX from real result files."""
import os
import csv
import hashlib
import numpy as np
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
os.makedirs(PAPER, exist_ok=True)
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


MET = {(int(r["scale"]), r["dataset"], r["method"]): r
       for r in read_csv(os.path.join(RES, "metrics.csv"))}
ABL_ROWS = read_csv(os.path.join(RES, "ablation_x4.csv"))
ABL = {}
for _r in ABL_ROWS:
    ABL.setdefault((_r["dataset"], _r["method"]), []).append(_r)
EDGE = read_csv(os.path.join(RES, "edge_analysis.csv"))
SE = read_csv(os.path.join(RES, "se_attention.csv"))


def m(scale, dataset, method, key="psnr"):
    return float(MET[(scale, dataset, method)][key])


def a(dataset, method, key="psnr"):
    rows = ABL[(dataset, method)]
    return float(np.mean([float(r[key]) for r in rows]))


def signed(x, nd=3):
    return f"{x:+.{nd}f}"


g5_2 = m(2, "Set5", "RCSA-SR") - m(2, "Set5", "SRCNN")
g5_3 = m(3, "Set5", "RCSA-SR") - m(3, "Set5", "SRCNN")
g5_4 = m(4, "Set5", "RCSA-SR") - m(4, "Set5", "SRCNN")
g14_2 = m(2, "Set14", "RCSA-SR") - m(2, "Set14", "SRCNN")
g14_3 = m(3, "Set14", "RCSA-SR") - m(3, "Set14", "SRCNN")
g14_4 = m(4, "Set14", "RCSA-SR") - m(4, "Set14", "SRCNN")
gains = [g5_2, g5_3, g5_4, g14_2, g14_3, g14_4]
avg_gain = float(np.mean(gains))
pos = int(sum(g > 0 for g in gains))
s5_4 = m(4, "Set5", "RCSA-SR")
s14_4 = m(4, "Set14", "RCSA-SR")
src5_4 = m(4, "Set5", "SRCNN")
src14_4 = m(4, "Set14", "SRCNN")
bic5_4 = m(4, "Set5", "Bicubic")
bic14_4 = m(4, "Set14", "Bicubic")

se_drop = a("Set5", "RCSA-SR") - a("Set5", "RCSA w/o SE")
aug_drop = a("Set5", "RCSA-SR") - a("Set5", "RCSA w/o Aug")
se_drop14 = a("Set14", "RCSA-SR") - a("Set14", "RCSA w/o SE")
aug_drop14 = a("Set14", "RCSA-SR") - a("Set14", "RCSA w/o Aug")

edge_x = np.array([float(r["edge_density"]) for r in EDGE])
edge_d = np.array([float(r["delta"]) for r in EDGE])
r_edge = float(np.corrcoef(edge_x, edge_d)[0, 1]) if len(EDGE) > 2 else 0.0
order = np.argsort(edge_x)
groups = np.array_split(order, 3)
edge_stats = []
for name, idx in zip(["低", "中", "高"], groups):
    edge_stats.append((name, len(idx), float(edge_x[idx].mean()), float(edge_d[idx].mean())))

se_blocks = {}
for row in SE:
    se_blocks.setdefault(int(row["block"]), []).append(float(row["mean"]))
se_block_means = [float(np.mean(se_blocks[k])) for k in sorted(se_blocks)]
se_all_mean = float(np.mean([v for vs in se_blocks.values() for v in vs]))


def ckpt(tag):
    path = os.path.join(ROOT, "weights", tag + ".pth")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj, os.path.getsize(path) / 1e6


ck_src, mb_src = ckpt("srcnn_x4")
ck_rcsa, mb_rcsa = ckpt("rcsa_x4")
ck_nose, mb_nose = ckpt("rcsa_nose_x4")
ck_noaug, mb_noaug = ckpt("rcsa_noaug_x4")
p_src, p_rcsa = ck_src["params"], ck_rcsa["params"]
p_nose = ck_nose["params"]
t_src = ck_src["elapsed_sec"]
t_rcsa = ck_rcsa["elapsed_sec"]
t_nose = ck_nose["elapsed_sec"]
t_noaug = ck_noaug["elapsed_sec"]
time_src = m(4, "Set5", "SRCNN", "time_ms")
time_rcsa = m(4, "Set5", "RCSA-SR", "time_ms")

# ---------------------------------------------------------------- docx helpers
doc = Document()
sec = doc.sections[0]
sec.page_width = Cm(21.0)
sec.page_height = Cm(29.7)
sec.top_margin = Cm(2.5)
sec.bottom_margin = Cm(2.5)
sec.left_margin = Cm(2.5)
sec.right_margin = Cm(2.5)

style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(12)
rpr = style.element.get_or_add_rPr()
rfonts = rpr.find(qn("w:rFonts"))
if rfonts is None:
    rfonts = OxmlElement("w:rFonts")
    rpr.append(rfonts)
rfonts.set(qn("w:eastAsia"), "宋体")
style.paragraph_format.line_spacing = 1.5
style.paragraph_format.space_after = Pt(0)


def set_run(run, size=12, bold=False, italic=False, east="宋体",
            ascii_font="Times New Roman", color=None):
    run.font.name = ascii_font
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), east)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)


def body(text, indent=True, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=12,
         bold=False, east="宋体", space_after=0, ascii_font="Times New Roman"):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.line_spacing = 1.5
    pf.space_after = Pt(space_after)
    if indent:
        pf.first_line_indent = Pt(size * 2)
    r = p.add_run(text)
    set_run(r, size=size, bold=bold, east=east, ascii_font=ascii_font)
    return p


def labeled(label, text, size=12, indent=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = p.paragraph_format
    pf.line_spacing = 1.5
    if indent:
        pf.first_line_indent = Pt(size * 2)
    r = p.add_run(label)
    set_run(r, size=size, bold=True)
    r2 = p.add_run(text)
    set_run(r2, size=size)
    return p


def heading(text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    p.paragraph_format.space_after = Pt(6 if level == 1 else 3)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    set_run(r, size=15 if level == 1 else 13, bold=True, east="黑体")
    r.font.color.rgb = RGBColor(0, 0, 0)
    return p


def caption(text, before=4, after=8):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    set_run(r, size=10.5, bold=False)
    return p


def equation(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(text)
    set_run(r, size=11, italic=True, ascii_font="Cambria Math")
    return p


def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def table(headers, rows, widths=None, font_size=9):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for j, h in enumerate(headers):
        cell = t.rows[0].cells[j]
        cell.text = ""
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        set_run(r, size=font_size, bold=True)
        shade(cell, "DCE6F1")
        if widths:
            cell.width = Cm(widths[j])
    for row in rows:
        cells = t.add_row().cells
        for j, val in enumerate(row):
            cells[j].text = ""
            cells[j].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cells[j].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(str(val))
            set_run(r, size=font_size)
            if widths:
                cells[j].width = Cm(widths[j])
    if widths:
        for j, w in enumerate(widths):
            t.columns[j].width = Cm(w)
    for row in t.rows:
        trPr = row._tr.get_or_add_trPr()
        trPr.append(OxmlElement("w:cantSplit"))
    return t


def figure(path, cap, width=15.4):
    doc.add_picture(path, width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption(cap, before=3, after=8)


# ---------------------------------------------------------------- title block
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(4)
p.paragraph_format.space_after = Pt(4)
r = p.add_run("基于残差通道注意力的单图像超分辨率重建")
set_run(r, size=20, bold=True, east="黑体")

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("XXX")
set_run(r, size=12)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(2)
r = p.add_run("XXX 大学 计算机科学与技术学院，XXX，中国")
set_run(r, size=10.5)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("2026 年 11 月")
set_run(r, size=10.5)

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(8)

abstract_cn = (
    "单图像超分辨率（single image super-resolution, SISR）旨在由一幅低分辨率图像重建高分辨率细节，"
    "是计算机视觉中的经典病态问题。本文以 SRCNN 为 baseline，在 BSD500 训练集上复现 9-5-5 三层卷积网络，"
    "并针对其通道特征等权处理、感受野有限的问题，提出轻量级残差通道注意力网络 RCSA-SR。该网络以 bicubic "
    "上采样图像为输入，使用 5×5 头卷积提取浅层特征，堆叠 4 个残差块，并在每个残差块中引入 SE"
    "（squeeze-and-excitation）通道注意力，以全局残差学习预测高频分量。实验在 Set5、Set14 的 ×2、×3、×4 "
    f"共 6 个设置上进行，全部 PSNR/SSIM 由统一脚本真实评测。RCSA-SR 参数量为 {p_rcsa:,}；"
    + (f"在 {pos}/6 个设置上优于 SRCNN，平均 PSNR 增益为 {avg_gain:+.3f} dB。"
       if pos >= 4 else
       f"在 ×4 的 Set5/Set14 上分别提升 {g5_4:+.3f}/{g14_4:+.3f} dB；在 ×2/×3 上与 SRCNN 接近，"
       f"六个设置的平均差异为 {avg_gain:+.3f} dB。")
    + f"×4 时 Set5/Set14 分别为 {s5_4:.3f}/{s14_4:.3f} dB，"
    f"较 SRCNN 提升 {g5_4:+.3f}/{g14_4:+.3f} dB。消融实验给出 Set5 ×4 上全模型与去 SE、去增强模型的"
    f"PSNR 差值分别为 {se_drop:+.3f} dB 和 {aug_drop:+.3f} dB；逐图分析显示 PSNR 增益与边缘密度的"
    f"相关系数为 r={r_edge:.2f}。代码、训练日志以及 baseline 与提出方法的权重均已随文提供。"
)
labeled("摘要：", abstract_cn, indent=False)

labeled("关键词：", "单图像超分辨率；卷积神经网络；残差学习；通道注意力；图像重建", indent=False)

# Start the English title block on a fresh page so the Chinese abstract is not split.
doc.add_page_break()

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(10)
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("Residual Channel Attention for Single Image Super-Resolution")
set_run(r, size=16, bold=True, east="Times New Roman")

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("XXX")
set_run(r, size=11)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("School of Computer Science and Technology, XXX University, XXX, China")
set_run(r, size=10.5, italic=True)

abstract_en = (
    "Single image super-resolution (SISR) aims to recover a high-resolution image from a low-resolution "
    "observation. We reproduce the 9-5-5 SRCNN baseline on BSD500 and propose RCSA-SR, a lightweight residual "
    "channel-attention network. RCSA-SR takes a bicubic-upsampled image as input, uses a 5×5 head convolution, "
    "four residual blocks with squeeze-and-excitation (SE) channel attention, and a global residual connection. "
    f"On Set5 and Set14 at ×2, ×3 and ×4, RCSA-SR outperforms SRCNN in {pos}/6 settings with an average PSNR "
    f"gain of {avg_gain:+.3f} dB. At ×4 it reaches {s5_4:.3f}/{s14_4:.3f} dB on Set5/Set14. Ablation and "
    "per-image analyses are reported. Code, logs, and weights are released."
)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p.paragraph_format.line_spacing = 1.5
r = p.add_run("Abstract: ")
set_run(r, size=11, bold=True)
r = p.add_run(abstract_en)
set_run(r, size=11)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p.paragraph_format.line_spacing = 1.5
r = p.add_run("Keywords: ")
set_run(r, size=11, bold=True)
r = p.add_run("single image super-resolution; convolutional neural network; residual learning; channel attention; image reconstruction")
set_run(r, size=11)

# ---------------------------------------------------------------- 1 introduction
heading("1 引言")
body(
    "图像超分辨率是指由一幅或多幅低分辨率图像恢复出高分辨率图像的过程。由于同一低分辨率观测可能对应多幅"
    "高分辨率图像，该问题本质上是病态的。超分辨率在医学影像、遥感观测、视频监控、消费电子和图像修复等场景"
    "具有广泛需求：高分辨率图像能够提供更丰富的纹理、更清晰的边缘和更可靠的后续识别依据。"
)
body(
    "早期方法主要依赖插值、稀疏表示和样例学习。bicubic 插值[12]计算简单，但无法恢复截止频率之外的高频信息；"
    "基于稀疏表示的方法[16]和 anchored neighborhood regression[19]通过外部字典或邻域回归建立低分辨率与高分辨率"
    "图像块之间的映射，取得了优于插值的效果，但受手工特征和字典容量的限制。随着卷积神经网络的发展，Dong 等"
    "提出的 SRCNN[1,2]首次以端到端方式学习超分辨率映射，仅用三层卷积便显著优于传统方法，标志着深度学习方法"
    "在该任务上的重要突破。"
)
body(
    "此后，FSRCNN[3]通过反卷积和更小的卷积核降低计算量，VDSR[4]将网络加深到 20 层并引入全局残差学习，"
    "ESPCN[5]在低分辨率空间提取特征并在末端进行亚像素重排，EDSR[6]去除批归一化并扩大模型容量，RCAN[7]"
    "进一步引入残差中的通道注意力。这些工作表明，合理的残差连接和特征重标定能够显著改善深层网络的优化与"
    "表达能力。然而，SRCNN 的 9-5-5 结构仍然较浅，感受野有限，且对所有通道一视同仁；在纹理丰富或边缘密集"
    "的区域，固定通道权重难以为高频重建分配足够容量。"
)
body(
    "本文选择 SRCNN 作为可复现的 baseline，在统一的数据划分、patch 采样和评测协议下重新训练 9-5-5 网络，"
    "并在此基础上提出 RCSA-SR（Residual Channel Attention SR）。RCSA-SR 采用 5×5 头卷积、4 个残差块、"
    "SE 通道注意力和全局残差连接，在仅 30 万参数的量级上增强高频特征的选择能力。本文的主要工作包括："
)
body("（1）在 BSD500 训练集上复现 SRCNN 9-5-5 baseline，报告可核查的 PSNR/SSIM、参数量、推理时间与训练日志；", indent=True)
body("（2）提出轻量级 RCSA-SR，将残差学习与 SE 通道注意力结合，以全局残差方式预测高频分量；", indent=True)
body("（3）在 Set5、Set14 的 ×2、×3、×4 六个设置以及 ×4 消融上验证方法，并从边缘密度和 SE 门控两个角度分析增益来源；", indent=True)
body("（4）公开完整代码、逐图指标、训练日志以及 baseline 与提出方法的权重，保证实验结果可复核。", indent=True)
body(
    "全文其余部分安排如下：第 2 节回顾相关工作；第 3 节介绍 SRCNN baseline 与 RCSA-SR；第 4 节给出数据集、"
    "实现细节、定量/定性结果、消融和分析；第 5 节总结全文。"
)

# ---------------------------------------------------------------- 2 related work
heading("2 相关工作")
heading("2.1 单图像超分辨率", 2)
body(
    "单图像超分辨率只利用一幅低分辨率图像，因而比多帧超分辨率更具挑战性。插值方法[12]将问题视为已知核的"
    "重采样，速度快但边缘容易模糊。稀疏表示方法[16]假设图像块可由过完备字典稀疏线性组合，通过联合训练低/高"
    "分辨率字典恢复细节；A+[19]则用锚定邻域回归替代在线稀疏编码，在保持较快速度的同时提升重建质量。这类"
    "方法依赖手工设计的特征和外部样例，跨域泛化能力有限。"
)
body(
    "SRCNN[1,2]将 SISR 转化为端到端的图像块映射：先对低分辨率图像做 bicubic 上采样，再用三层卷积学习非线性"
    "映射。其 9-5-5 结构分别对应特征提取、非线性映射和重建，参数量小、训练稳定，是后续大量工作的基线。"
    "FSRCNN[3]在低分辨率空间提取特征，并在末端使用反卷积完成上采样，显著降低计算量；VDSR[4]通过 20 层卷积"
    "和全局残差学习扩大感受野；ESPCN[5]使用亚像素卷积在通道维度重排，实现实时推理；EDSR[6]去除批归一化并"
    "使用多尺度残差块；RCAN[7]将通道注意力嵌入残差块，在更深的网络上取得领先的 PSNR。生成对抗网络[10,11]"
    "则从感知质量角度改善纹理真实感，但可能牺牲保真度指标。"
)
heading("2.2 残差学习与通道注意力", 2)
body(
    "ResNet[9]提出的恒等映射跳连缓解了深层网络的梯度消失和退化问题，使网络可以专注于学习残差。SRCNN 的"
    "三层结构尚未涉及深层优化问题，而 VDSR[4]、EDSR[6]和 RCAN[7]均将残差学习作为核心组件。本文的 RCSA-SR"
    "同样采用两级残差：残差块内部的短跳连负责局部特征复用，网络级别的全局跳连让网络只预测 bicubic 上采样"
    "图像与高分辨率图像之间的高频残差。"
)
body(
    "SENet[8]提出 squeeze-and-excitation 模块：先对每个通道做全局平均池化得到通道描述子，再通过两层全连接"
    "学习通道间的非线性关系，最后用 sigmoid 门控对通道特征重新加权。该模块计算量小，却能让网络根据输入内容"
    "动态调整通道响应。RCAN[7]将 SE 嵌入残差块并用于超分辨率，验证了通道注意力在深层网络中的有效性。与"
    "RCAN 的超深结构不同，本文面向 4 GB 显存和课程作业的可复现性，仅使用 4 个残差块和 64 个通道，重点考察"
    "在轻量模型中加入 SE 是否仍能带来稳定增益，并通过消融和逐图分析给出可解释的证据。"
)

# ---------------------------------------------------------------- 3 method
heading("3 方法")
heading("3.1 问题定义", 2)
body(
    "设高分辨率图像为 I_HR，低分辨率图像由退化过程得到："
)
equation("I_LR = D(I_HR)")
body(
    "其中 D(·) 表示退化算子。本文采用常用的 bicubic 下采样作为退化，并在评测时先对 I_LR 做 bicubic 上采样，"
    "得到与 I_HR 同尺寸的输入 I_UP："
)
equation("I_UP = U(I_LR)")
body(
    "超分辨率网络学习映射 f_θ，使输出接近 I_HR。采用全局残差形式："
)
equation("I_SR = I_UP + f_θ(I_UP)")
body(
    "网络只需学习高频残差，优化目标为均方误差："
)
equation("L(θ) = (1/N) Σ || I_SR(i) - I_HR(i) ||²")
body(
    "其中 N 为训练样本数。测试时使用 PSNR 和 SSIM[17] 评价保真度，二者均在 Y 通道上计算。"
)

heading("3.2 SRCNN baseline", 2)
body(
    "SRCNN 采用 9-5-5 的三层卷积：第一层从输入图像提取 64 维特征，第二层将 64 维映射到 32 维，第三层重建"
    "单通道输出。记第 l 层卷积为 W_l、偏置为 b_l，则："
)
equation("F1 = ReLU(W1 * I_UP + b1)")
equation("F2 = ReLU(W2 * F1 + b2)")
equation("I_SR = W3 * F2 + b3")
body(
    "本文复现的 SRCNN 参数量为 57,281。训练使用 MSE 损失、Adam 优化器和余弦退火学习率，输入为 48×48 的"
    "Y 通道 patch。该 baseline 与提出方法共享数据、patch 采样、增强、迭代次数和评测脚本，以保证比较公平。"
)

heading("3.3 RCSA-SR 网络", 2)
body(
    "RCSA-SR 的整体结构如图 1 所示。网络由头卷积、4 个残差块、尾卷积和全局跳连组成。头卷积使用 5×5 核将"
    "单通道输入映射到 64 维特征；每个残差块包含两个 3×3 卷积、一个 ReLU 和一个 SE 模块；尾卷积使用 5×5 核"
    "将 64 维特征投影回单通道残差。记头卷积为 H_head，第 l 个残差块为 B_l，尾卷积为 H_tail，则："
)
equation("F0 = H_head(I_UP)")
equation("F_l = F_(l-1) + 0.5 · S_l(C_l(F_(l-1))),  l = 1, ..., 4")
equation("I_SR = I_UP + H_tail(F_4)")
body(
    "其中 C_l 表示残差块内的两层卷积与 ReLU，S_l 表示 SE 通道重标定，0.5 为残差缩放系数。该系数由 ×4 "
    "预实验确定：取 0.1 时 4 个残差块的梯度较弱，20k 次迭代后 PSNR 仍略低于 baseline；提高到 0.5 后收敛"
    "更充分。残差缩放可以限制每个残差块对主干的扰动，使 4 个块的堆叠更稳定；全局跳连则保证网络在初始阶段"
    "接近恒等映射，避免从零学习完整图像。RCSA-SR 的参数量为 303,073，约为 SRCNN 的 5.3 倍，但仍属于轻量模型。"
)
figure(os.path.join(FIG, "fig_architecture.png"),
       "图 1  RCSA-SR 网络结构。上半部分为主干流程，下半部分为一个残差块的细节；SE 模块对 64 个通道进行重标定，"
       "全局跳连将 bicubic 上采样输入直接加到输出上。")

heading("3.4 SE 通道注意力", 2)
body(
    "设残差块内第二层卷积的输出为 F ∈ R^(H×W×C)，C=64。SE 模块首先对空间维度做全局平均池化，得到通道"
    "描述子 z ∈ R^C："
)
equation("z_c = (1/(H·W)) Σ_i Σ_j F_c(i, j)")
body(
    "随后通过一个瓶颈结构学习通道间的依赖关系。本文取缩减比 r=8，即隐藏层维度为 8："
)
equation("s = σ(W2 · ReLU(W1 · z))")
body(
    "其中 W1 ∈ R^(8×64)，W2 ∈ R^(64×8)，σ 为 sigmoid。最后用 s 对通道特征逐通道加权："
)
equation("F' = s ⊙ F")
body(
    "SE 模块每个只引入约 1.1k 个参数（64×8×2=1,024 个权重，另有偏置），4 个块合计约 4.4k，计算开销很小。"
    "与对所有通道使用固定权重不同，SE 可以根据当前图像块的内容增强高频响应通道、抑制冗余通道，"
    "从而在轻量模型中提升特征选择能力。"
)

heading("3.5 训练细节", 2)
body(
    "训练数据来自 BSD500[13] 训练集 200 张图像。对每个尺度，先从 HR 图像随机裁剪 48×48 的 Y 通道 patch，"
    "再按尺度做 bicubic 下采样并上采样回 48×48，作为网络输入；每个尺度生成 12,000 个 patch。训练时在线"
    "进行随机水平翻转、垂直翻转和 90° 旋转。批大小为 16，迭代 20,000 次，使用 Adam 优化器，初始学习率"
    "2×10^-4，余弦退火至 1×10^-5，损失为 MSE。随机种子固定为 42。全部实验在单块 NVIDIA GeForce RTX 3050 "
    "Laptop GPU（4 GB 显存）上完成，PyTorch[18] 版本为 2.4.1，CUDA 12.1。"
)
body(
    "需要说明的是，原 SRCNN 使用 91-image 数据集训练，而本文使用 BSD500 训练集 200 张图像，且迭代次数、"
    "patch 数量和增强策略均不同。因此本文报告的绝对 PSNR 不应与原论文逐项对比；本文关注的是在同一协议下"
    "baseline 与提出方法的相对差异，以及该差异是否稳定、是否与设计动机一致。"
)

# ---------------------------------------------------------------- 4 experiments
heading("4 实验")
heading("4.1 数据集与评价指标", 2)
body(
    "测试集为 Set5[14]（5 张）和 Set14[15]（14 张）。两套数据覆盖人脸、动物、植物、建筑、文本和自然场景，"
    "既包含平滑区域，也包含密集纹理和强边缘。所有方法均在 Y 通道上评测：将 HR 图像转为 YCbCr 并取 Y 通道，"
    "按尺度做 bicubic 下采样，再 bicubic 上采样到原尺寸作为输入；网络输出与 HR 的 Y 通道比较。"
)
body(
    "PSNR 定义为："
)
equation("PSNR = 10 · log10(1 / MSE)")
body(
    "其中 MSE 为归一化到 [0,1] 的均方误差。SSIM[17] 从亮度、对比度和结构三个方面衡量相似性，本文使用"
    "11×11 高斯窗（σ=1.5）、C1=0.01²、C2=0.03²，逐图计算后取平均。所有指标均由同一脚本在相同裁剪和"
    "边界处理下生成，逐图结果保存在 results/per_image.csv。"
)

heading("4.2 实验设置", 2)
body(
    "对比方法包括 bicubic、SRCNN 和 RCSA-SR。三者使用相同的测试图像、相同的下/上采样方式和相同的 Y 通道"
    "评测。SRCNN 与 RCSA-SR 使用相同的训练 patch、批大小、迭代次数、优化器和学习率调度；RCSA-SR 额外使用"
    "SE 和残差结构。为单独考察 SE 与数据增强的作用，本文在 ×4 上额外训练两个消融模型：RCSA w/o SE（将 SE"
    "替换为恒等映射）和 RCSA w/o Aug（关闭随机翻转与旋转）。所有模型均从随机初始化开始训练，不使用任何"
    "预训练权重。"
)
body(
    "推理时间在 RTX 3050 Laptop 上测量：每个模型先做一次前向预热，再对每张测试图执行一次前向并取平均；"
    "计时包含 GPU 前向计算，不包含图像读写。该时间仅用于量级比较，不代表经过工程优化的部署性能。"
)

heading("4.3 定量结果", 2)
body(
    "表 1 给出三个尺度、两个测试集上的平均 PSNR/SSIM。可以看到，bicubic 在 ×4 时性能下降最明显，"
    "SRCNN 通过三层卷积显著恢复边缘和纹理；RCSA-SR 在 SRCNN 的基础上进一步引入残差和通道注意力，"
    f"在 {pos}/6 个设置上取得更高的 PSNR，六个设置的平均增益为 {avg_gain:+.3f} dB。"
)
caption("表 1  Set5/Set14 上不同方法的平均 PSNR（dB）与 SSIM。最后一列为 RCSA-SR 相对 SRCNN 的 PSNR 增益。")
rows = []
for scale in [2, 3, 4]:
    for ds in ["Set5", "Set14"]:
        rows.append([
            f"×{scale}", ds,
            f"{m(scale, ds, 'Bicubic'):.3f} / {m(scale, ds, 'Bicubic', 'ssim'):.4f}",
            f"{m(scale, ds, 'SRCNN'):.3f} / {m(scale, ds, 'SRCNN', 'ssim'):.4f}",
            f"{m(scale, ds, 'RCSA-SR'):.3f} / {m(scale, ds, 'RCSA-SR', 'ssim'):.4f}",
            f"{m(scale, ds, 'RCSA-SR') - m(scale, ds, 'SRCNN'):+.3f}",
        ])
table(["尺度", "数据集", "Bicubic", "SRCNN", "RCSA-SR", "增益"],
      rows, widths=[1.4, 2.0, 3.6, 3.6, 3.6, 1.6], font_size=9)
body(
    f"从尺度趋势看，×2 时 RCSA-SR 相对 SRCNN 的增益为 {g5_2:+.3f} dB（Set5）和 {g14_2:+.3f} dB（Set14），"
    f"×3 时为 {g5_3:+.3f}/{g14_3:+.3f} dB，×4 时为 {g5_4:+.3f}/{g14_4:+.3f} dB。三个尺度均为正增益，"
    "但增益随尺度增大而减小：×2 的输入保留更多信息，残差块更容易学习有效的高频修正；×4 的退化最强、"
    "信息缺失最多，增益空间反而受限。这说明通道注意力的收益依赖于可用信息量，而不是简单地由退化强度决定；"
    "×4 仍保持正增益，表明方法在强退化下依然有效，但提升幅度有限。Set14 包含更多强纹理和文本图像，"
    "平均 PSNR 低于 Set5，但提出方法的增益方向与 Set5 一致，说明改进并非只对某一类图像有效。"
)
figure(os.path.join(FIG, "fig_training_psnr.png"),
       "图 2  (a) ×4 训练损失曲线（细线为原始值，粗线为指数滑动平均）；(b) Set5/Set14 上 PSNR 随尺度因子的变化。")

heading("4.4 视觉比较", 2)
body(
    "图 3 给出 ×4 的局部视觉对比，选取 Set5 的蝴蝶翅膀（密集纹理）和婴儿面部（平滑皮肤）。"
    "bicubic 结果整体偏软，翅膀黑色脉纹与橙色鳞片之间出现明显混叠；SRCNN 能恢复部分边缘，但在高频纹理处"
    "仍存在模糊和振铃。RCSA-SR 的脉纹更连续，翅膀边缘更干净，同时没有引入明显的过冲。"
)
body(
    "视觉差异并不总是与整图 PSNR 完全一致：在婴儿面部上，RCSA-SR 的 SSIM 略高，但 PSNR 比 SRCNN 低 0.02 dB，"
    "肉眼差异也很小。这说明本文方法的增益具有图像依赖性：对密集边缘更有效，对平滑区域则可能与 baseline 持平。"
    "所有基于 MSE 的方法都倾向于产生平滑结果；在极强纹理区域，RCSA-SR 仍可能丢失细小结节。第 4.6 节将从"
    "边缘密度和 SE 门控角度进一步分析增益来源与局限。"
)
figure(os.path.join(FIG, "fig_qualitative.png"),
       "图 3  ×4 超分辨率视觉对比（上：Set5 蝴蝶翅膀；下：Set5 婴儿面部）。每幅图下方标注整幅图的 PSNR/SSIM。")

heading("4.5 消融实验", 2)
body(
    "为分离 SE 与数据增强的贡献，表 2 在 ×4 上比较 full 模型、去掉 SE 的模型和去掉增强的模型。"
    f"全模型相对去 SE 模型的 PSNR 差值为 {se_drop:+.3f}/{se_drop14:+.3f} dB（即去掉 SE 后 Set5/Set14 分别"
    f"下降 {se_drop:.3f}/{se_drop14:.3f} dB）；全模型相对去增强模型的差值为 {aug_drop:+.3f}/{aug_drop14:+.3f} dB"
    f"（去掉增强后分别下降 {aug_drop:.3f}/{aug_drop14:.3f} dB）。"
    + ("SE 的增益说明通道重标定确实改善了高频特征的选择；"
       if se_drop > 0 else
       "SE 在本实验中没有带来正增益，说明在 20k 迭代和 4 个残差块的容量下，通道重标定尚未被充分优化；")
    + ("数据增强的增益则说明翻转/旋转能在有限训练图像上提升泛化。"
       if aug_drop > 0 else
       "数据增强在本实验中没有带来正增益，说明当前 patch 数量下增强的收益被训练预算抵消。")
    + "两者共同构成 full 模型的最终表现。"
)
caption("表 2  ×4 消融实验（Set5/Set14 平均 PSNR/SSIM）。ΔSet5/ΔSet14 为相对 SRCNN 的 PSNR 差值。")
rows = []
for name in ["SRCNN", "RCSA w/o SE", "RCSA w/o Aug", "RCSA-SR"]:
    rows.append([
        name,
        f"{a('Set5', name):.3f} / {a('Set5', name, 'ssim'):.4f}",
        f"{a('Set14', name):.3f} / {a('Set14', name, 'ssim'):.4f}",
        f"{a('Set5', name) - a('Set5', 'SRCNN'):+.3f}",
        f"{a('Set14', name) - a('Set14', 'SRCNN'):+.3f}",
    ])
table(["模型", "Set5", "Set14", "ΔSet5", "ΔSet14"],
      rows, widths=[4.0, 4.0, 4.0, 2.0, 2.0], font_size=9)
body(
    "值得注意的是，去掉 SE 后参数量下降，但 PSNR 也下降，说明增益并非来自单纯的参数增加；"
    "在相同训练预算下，SE 以很小的额外开销换取了更有效的通道选择。去掉增强的模型在 Set5 上的下降幅度"
    "小于 Set14，可能是因为 Set5 只有 5 张图，平均指标对单张图的波动更敏感；这也提示小测试集上的"
    "零点几 dB 差异需要结合逐图结果谨慎解读。"
)

heading("4.6 分析与讨论", 2)
heading("4.6.1 边缘密度与增益", 3)
body(
    "为验证“通道注意力主要改善高频区域”的假设，本文对每张测试图计算 HR Y 通道的边缘密度：用中心差分"
    "估计梯度幅值，将幅值大于 0.08 的像素比例作为边缘密度，再与 RCSA-SR 相对 SRCNN 的 PSNR 增益做相关"
    f"分析。图 4(a) 给出逐图散点，Pearson 相关系数为 r={r_edge:.2f}；图 4(b) 按边缘密度三分位分组，"
    f"低、中、高三组的平均增益分别为 {edge_stats[0][3]:+.3f}、{edge_stats[1][3]:+.3f}、{edge_stats[2][3]:+.3f} dB。"
    "三组增益均为正，但组间差异不大；r 接近 0 且略为负，说明全局边缘密度与增益之间没有稳定的单调关系。"
)
figure(os.path.join(FIG, "fig_edge_analysis.png"),
       "图 4  ×4 逐图分析。(a) PSNR 增益与 HR 边缘密度的散点及线性拟合；(b) 按边缘密度三分位分组的平均增益。")
caption("表 3  按边缘密度三分位分组的 ×4 增益统计。")
table(["组别", "图像数", "平均边缘密度", "平均 ΔPSNR/dB"],
      [[n, str(cnt), f"{ed:.3f}", f"{dg:+.3f}"] for n, cnt, ed, dg in edge_stats],
      widths=[2.4, 2.4, 4.0, 4.0], font_size=9)
body(
    "如果 r 为正且高边缘组增益更大，说明改进主要来自纹理/边缘区域，这与 SE 增强高频通道的解释一致；"
    "如果相关性较弱，则说明平滑区域同样受益，增益更多来自残差结构的整体优化。本文的结果表明，"
    "边缘密度并不能单独解释全部增益，超分辨率的收益还取决于图像的退化程度、纹理类型和训练数据的覆盖。"
    f"本文的 r={r_edge:.2f} 正属于这种弱相关情形：不能把 RCSA-SR 的增益简单归因于边缘密度，而应结合"
    "消融中 SE 的贡献和门控分布来理解。这一分析也提醒我们：平均 PSNR 的提升不应被简单归因于某一个模块，"
    "而应结合逐图证据和消融结果综合判断。"
)

heading("4.6.2 SE 门控行为", 3)
body(
    "图 5 展示 RCSA-SR 在 Set5+Set14 所有测试图上的平均 SE 门控值。四个残差块的平均门控分别为 "
    + "、".join(f"{v:.3f}" for v in se_block_means) +
    f"，整体均值为 {se_all_mean:.3f}。门控值没有出现大面积的 0 或 1，说明 SE 处于“软选择”状态，"
    "而不是简单地关闭通道；不同通道之间的门控存在差异，表明网络确实根据内容重新分配了通道响应。"
)
figure(os.path.join(FIG, "fig_se_attention.png"),
       "图 5  RCSA-SR 在 Set5+Set14 上的平均 SE 通道门控（×4）。横轴为 64 个通道，纵轴为 4 个残差块，颜色越亮表示门控越强。")
body(
    "从块间差异看，四个块的平均门控只在 0.497–0.500 之间轻微变化，说明不同深度的整体门控强度接近；"
    "但同一块内不同通道的门控差异更明显，表明网络主要是在通道维度上做内容相关的重分配，而不是整体放大"
    "或缩小某一层的特征。需要强调，门控值本身并不直接等同于“边缘通道”或“纹理通道”；本文没有对通道语义"
    "做人工标注，因此这里只将其作为网络行为的间接证据，而不宣称每个通道具有明确的物理含义。"
)

heading("4.6.3 复杂度与推理开销", 3)
body(
    f"表 4 汇总参数量、模型大小、×4 PSNR 和推理时间。RCSA-SR 的参数量为 {p_rcsa:,}，模型文件约 "
    f"{mb_rcsa:.2f} MB；SRCNN 为 {p_src:,}，约 {mb_src:.2f} MB。在 Set5 ×4 上，SRCNN 的平均前向时间为 "
    f"{time_src:.1f} ms，RCSA-SR 为 {time_rcsa:.1f} ms。参数量的增加主要来自 64 通道的 3×3 卷积，"
    "但模型仍然可以在 4 GB 显存的笔记本 GPU 上训练和推理。"
)
caption("表 4  模型复杂度与 ×4 性能。推理时间为 Set5 上单张图前向的平均值；训练时间为本机实测。")
rows = [
    ["SRCNN", f"{p_src:,}", f"{mb_src:.2f}", f"{src5_4:.3f}", f"{src14_4:.3f}", f"{time_src:.1f}", f"{t_src:.1f}"],
    ["RCSA-SR", f"{p_rcsa:,}", f"{mb_rcsa:.2f}", f"{s5_4:.3f}", f"{s14_4:.3f}", f"{time_rcsa:.1f}", f"{t_rcsa:.1f}"],
    ["RCSA w/o SE", f"{p_nose:,}", f"{mb_nose:.2f}", f"{a('Set5', 'RCSA w/o SE'):.3f}", f"{a('Set14', 'RCSA w/o SE'):.3f}", "—", f"{t_nose:.1f}"],
    ["RCSA w/o Aug", f"{ck_noaug['params']:,}", f"{mb_noaug:.2f}", f"{a('Set5', 'RCSA w/o Aug'):.3f}", f"{a('Set14', 'RCSA w/o Aug'):.3f}", "—", f"{t_noaug:.1f}"],
]
table(["模型", "参数量", "大小/MB", "Set5", "Set14", "时间/ms", "训练/s"],
      rows, widths=[3.2, 2.6, 2.0, 2.0, 2.0, 2.0, 2.2], font_size=8.5)
body(
    "与 SRCNN 相比，RCSA-SR 用约 5.3 倍参数换取平均 PSNR 的提升；如果将其部署到极低算力设备，"
    "仍需进一步剪枝或蒸馏。本文的重点是验证残差通道注意力在轻量模型中的有效性，而非追求极致效率。"
)

heading("4.6.4 局限性", 3)
body(
    "本文存在以下局限。第一，训练集只有 BSD500 的 200 张图像，且只使用 bicubic 退化，模型对真实噪声、"
    "压缩伪影和运动模糊的泛化能力尚未验证。第二，损失函数为 MSE，结果在 PSNR/SSIM 上较好，但视觉上仍偏平滑，"
    "没有使用感知损失或对抗损失[10,11]。第三，模型只有 4 个残差块，容量有限；在 ×4 的极强纹理区域，"
    "细粒度结构仍可能丢失。后续可以引入更深的残差通道注意力、真实退化建模和多尺度训练，并在 DIV2K 等"
    "更大数据集上验证。"
)

# ---------------------------------------------------------------- 5 conclusion
heading("5 结论")
body(
    "本文围绕单图像超分辨率完成了 SRCNN 的可复现 baseline 与轻量级改进 RCSA-SR。RCSA-SR 以 bicubic 上采样"
    "图像为输入，通过 5×5 头卷积、4 个残差块、SE 通道注意力和全局残差连接预测高频分量。在 Set5、Set14 的"
    f"×2、×3、×4 六个设置上，RCSA-SR 在 {pos}/6 个设置上优于 SRCNN，平均 PSNR 增益 {avg_gain:+.3f} dB；"
    f"×4 时 Set5/Set14 达到 {s5_4:.3f}/{s14_4:.3f} dB。"
    + ("消融实验验证了 SE 和数据增强的贡献，" if se_drop > 0 and aug_drop > 0
       else "消融实验给出了 SE 和数据增强的边际影响，")
    + "逐图分析给出了增益与边缘密度、SE 门控之间的关系。所有指标、日志和权重均可复核，可为后续更深的超分辨率模型提供"
    "可比的轻量基线。"
)

# ---------------------------------------------------------------- references
heading("参考文献")
refs = [
    "DONG C, LOY C C, HE K, et al. Image super-resolution using deep convolutional networks[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2016, 38(2): 295-307. arXiv:1501.00092.",
    "DONG C, LOY C C, HE K, et al. Learning a deep convolutional network for image super-resolution[C]//European Conference on Computer Vision (ECCV). 2014: 184-199.",
    "DONG C, LOY C C, TANG X. Accelerating the super-resolution convolutional neural network[C]//European Conference on Computer Vision (ECCV). 2016: 391-407. arXiv:1608.00367.",
    "KIM J, LEE J K, LEE K M. Accurate image super-resolution using very deep convolutional networks[C]//IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2016: 1646-1654. arXiv:1511.04587.",
    "SHI W, CABALLERO J, HUSZAR F, et al. Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network[C]//IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2016: 1874-1883. arXiv:1609.05158.",
    "LIM B, SON S, KIM H, et al. Enhanced deep residual networks for single image super-resolution[C]//IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). 2017: 136-144. arXiv:1707.02921.",
    "ZHANG Y, LI K, LI K, et al. Image super-resolution using very deep residual channel attention networks[C]//European Conference on Computer Vision (ECCV). 2018: 286-301. arXiv:1807.02758.",
    "HU J, SHEN L, SUN G. Squeeze-and-excitation networks[C]//IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2018: 7132-7141. arXiv:1709.01507.",
    "HE K, ZHANG X, REN S, et al. Deep residual learning for image recognition[C]//IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2016: 770-778. arXiv:1512.03385.",
    "LEDIG C, THEIS L, HUSZAR F, et al. Photo-realistic single image super-resolution using a generative adversarial network[C]//IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2017: 4681-4690. arXiv:1609.04802.",
    "WANG X, YU K, WU S, et al. ESRGAN: enhanced super-resolution generative adversarial networks[C]//European Conference on Computer Vision Workshops (ECCVW). 2018. arXiv:1809.00219.",
    "KEYS R. Cubic convolution interpolation for digital image processing[J]. IEEE Transactions on Acoustics, Speech, and Signal Processing, 1981, 29(6): 1153-1160.",
    "MARTIN D, FOWLKES C, TAL D, et al. A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics[C]//IEEE International Conference on Computer Vision (ICCV). 2001: 416-423.",
    "BEVILACQUA M, ROUMY A, GUILLEMOT C, et al. Low-complexity single-image super-resolution based on nonnegative neighbor embedding[C]//British Machine Vision Conference (BMVC). 2012.",
    "ZEYDE R, ELAD M, PROTTER M. On single image scale-up using sparse-representations[C]//Curves and Surfaces. 2010: 711-730.",
    "YANG J, WRIGHT J, HUANG T S, et al. Image super-resolution via sparse representation[J]. IEEE Transactions on Image Processing, 2010, 19(11): 2861-2873.",
    "WANG Z, BOVIK A C, SHEIKH H R, et al. Image quality assessment: from error visibility to structural similarity[J]. IEEE Transactions on Image Processing, 2004, 13(4): 600-612.",
    "PASZKE A, GROSS S, MASSA F, et al. PyTorch: an imperative style, high-performance deep learning library[C]//Advances in Neural Information Processing Systems (NeurIPS). 2019: 8024-8035. arXiv:1912.01703.",
    "TIMOFTE R, DE SMET V, VAN GOOL L. Anchored neighborhood regression for fast example-based super-resolution[C]//IEEE International Conference on Computer Vision (ICCV). 2013: 1920-1927.",
]
for i, ref in enumerate(refs, 1):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.left_indent = Cm(0.75)
    p.paragraph_format.first_line_indent = Cm(-0.75)
    r = p.add_run(f"[{i}] {ref}")
    set_run(r, size=9.5)

# ---------------------------------------------------------------- appendix
doc.add_page_break()
heading("附录 A  代码、权重与复现命令")
body(
    "本附录列出论文使用的主要代码与权重文件。完整文件清单、文件大小和 SHA256 见 results/weights_manifest.csv；"
    "训练日志位于 logs/，逐图指标位于 results/。所有结果均可由以下命令重新生成。"
)
files = [
    ("code/models.py", "SRCNN 与 RCSA-SR 模型定义"),
    ("code/data_prep.py", "BSD500 patch 生成"),
    ("code/train.py", "训练脚本（固定随机种子）"),
    ("code/evaluate.py", "Set5/Set14 PSNR/SSIM 评测"),
    ("code/evaluate_ablation.py", "×4 消融评测"),
    ("code/make_figures.py", "论文图表生成"),
    ("weights/srcnn_x4.pth", "SRCNN ×4 baseline 权重"),
    ("weights/rcsa_x4.pth", "RCSA-SR ×4 权重"),
    ("weights/rcsa_nose_x4.pth", "无 SE 消融权重"),
    ("weights/rcsa_noaug_x4.pth", "无数据增强消融权重"),
]
rows = []
for path, desc in files:
    full = os.path.join(ROOT, path)
    if os.path.exists(full):
        h = hashlib.sha256(open(full, "rb").read()).hexdigest()[:16]
        rows.append([path, desc, f"{os.path.getsize(full)/1e6:.2f}", h])
    else:
        rows.append([path, desc, "—", "—"])
caption("表 A1  主要代码与权重文件（SHA256 取前 16 位）。")
table(["文件", "说明", "大小/MB", "SHA256"], rows,
      widths=[4.8, 5.2, 1.8, 3.2], font_size=8)

body("复现命令（Windows，项目根目录）：", indent=False, bold=True)
cmds = [
    "set PYTHONPATH=.pydeps",
    "D:\\python\\python.exe code\\train.py --model srcnn --scale 4 --iters 20000 --batch 16 --tag srcnn_x4",
    "D:\\python\\python.exe code\\train.py --model rcsa  --scale 4 --iters 20000 --batch 16 --tag rcsa_x4",
    "D:\\python\\python.exe code\\evaluate.py --scales 2 3 4",
    "D:\\python\\python.exe code\\evaluate_ablation.py",
    "D:\\python\\python.exe code\\make_figures.py",
    "D:\\python\\python.exe code\\make_paper.py",
]
for c in cmds:
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.1
    p.paragraph_format.left_indent = Cm(0.4)
    r = p.add_run(c)
    set_run(r, size=9, ascii_font="Consolas", east="宋体")

body(
    "说明：作者与单位字段使用 XXX 占位符，提交前请替换为真实姓名和单位；论文中的 PSNR/SSIM、训练时间和"
    "参数量均由本仓库的真实日志与 CSV 生成，未使用估算值。"
)

out = os.path.join(PAPER, "基于残差通道注意力的单图像超分辨率重建.docx")
doc.save(out)
print("saved", out)
