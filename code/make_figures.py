# -*- coding: utf-8 -*-
"""Generate paper figures from real logs / checkpoints / result CSV files."""
import os
import csv
import json
import numpy as np
from PIL import Image
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import build_model
from evaluate import to_y_img, psnr_torch, ssim_torch, gaussian_window, load_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
os.makedirs(FIG, exist_ok=True)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_log(path):
    rows = read_csv(path)
    it = np.array([int(r["iter"]) for r in rows])
    loss = np.array([float(r["loss"]) for r in rows])
    return it, loss


def smooth(y, alpha=0.03):
    out = np.empty_like(y)
    acc = y[0]
    for i, v in enumerate(y):
        acc = (1 - alpha) * acc + alpha * v
        out[i] = acc
    return out


def y_tensor(img):
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)[None, None]


def colorize(out_y, up_ycbcr):
    ycbcr = up_ycbcr.copy()
    ycbcr[:, :, 0] = np.clip(out_y * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(ycbcr, "YCbCr").convert("RGB")


def fig_qualitative():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    window = gaussian_window(11, 1.5, device=device)
    models = {}
    for name, tag in [("SRCNN", "srcnn_x4"), ("RCSA-SR", "rcsa_x4")]:
        path = os.path.join(ROOT, "weights", tag + ".pth")
        if os.path.exists(path):
            models[name], _ = load_model(path, device)
    samples = [
        ("Set5", "img_003", (62, 28, 198, 164)),
        ("Set5", "img_001", (120, 105, 330, 315)),
    ]
    fig, axes = plt.subplots(len(samples), 3, figsize=(8.6, 5.6))
    for r, (dataset, name, box) in enumerate(samples):
        path = os.path.join(ROOT, "data", "benchmarks", dataset, "HR", name + ".png")
        hr_img = Image.open(path).convert("RGB")
        hr_y_img = to_y_img(path)
        w, h = hr_y_img.size
        lr_y = hr_y_img.resize((w // 4, h // 4), Image.BICUBIC)
        up_y = lr_y.resize((w, h), Image.BICUBIC)
        up_img = hr_img.resize((w // 4, h // 4), Image.BICUBIC).resize((w, h), Image.BICUBIC)
        up_ycbcr = np.asarray(up_img.convert("YCbCr"))
        up_t = y_tensor(up_y).to(device)
        hr_t = y_tensor(hr_y_img).to(device)
        panels = [("Bicubic", up_img, None)]
        for mname, model in models.items():
            with torch.no_grad():
                out = model(up_t).clamp(0, 1)
            p = psnr_torch(out, hr_t)
            s = ssim_torch(out, hr_t, window)
            rgb = colorize(out[0, 0].cpu().numpy(), up_ycbcr)
            panels.append((f"{mname}\n{p:.2f} dB / {s:.4f}", rgb, (p, s)))
        bp = psnr_torch(up_t, hr_t)
        bs = ssim_torch(up_t, hr_t, window)
        panels[0] = (f"Bicubic\n{bp:.2f} dB / {bs:.4f}", panels[0][1], (bp, bs))
        for c, (title, rgb, _) in enumerate(panels):
            crop = rgb.crop(box).resize((240, 240), Image.BICUBIC)
            ax = axes[r, c]
            ax.imshow(crop)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, fontsize=8.5)
    fig.tight_layout(pad=0.4)
    out = os.path.join(FIG, "fig_qualitative.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def fig_training_and_psnr():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    for tag, label, color in [("srcnn_x4", "SRCNN", "#1f77b4"), ("rcsa_x4", "RCSA-SR", "#d62728")]:
        path = os.path.join(ROOT, "logs", tag + ".csv")
        if not os.path.exists(path):
            continue
        it, loss = read_log(path)
        ax.plot(it, loss, color=color, alpha=0.20, linewidth=0.7)
        ax.plot(it, smooth(loss), color=color, linewidth=1.8, label=label)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE loss")
    ax.set_title("(a) Training loss at x4")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)

    metrics = read_csv(os.path.join(RES, "metrics.csv"))
    ax = axes[1]
    for dataset, ls in [("Set5", "-"), ("Set14", "--")]:
        for method, color, marker in [("SRCNN", "#1f77b4", "o"), ("RCSA-SR", "#d62728", "s")]:
            vals = [m for m in metrics if m["dataset"] == dataset and m["method"] == method]
            vals.sort(key=lambda m: int(m["scale"]))
            if vals:
                ax.plot([int(v["scale"]) for v in vals], [float(v["psnr"]) for v in vals],
                        ls, color=color, marker=marker, label=f"{dataset} {method}")
    ax.set_xlabel("Scale factor")
    ax.set_ylabel("PSNR (dB)")
    ax.set_xticks([2, 3, 4])
    ax.set_title("(b) PSNR on benchmark sets")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    out = os.path.join(FIG, "fig_training_psnr.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def edge_density(img):
    y = np.asarray(img, dtype=np.float32) / 255.0
    gy, gx = np.gradient(y)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return float((mag > 0.08).mean())


def fig_edge():
    rows = read_csv(os.path.join(RES, "per_image.csv"))
    rows = [r for r in rows if int(r["scale"]) == 4]
    by = {}
    for r in rows:
        by.setdefault((r["dataset"], r["image"]), {})[r["method"]] = float(r["psnr"])
    out_rows = []
    for (dataset, image), d in by.items():
        if "SRCNN" not in d or "RCSA-SR" not in d:
            continue
        path = os.path.join(ROOT, "data", "benchmarks", dataset, "HR", image + ".png")
        ed = edge_density(to_y_img(path))
        out_rows.append(dict(dataset=dataset, image=image, edge_density=ed,
                             psnr_srcnn=d["SRCNN"], psnr_rcsa=d["RCSA-SR"],
                             delta=d["RCSA-SR"] - d["SRCNN"]))
    with open(os.path.join(RES, "edge_analysis.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "image", "edge_density", "psnr_srcnn", "psnr_rcsa", "delta"])
        w.writeheader(); w.writerows(out_rows)
    x = np.array([r["edge_density"] for r in out_rows])
    y = np.array([r["delta"] for r in out_rows])
    r = float(np.corrcoef(x, y)[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    ax = axes[0]
    for dataset, color in [("Set5", "#d62728"), ("Set14", "#1f77b4")]:
        m = np.array([row["dataset"] == dataset for row in out_rows])
        ax.scatter(x[m], y[m], s=34, color=color, label=dataset, alpha=0.85)
    if len(x) > 2:
        k, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, k * xs + b, color="black", linewidth=1.2, linestyle="--")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Edge density of HR image")
    ax.set_ylabel("PSNR gain (dB)")
    ax.set_title(f"(a) Per-image gain vs. edge density (r={r:.2f})")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)

    order = np.argsort(x)
    groups = np.array_split(order, 3)
    labels = ["low", "mid", "high"]
    means = [float(np.mean(y[g])) for g in groups]
    ax = axes[1]
    bars = ax.bar(labels, means, color=["#9ecae1", "#6baed6", "#2171b5"])
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + (0.002 if m >= 0 else -0.006),
                f"{m:+.3f}", ha="center", fontsize=8)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylabel("Mean PSNR gain (dB)")
    ax.set_title("(b) Gain by edge-density tertile")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, "fig_edge_analysis.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", out, "pearson", r)


def fig_se():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = os.path.join(ROOT, "weights", "rcsa_x4.pth")
    if not os.path.exists(path):
        print("missing", path); return
    model, _ = load_model(path, device)
    blocks = [m for m in model.body if hasattr(m, "se")]
    if not blocks:
        print("no SE blocks"); return
    acc = [[] for _ in blocks]
    for dataset in ["Set5", "Set14"]:
        folder = os.path.join(ROOT, "data", "benchmarks", dataset, "HR")
        for f in sorted(os.listdir(folder)):
            if not f.lower().endswith((".png", ".jpg", ".bmp")):
                continue
            img = to_y_img(os.path.join(folder, f))
            w, h = img.size
            up = img.resize((w // 4, h // 4), Image.BICUBIC).resize((w, h), Image.BICUBIC)
            with torch.no_grad():
                model(y_tensor(up).to(device))
            for i, blk in enumerate(blocks):
                acc[i].append(blk.se.last_attention.view(-1).cpu().numpy())
    means = np.stack([np.mean(np.stack(a), axis=0) for a in acc])  # blocks x channels
    stds = np.stack([np.std(np.stack(a), axis=0) for a in acc])
    with open(os.path.join(RES, "se_attention.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["block", "channel", "mean", "std"])
        for i in range(means.shape[0]):
            for c in range(means.shape[1]):
                w.writerow([i + 1, c, f"{means[i, c]:.6f}", f"{stds[i, c]:.6f}"])
    fig, ax = plt.subplots(figsize=(10.0, 2.9))
    im = ax.imshow(means, aspect="auto", cmap="viridis")
    ax.set_xlabel("Channel index")
    ax.set_ylabel("Residual block")
    ax.set_yticks(range(means.shape[0]))
    ax.set_yticklabels([str(i + 1) for i in range(means.shape[0])])
    ax.set_title("Mean SE channel attention over Set5 + Set14 (x4)")
    cbar = fig.colorbar(im, ax=ax, pad=0.015)
    cbar.set_label("Gate value")
    fig.tight_layout()
    out = os.path.join(FIG, "fig_se_attention.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    fig_qualitative()
    fig_training_and_psnr()
    fig_edge()
    fig_se()
