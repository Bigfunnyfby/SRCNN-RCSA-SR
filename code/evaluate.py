# -*- coding: utf-8 -*-
"""Evaluate bicubic / SRCNN / RCSA-SR on Set5 and Set14 (Y channel)."""
import os
import csv
import json
import time
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from models import build_model, count_params

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "data", "benchmarks")


def to_y_img(path):
    img = Image.open(path).convert("YCbCr")
    return img.split()[0]


def psnr_torch(a, b):
    mse = torch.mean((a - b) ** 2)
    if mse.item() <= 0:
        return 99.0
    return float(10.0 * torch.log10(1.0 / mse))


def gaussian_window(size=11, sigma=1.5, device="cpu"):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).view(1, 1, size, size)


def ssim_torch(a, b, window):
    # a, b: (1,1,H,W) in [0,1]
    mu1 = F.conv2d(a, window, padding=window.shape[-1] // 2)
    mu2 = F.conv2d(b, window, padding=window.shape[-1] // 2)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(a * a, window, padding=window.shape[-1] // 2) - mu1_sq
    sigma2_sq = F.conv2d(b * b, window, padding=window.shape[-1] // 2) - mu2_sq
    sigma12 = F.conv2d(a * b, window, padding=window.shape[-1] // 2) - mu1_mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(ssim_map.mean())


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    kwargs = {}
    if ckpt["model"] == "rcsa":
        a = ckpt["args"]
        kwargs = dict(num_blocks=a["blocks"], use_se=bool(a["use_se"]),
                      channels=a.get("channels", 64),
                      res_scale=a.get("res_scale", 0.1))
    model = build_model(ckpt["model"], **kwargs).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--out", type=str, default=os.path.join(ROOT, "results"))
    ap.add_argument("--srcnn-tag", type=str, default="srcnn_x{scale}",
                    help="weight tag template; {scale} is replaced by 2/3/4")
    ap.add_argument("--rcsa-tag", type=str, default="rcsa_x{scale}",
                    help="weight tag template; {scale} is replaced by 2/3/4")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    window = gaussian_window(11, 1.5, device=device)

    rows = []
    summary = []
    for scale in args.scales:
        methods = [("Bicubic", None)]
        for name, template in [("SRCNN", args.srcnn_tag), ("RCSA-SR", args.rcsa_tag)]:
            tag = template.format(scale=scale)
            path = os.path.join(ROOT, "weights", tag + ".pth")
            if os.path.exists(path):
                methods.append((name, path))
            else:
                print("missing", path)
        models = {}
        for name, path in methods:
            if path:
                models[name], ckpt = load_model(path, device)
                print(name, "params", f"{ckpt['params']:,}")
        for dataset in ["Set5", "Set14"]:
            paths = sorted(
                os.path.join(BENCH, dataset, "HR", f)
                for f in os.listdir(os.path.join(BENCH, dataset, "HR"))
                if f.lower().endswith((".png", ".jpg", ".bmp"))
            )
            for path in paths:
                img_name = os.path.splitext(os.path.basename(path))[0]
                hr_y_img = to_y_img(path)
                w, h = hr_y_img.size
                lr_size = (max(1, w // scale), max(1, h // scale))
                lr_y = hr_y_img.resize(lr_size, Image.BICUBIC)
                up_y = lr_y.resize((w, h), Image.BICUBIC)
                up = torch.from_numpy(np.asarray(up_y, dtype=np.float32) / 255.0)[None, None].to(device)
                hr = torch.from_numpy(np.asarray(hr_y_img, dtype=np.float32) / 255.0)[None, None].to(device)

                # bicubic
                rows.append(dict(dataset=dataset, image=img_name, scale=scale,
                                 method="Bicubic",
                                 psnr=psnr_torch(up, hr),
                                 ssim=ssim_torch(up, hr, window),
                                 time_ms=0.0))
                for name, _ in methods:
                    if name == "Bicubic":
                        continue
                    model = models[name]
                    # warmup once
                    if name not in getattr(main, "_warmed", set()):
                        with torch.no_grad():
                            _ = model(up)
                        torch.cuda.synchronize()
                        main._warmed = getattr(main, "_warmed", set()) | {name}
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        out = model(up).clamp(0, 1)
                    torch.cuda.synchronize()
                    dt = (time.perf_counter() - t0) * 1000.0
                    rows.append(dict(dataset=dataset, image=img_name, scale=scale,
                                     method=name,
                                     psnr=psnr_torch(out, hr),
                                     ssim=ssim_torch(out, hr, window),
                                     time_ms=dt))

    per_path = os.path.join(args.out, "per_image.csv")
    with open(per_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "image", "scale", "method", "psnr", "ssim", "time_ms"])
        w.writeheader()
        w.writerows(rows)

    for scale in args.scales:
        for dataset in ["Set5", "Set14"]:
            for method in ["Bicubic", "SRCNN", "RCSA-SR"]:
                vals = [r for r in rows if r["scale"] == scale and r["dataset"] == dataset and r["method"] == method]
                if not vals:
                    continue
                summary.append(dict(
                    scale=scale, dataset=dataset, method=method,
                    n=len(vals),
                    psnr=float(np.mean([v["psnr"] for v in vals])),
                    ssim=float(np.mean([v["ssim"] for v in vals])),
                    time_ms=float(np.mean([v["time_ms"] for v in vals])),
                ))
    with open(os.path.join(args.out, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["scale", "dataset", "method", "n", "psnr", "ssim", "time_ms"])
        w.writeheader()
        w.writerows(summary)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== summary ===")
    for s in summary:
        print(f"x{s['scale']} {s['dataset']:5s} {s['method']:8s} "
              f"PSNR {s['psnr']:.3f} SSIM {s['ssim']:.4f} time {s['time_ms']:.1f} ms")
    print("saved", per_path)


if __name__ == "__main__":
    main()
