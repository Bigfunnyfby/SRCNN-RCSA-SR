# -*- coding: utf-8 -*-
"""Evaluate x4 ablation variants on Set5 + Set14 (Y channel)."""
import os
import csv
import argparse
import numpy as np
from PIL import Image
import torch

from evaluate import to_y_img, psnr_torch, ssim_torch, gaussian_window, load_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "data", "benchmarks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--out", type=str, default=os.path.join(ROOT, "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    window = gaussian_window(11, 1.5, device=device)
    specs = [
        ("SRCNN", "srcnn_x4"),
        ("RCSA-SR", "rcsa_x4"),
        ("RCSA w/o SE", "rcsa_nose_x4"),
        ("RCSA w/o Aug", "rcsa_noaug_x4"),
    ]
    models = {}
    for name, tag in specs:
        path = os.path.join(ROOT, "weights", tag + ".pth")
        if not os.path.exists(path):
            print("missing", path)
            continue
        models[name], ckpt = load_model(path, device)
        print(name, "params", f"{ckpt['params']:,}")
    rows = []
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
            lr_y = hr_y_img.resize((max(1, w // args.scale), max(1, h // args.scale)), Image.BICUBIC)
            up_y = lr_y.resize((w, h), Image.BICUBIC)
            up = torch.from_numpy(np.asarray(up_y, dtype=np.float32) / 255.0)[None, None].to(device)
            hr = torch.from_numpy(np.asarray(hr_y_img, dtype=np.float32) / 255.0)[None, None].to(device)
            for name, model in models.items():
                with torch.no_grad():
                    out = model(up).clamp(0, 1)
                rows.append(dict(dataset=dataset, image=img_name, method=name,
                                 psnr=psnr_torch(out, hr), ssim=ssim_torch(out, hr, window)))
    out_path = os.path.join(args.out, "ablation_x4.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "image", "method", "psnr", "ssim"])
        w.writeheader()
        w.writerows(rows)
    print("\n=== ablation x4 ===")
    for dataset in ["Set5", "Set14"]:
        for name, _ in specs:
            vals = [r for r in rows if r["dataset"] == dataset and r["method"] == name]
            if vals:
                print(f"{dataset:5s} {name:14s} PSNR {np.mean([v['psnr'] for v in vals]):.3f} "
                      f"SSIM {np.mean([v['ssim'] for v in vals]):.4f}")
    print("saved", out_path)


if __name__ == "__main__":
    main()
