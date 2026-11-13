# -*- coding: utf-8 -*-
"""Run a trained SR model on one image and save the reconstructed RGB image."""
import os
import argparse
import numpy as np
from PIL import Image
import torch

from evaluate import load_model, to_y_img, psnr_torch, ssim_torch, gaussian_window

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_path = args.image if os.path.isabs(args.image) else os.path.join(ROOT, args.image)
    weight_path = args.weights if os.path.isabs(args.weights) else os.path.join(ROOT, args.weights)
    img = Image.open(image_path).convert("RGB")
    hr_y_img = to_y_img(image_path)
    w, h = hr_y_img.size
    lr_size = (max(1, w // args.scale), max(1, h // args.scale))

    lr_y = hr_y_img.resize(lr_size, Image.BICUBIC)
    up_y = lr_y.resize((w, h), Image.BICUBIC)
    up_rgb = img.resize(lr_size, Image.BICUBIC).resize((w, h), Image.BICUBIC)
    up_ycbcr = np.asarray(up_rgb.convert("YCbCr"))

    model, ckpt = load_model(weight_path, device)
    x = torch.from_numpy(np.asarray(up_y, dtype=np.float32) / 255.0)[None, None].to(device)
    hr = torch.from_numpy(np.asarray(hr_y_img, dtype=np.float32) / 255.0)[None, None].to(device)
    with torch.no_grad():
        out = model(x).clamp(0, 1)

    ycbcr = up_ycbcr.copy()
    ycbcr[:, :, 0] = np.clip(out[0, 0].cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    sr_rgb = Image.frombytes("YCbCr", (w, h), ycbcr.tobytes()).convert("RGB")

    out_path = args.out
    if not out_path:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(ROOT, "results", f"{stem}_x{args.scale}_sr.png")
    elif not os.path.isabs(out_path):
        out_path = os.path.join(ROOT, out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sr_rgb.save(out_path)

    window = gaussian_window(11, 1.5, device=device)
    p, s = psnr_torch(out, hr), ssim_torch(out, hr, window)
    print(f"model={ckpt['model']} scale=x{args.scale} params={ckpt['params']:,}")
    print(f"PSNR={p:.3f} dB  SSIM={s:.4f}")
    print("saved", out_path)


if __name__ == "__main__":
    main()
