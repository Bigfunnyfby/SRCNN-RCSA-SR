# -*- coding: utf-8 -*-
"""Make a Bicubic / SRCNN / RCSA-SR comparison figure for one image.

Default mode ``restore`` simulates a low-resolution observation by downsampling
the input by ``--scale`` and upsampling it again, so PSNR/SSIM can be computed
against the original image.  ``upscale`` treats the input as the LR image and
produces a larger output without ground-truth metrics.
"""
import os
import sys
import time
import argparse
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPS = os.path.join(ROOT, ".pydeps")
if os.path.isdir(DEPS) and DEPS not in sys.path:
    sys.path.insert(0, DEPS)

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate import load_model, to_y_img, psnr_torch, ssim_torch, gaussian_window
from enhance import infer_tiled, colorize, y_tensor


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--scale", type=int, default=4, choices=[2, 3, 4])
    ap.add_argument("--mode", choices=["restore", "upscale"], default="restore")
    ap.add_argument("--srcnn", default="weights/srcnn_x{scale}.pth")
    ap.add_argument("--rcsa", default="weights/rcsa_x{scale}.pth")
    ap.add_argument("--out", default="")
    ap.add_argument("--crop", type=int, nargs=4, default=None,
                    metavar=("X", "Y", "W", "H"),
                    help="optional crop in HR/output pixels")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=32)
    args = ap.parse_args()

    image_path = resolve(args.image)
    if not os.path.exists(image_path):
        raise SystemExit("image not found: " + image_path)
    srcnn_path = resolve(args.srcnn.format(scale=args.scale))
    rcsa_path = resolve(args.rcsa.format(scale=args.scale))
    for p in (srcnn_path, rcsa_path):
        if not os.path.exists(p):
            raise SystemExit("weights not found: " + p)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = Image.open(image_path).convert("RGB")
    hr_y_img = to_y_img(image_path)
    w, h = hr_y_img.size

    if args.mode == "restore":
        out_size = (w, h)
        lr_size = (max(1, w // args.scale), max(1, h // args.scale))
        lr_y = hr_y_img.resize(lr_size, Image.BICUBIC)
        up_y = lr_y.resize(out_size, Image.BICUBIC)
        up_rgb = img.resize(lr_size, Image.BICUBIC).resize(out_size, Image.BICUBIC)
        reference = hr_y_img
    else:
        out_size = (w * args.scale, h * args.scale)
        up_y = hr_y_img.resize(out_size, Image.BICUBIC)
        up_rgb = img.resize(out_size, Image.BICUBIC)
        reference = None

    up_ycbcr = np.asarray(up_rgb.convert("YCbCr"))
    x = y_tensor(up_y).to(device)
    window = gaussian_window(11, 1.5, device=device)

    panels = [("Bicubic", up_rgb)]
    for name, path in [("SRCNN", srcnn_path), ("RCSA-SR", rcsa_path)]:
        model, ckpt = load_model(path, device)
        ckpt_scale = int(ckpt.get("scale", args.scale))
        if ckpt_scale != args.scale:
            raise SystemExit(f"{path} is x{ckpt_scale}, expected x{args.scale}")
        t0 = time.time()
        with torch.no_grad():
            out = infer_tiled(model, x, tile=args.tile, overlap=args.overlap)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.time() - t0) * 1000.0
        out_y = out[0, 0].cpu().numpy()
        panels.append((name, colorize(out_y, up_ycbcr)))
        if reference is not None:
            ref = y_tensor(reference).to(device)
            p = psnr_torch(out, ref)
            s = ssim_torch(out, ref, window)
            panels[-1] = (f"{name}\n{p:.2f} dB / {s:.4f}", panels[-1][1])
        print(f"{name}: {elapsed:.1f} ms")

    if reference is not None:
        ref = y_tensor(reference).to(device)
        p = psnr_torch(x, ref)
        s = ssim_torch(x, ref, window)
        panels[0] = (f"Bicubic\n{p:.2f} dB / {s:.4f}", panels[0][1])

    # Match the figure aspect to the whole image, so an uncropped portrait or
    # landscape image is not squeezed into a fixed 4:3 panel.
    panel_h = 4.0 * out_size[1] / max(1, out_size[0])
    panel_h = max(3.0, min(7.0, panel_h))
    fig, axes = plt.subplots(1, 3, figsize=(12.0, panel_h))
    for ax, (title, panel) in zip(axes, panels):
        if args.crop:
            panel = panel.crop(tuple(args.crop))
        ax.imshow(panel)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(pad=0.4)
    if not args.out:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(ROOT, "results",
                                f"compare_{stem}_x{args.scale}_{args.mode}.png")
    else:
        out_path = resolve(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", out_path)


if __name__ == "__main__":
    main()
