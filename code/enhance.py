# -*- coding: utf-8 -*-
"""PyCharm-friendly single-image super-resolution demo.

Two modes:
  upscale : input is a low-resolution image; output is scale x larger.
  restore : input is a normal image; simulate LR, then reconstruct at same size.
"""
import os
import sys
import time
import argparse

# Make the project-local dependency folder importable when run from PyCharm,
# where PYTHONPATH is not always inherited from the PowerShell session.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPS = os.path.join(ROOT, ".pydeps")
if os.path.isdir(DEPS) and DEPS not in sys.path:
    sys.path.insert(0, DEPS)

import numpy as np
from PIL import Image
import torch

from evaluate import load_model, to_y_img, psnr_torch, ssim_torch, gaussian_window



def resolve(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def y_tensor(img):
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)[None, None]


def colorize(out_y, up_ycbcr):
    h, w = out_y.shape
    ycbcr = up_ycbcr.copy()
    ycbcr[:, :, 0] = np.clip(out_y * 255.0, 0, 255).astype(np.uint8)
    return Image.frombytes("YCbCr", (w, h), ycbcr.tobytes()).convert("RGB")


def infer_tiled(model, x, tile=256, overlap=32):
    """Run a fully-convolutional SR model in overlapping tiles.

    A 4x upscale of a 2K image would otherwise create multi-GB activations and
    exhaust a 4 GB laptop GPU. Tiling keeps peak memory bounded.
    """
    h, w = x.shape[-2:]
    if tile <= 0 or h * w <= 512 * 512 or (h <= tile and w <= tile):
        return model(x).clamp(0, 1)
    overlap = max(0, min(overlap, tile // 2))
    step = tile - overlap
    ys = list(range(0, max(1, h - tile + 1), step))
    xs = list(range(0, max(1, w - tile + 1), step))
    if ys[-1] != max(0, h - tile):
        ys.append(max(0, h - tile))
    if xs[-1] != max(0, w - tile):
        xs.append(max(0, w - tile))

    out = torch.zeros(1, 1, h, w, dtype=x.dtype, device=x.device)
    count = torch.zeros(1, 1, h, w, dtype=x.dtype, device=x.device)
    for y in ys:
        for x0 in xs:
            patch = x[:, :, y:y + tile, x0:x0 + tile]
            pred = model(patch).clamp(0, 1)
            out[:, :, y:y + tile, x0:x0 + tile] += pred
            count[:, :, y:y + tile, x0:x0 + tile] += 1.0
    return out / count.clamp_min(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="input image path")
    ap.add_argument("--weights", default="",
                    help="checkpoint produced by train.py; default rcsa_x{scale}.pth")
    ap.add_argument("--scale", type=int, default=4, choices=[2, 3, 4])
    ap.add_argument("--mode", choices=["upscale", "restore"], default="upscale",
                    help="upscale: input is LR; restore: input is HR, output same size")
    ap.add_argument("--out", default="", help="output PNG path")
    ap.add_argument("--tile", type=int, default=256,
                    help="tile size for large images; 0 disables tiling")
    ap.add_argument("--overlap", type=int, default=32,
                    help="overlap between neighboring tiles")
    ap.add_argument("--cpu", action="store_true", help="force CPU inference")
    args = ap.parse_args()
    if not args.weights:
        args.weights = f"weights/rcsa_x{args.scale}.pth"

    image_path = resolve(args.image)
    weight_path = resolve(args.weights)
    if not os.path.exists(image_path):
        raise SystemExit("image not found: " + image_path)
    if not os.path.exists(weight_path):
        raise SystemExit("weights not found: " + weight_path)

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    img = Image.open(image_path).convert("RGB")
    hr_y_img = to_y_img(image_path)
    w, h = hr_y_img.size

    if args.mode == "upscale":
        out_w, out_h = w * args.scale, h * args.scale
        up_y = hr_y_img.resize((out_w, out_h), Image.BICUBIC)
        up_rgb = img.resize((out_w, out_h), Image.BICUBIC)
        reference = None
    else:
        out_w, out_h = w, h
        lr_size = (max(1, w // args.scale), max(1, h // args.scale))
        lr_y = hr_y_img.resize(lr_size, Image.BICUBIC)
        up_y = lr_y.resize((w, h), Image.BICUBIC)
        up_rgb = img.resize(lr_size, Image.BICUBIC).resize((w, h), Image.BICUBIC)
        reference = hr_y_img

    up_ycbcr = np.asarray(up_rgb.convert("YCbCr"))
    model, ckpt = load_model(weight_path, device)
    ckpt_scale = int(ckpt.get("scale", args.scale))
    if ckpt_scale != args.scale:
        raise SystemExit(
            f"checkpoint is x{ckpt_scale}, but --scale is x{args.scale}; "
            "use the matching weight or change --scale")
    x = y_tensor(up_y).to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    t0 = time.time()
    used_cpu_fallback = False
    try:
        with torch.no_grad():
            out = infer_tiled(model, x, tile=args.tile, overlap=args.overlap)
    except torch.cuda.OutOfMemoryError:
        print("CUDA out of memory; retrying on CPU (slower)...")
        torch.cuda.empty_cache()
        model.to("cpu")
        x = x.cpu()
        with torch.no_grad():
            out = infer_tiled(model, x, tile=args.tile, overlap=args.overlap)
        used_cpu_fallback = True
    if device.type == "cuda" and not used_cpu_fallback:
        torch.cuda.synchronize()
    elapsed = (time.time() - t0) * 1000.0

    sr_rgb = colorize(out[0, 0].cpu().numpy(), up_ycbcr)
    if not args.out:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(ROOT, "results",
                                f"{stem}_x{args.scale}_{args.mode}.png")
    else:
        out_path = resolve(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sr_rgb.save(out_path)

    print(f"model={ckpt['model']} params={ckpt['params']:,} scale=x{args.scale} mode={args.mode}")
    if args.tile > 0 and out_w * out_h > 512 * 512:
        print(f"tiled inference: tile={args.tile}, overlap={args.overlap}")
    print(f"input={w}x{h} -> output={out_w}x{out_h}  time={elapsed:.1f} ms")
    if reference is not None:
        ref = y_tensor(reference).to(out.device)
        window = gaussian_window(11, 1.5, device=out.device)
        print(f"PSNR={psnr_torch(out, ref):.3f} dB  SSIM={ssim_torch(out, ref, window):.4f}")
    print("saved", out_path)


if __name__ == "__main__":
    main()
