# -*- coding: utf-8 -*-
"""Build fixed HR/LR patch arrays for fast SR training."""
import os
import argparse
import numpy as np
from PIL import Image
from random import Random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "data", "BSDS500", "train")
OUT_DIR = os.path.join(ROOT, "data", "patches")


def to_y(path):
    img = Image.open(path).convert("YCbCr")
    return np.asarray(img, dtype=np.float32)[:, :, 0] / 255.0


def make_patches(scale, patch=48, per_image=60, seed=1234, train_dir=TRAIN_DIR):
    rng = Random(seed + scale)
    files = sorted(
        os.path.join(train_dir, f) for f in os.listdir(train_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    )
    if not files:
        raise RuntimeError("no training images in " + train_dir)
    xs, ys = [], []
    for path in files:
        y = to_y(path)
        h, w = y.shape
        if h < patch or w < patch:
            continue
        for _ in range(per_image):
            top = rng.randrange(0, h - patch + 1)
            left = rng.randrange(0, w - patch + 1)
            hr = y[top:top + patch, left:left + patch]
            hr_u8 = Image.fromarray(np.clip(hr * 255.0 + 0.5, 0, 255).astype(np.uint8))
            lr_size = max(1, patch // scale)
            lr = hr_u8.resize((lr_size, lr_size), Image.BICUBIC)
            lr_up = lr.resize((patch, patch), Image.BICUBIC)
            xs.append(np.asarray(lr_up, dtype=np.float32) / 255.0)
            ys.append(hr.astype(np.float32))
    if not xs:
        raise RuntimeError(
            f"no {patch}x{patch} patches: all images are smaller than {patch}px, "
            "or the folder is empty")
    x = np.stack(xs, axis=0)[:, None, :, :]
    y = np.stack(ys, axis=0)[:, None, :, :]
    return x, y


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join("data", "BSDS500", "train"),
                    help="folder containing your own HR images (jpg/png/bmp)")
    ap.add_argument("--out", default=os.path.join("data", "patches"),
                    help="output folder for x{scale}_train.npy / y{scale}_train.npy")
    ap.add_argument("--patch", type=int, default=48)
    ap.add_argument("--per-image", type=int, default=60)
    ap.add_argument("--scales", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    train_dir = resolve(args.input)
    out_dir = resolve(args.out)
    if not os.path.isdir(train_dir):
        raise SystemExit("input folder not found: " + train_dir)
    os.makedirs(out_dir, exist_ok=True)
    for s in args.scales:
        x, y = make_patches(s, args.patch, args.per_image,
                            seed=args.seed, train_dir=train_dir)
        np.save(os.path.join(out_dir, f"x{s}_train.npy"), x)
        np.save(os.path.join(out_dir, f"y{s}_train.npy"), y)
        print(f"scale x{s}: x={x.shape} y={y.shape} "
              f"x_range=({x.min():.3f},{x.max():.3f}) out={out_dir}")


if __name__ == "__main__":
    main()
