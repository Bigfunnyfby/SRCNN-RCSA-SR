# -*- coding: utf-8 -*-
"""Train SRCNN / RCSA-SR on fixed patch arrays."""
import os
import argparse
import csv
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models import build_model, count_params

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PatchDataset(Dataset):
    def __init__(self, x_path, y_path, augment=True):
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        self.augment = augment

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        x = torch.from_numpy(np.array(self.x[i], dtype=np.float32))
        y = torch.from_numpy(np.array(self.y[i], dtype=np.float32))
        if self.augment:
            if random.random() < 0.5:
                x = torch.flip(x, dims=[2])
                y = torch.flip(y, dims=[2])
            if random.random() < 0.5:
                x = torch.flip(x, dims=[1])
                y = torch.flip(y, dims=[1])
            k = random.randrange(4)
            if k:
                x = torch.rot90(x, k, dims=[1, 2])
                y = torch.rot90(y, k, dims=[1, 2])
        return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["srcnn", "rcsa"], required=True)
    ap.add_argument("--scale", type=int, required=True)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--res-scale", type=float, default=0.5)
    ap.add_argument("--use-se", type=int, default=1)
    ap.add_argument("--augment", type=int, default=1)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--data-dir", type=str, default=os.path.join("data", "patches"),
                    help="folder containing x{scale}_train.npy / y{scale}_train.npy")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    data_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(ROOT, args.data_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    tag = args.tag or f"{args.model}_x{args.scale}"
    out_ckpt = os.path.join(ROOT, "weights", tag + ".pth")
    log_path = os.path.join(ROOT, "logs", tag + ".csv")
    os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    x_path = os.path.join(data_dir, f"x{args.scale}_train.npy")
    y_path = os.path.join(data_dir, f"y{args.scale}_train.npy")
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise SystemExit(
            f"patch files not found in {data_dir}; run data_prep.py with --out first")
    ds = PatchDataset(x_path, y_path, augment=bool(args.augment))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0,
                    pin_memory=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {}
    if args.model == "rcsa":
        kwargs = dict(num_blocks=args.blocks, use_se=bool(args.use_se),
                      channels=args.channels, res_scale=args.res_scale)
    model = build_model(args.model, **kwargs).to(device)
    n_params = count_params(model)
    print(f"tag={tag} params={n_params:,} device={device} "
          f"train={len(ds)} iters={args.iters} batch={args.batch} data={data_dir}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.iters, eta_min=args.min_lr)

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iter", "loss", "lr", "elapsed_sec"])

    model.train()
    t0 = time.time()
    it = 0
    running = 0.0
    running_n = 0
    while it < args.iters:
        for x, y in dl:
            it += 1
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running += loss.item()
            running_n += 1
            if it % 100 == 0:
                avg = running / max(1, running_n)
                elapsed = time.time() - t0
                lr = optimizer.param_groups[0]["lr"]
                print(f"[{tag}] it {it}/{args.iters} loss {avg:.6f} "
                      f"lr {lr:.2e} {elapsed:.1f}s", flush=True)
                with open(log_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([it, f"{avg:.6f}", f"{lr:.8f}", f"{elapsed:.2f}"])
                running = 0.0
                running_n = 0
            if it >= args.iters:
                break

    torch.save({
        "model": args.model,
        "state_dict": model.state_dict(),
        "scale": args.scale,
        "tag": tag,
        "params": n_params,
        "args": vars(args),
        "elapsed_sec": time.time() - t0,
    }, out_ckpt)
    print(f"saved {out_ckpt} ({os.path.getsize(out_ckpt)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
