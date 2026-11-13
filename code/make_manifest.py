# -*- coding: utf-8 -*-
"""Create a manifest of code, weights, logs and result files."""
import os
import csv
import glob
import hashlib
import platform
import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    patterns = [
        ("code", "code/*.py"),
        ("code", "code/*.sh"),
        ("weights", "weights/*.pth"),
        ("logs", "logs/*.csv"),
        ("results", "results/*.csv"),
        ("results", "results/*.json"),
        ("figures", "figures/*.png"),
        ("paper", "paper/*.docx"),
        ("paper", "paper/*.pdf"),
        ("root", "README.md"),
        ("root", "REPRODUCE.md"),
        ("root", "PYCHARM_GUIDE.md"),
        ("root", "GITHUB_UPLOAD.md"),
        ("root", "EXPERIMENT_LOG.md"),
        ("root", "requirements.txt"),
    ]
    rows = []
    for category, pat in patterns:
        for path in sorted(glob.glob(os.path.join(ROOT, pat))):
            if "__pycache__" in path:
                continue
            rel = os.path.relpath(path, ROOT).replace("\\", "/")
            rows.append(dict(path=rel, category=category,
                             bytes=os.path.getsize(path),
                             sha256=sha256(path)))
    out = os.path.join(ROOT, "results", "weights_manifest.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "category", "bytes", "sha256"])
        w.writeheader()
        w.writerows(rows)
    print("saved", out, "files", len(rows))

    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        env = [
            f"time={datetime.datetime.now().isoformat(timespec='seconds')}",
            f"platform={platform.platform()}",
            f"python={platform.python_version()}",
            f"torch={torch.__version__}",
            f"cuda_available={torch.cuda.is_available()}",
            f"cuda={torch.version.cuda}",
            f"gpu={gpu}",
        ]
    except Exception as e:
        env = [f"time={datetime.datetime.now().isoformat(timespec='seconds')}",
               f"platform={platform.platform()}", f"python={platform.python_version()}",
               f"torch_error={e}"]
    env_path = os.path.join(ROOT, "results", "env.txt")
    with open(env_path, "w", encoding="utf-8") as f:
        f.write("\n".join(env) + "\n")
    print("saved", env_path)


if __name__ == "__main__":
    main()
