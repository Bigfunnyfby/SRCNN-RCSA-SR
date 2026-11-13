"""Download the small SR benchmarks (Set5/Set14) and BSDS500 training images.

Set5/Set14 are fetched through a GitHub raw mirror because raw.githubusercontent.com
is not directly reachable in this environment.  Only the image files are fetched.
"""
import concurrent.futures as cf
import json
import os
import re
import shutil
import tarfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
MIRROR = "https://ghfast.top/https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/"
TREE_API = "https://api.github.com/repos/jbhuang0604/SelfExSR/git/trees/master?recursive=1"
UA = {"User-Agent": "Mozilla/5.0"}


def fetch(url, dst, retries=3):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) > 1024:
        return dst
    last = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as r, open(dst + ".part", "wb") as f:
                shutil.copyfileobj(r, f)
            os.replace(dst + ".part", dst)
            return dst
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"failed {url}: {last}")


def download_benchmarks():
    req = urllib.request.Request(TREE_API, headers=UA)
    tree = json.load(urllib.request.urlopen(req, timeout=60))["tree"]
    paths = [x["path"] for x in tree]
    jobs = []
    pat = re.compile(r"data/(Set5|Set14)/image_SRF_([234])/(img_\d+)_SRF_[234]_(HR|LR)\.png$")
    for p in paths:
        m = pat.match(p)
        if not m:
            continue
        ds, scale, name, kind = m.groups()
        if kind == "HR":
            # HR is shared by all three scales; fetch it once (from the x2 tree).
            if scale != "2":
                continue
            dst = os.path.join(DATA, "benchmarks", ds, "HR", name + ".png")
        else:
            dst = os.path.join(DATA, "benchmarks", ds, "LR", "X" + scale, name + ".png")
        jobs.append((MIRROR + p, dst))
    print(f"benchmark files: {len(jobs)}")
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda a: fetch(*a), jobs))


def download_bsds():
    url = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
    tgz = os.path.join(DATA, "BSDS500", "BSR_bsds500.tgz")
    fetch(url, tgz)
    out = os.path.join(DATA, "BSDS500")
    with tarfile.open(tgz, "r:gz") as t:
        t.extractall(out)
    src_root = os.path.join(out, "BSR", "BSDS500", "data", "images")
    for split in ("train", "val"):
        src = os.path.join(src_root, split)
        dst = os.path.join(out, split)
        os.makedirs(dst, exist_ok=True)
        for fn in os.listdir(src):
            if fn.lower().endswith(".jpg"):
                shutil.copy2(os.path.join(src, fn), os.path.join(dst, fn))
    print("train", len(os.listdir(os.path.join(out, "train"))),
          "val", len(os.listdir(os.path.join(out, "val"))))


if __name__ == "__main__":
    download_benchmarks()
    download_bsds()
    print("done")
