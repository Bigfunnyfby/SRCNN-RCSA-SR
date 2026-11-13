# -*- coding: utf-8 -*-
"""Package code, weights, logs, results, figures and paper into one zip."""
import os
import zipfile
import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "deliverables", "图像大作业_代码权重结果.zip")
INCLUDE_DIRS = ["code", "weights", "logs", "results", "figures", "paper"]
INCLUDE_FILES = ["README.md", "EXPERIMENT_LOG.md", "requirements.txt"]
EXCLUDE_DIRS = {"__pycache__", "x2_check", "x3_check", "x4_check"}
EXCLUDE_EXT = {".pyc"}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for name in INCLUDE_FILES:
        path = os.path.join(ROOT, name)
        if os.path.exists(path):
            z.write(path, name)
    for d in INCLUDE_DIRS:
        base = os.path.join(ROOT, d)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [x for x in dirnames if x not in EXCLUDE_DIRS]
            for fn in filenames:
                if os.path.splitext(fn)[1] in EXCLUDE_EXT:
                    continue
                path = os.path.join(dirpath, fn)
                rel = os.path.relpath(path, ROOT).replace("\\", "/")
                z.write(path, rel)
    n_files = len(z.namelist())
print("saved", OUT)
print("files", n_files)
print("bytes", os.path.getsize(OUT))
