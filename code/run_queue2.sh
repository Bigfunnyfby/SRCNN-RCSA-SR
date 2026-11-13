#!/usr/bin/env bash
set -e
export PYTHONPATH=.pydeps
PY=D:/python/python.exe

# x3 baseline (x2 baseline already trained)
$PY code/train.py --model srcnn --scale 3 --iters 20000 --batch 16 --tag srcnn_x3

# proposed model at x2/x3 with the selected residual scale
$PY code/train.py --model rcsa --scale 2 --iters 20000 --batch 16 --blocks 4 --channels 64 --res-scale 0.5 --tag rcsa_x2
$PY code/train.py --model rcsa --scale 3 --iters 20000 --batch 16 --blocks 4 --channels 64 --res-scale 0.5 --tag rcsa_x3

# x4 ablations, same residual scale as the full model
$PY code/train.py --model rcsa --scale 4 --iters 20000 --batch 16 --blocks 4 --channels 64 --res-scale 0.5 --use-se 0 --tag rcsa_nose_x4
$PY code/train.py --model rcsa --scale 4 --iters 20000 --batch 16 --blocks 4 --channels 64 --res-scale 0.5 --augment 0 --tag rcsa_noaug_x4
