#!/usr/bin/env bash
set -e
export PYTHONPATH=.pydeps
PY=D:/python/python.exe

for s in 2 3; do
  $PY code/train.py --model srcnn --scale $s --iters 20000 --batch 16 --tag srcnn_x$s
  $PY code/train.py --model rcsa  --scale $s --iters 20000 --batch 16 --tag rcsa_x$s
done

$PY code/train.py --model rcsa --scale 4 --iters 20000 --batch 16 --use-se 0 --tag rcsa_nose_x4
$PY code/train.py --model rcsa --scale 4 --iters 20000 --batch 16 --augment 0 --tag rcsa_noaug_x4
