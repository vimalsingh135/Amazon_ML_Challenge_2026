#!/bin/bash
# v3 train lane: stage B (train) -> train2 -> tune
cd "$(dirname "$0")/../src"; PY=../../../.venv/Scripts/python; W=../../../work_v3
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 ER_WORK=../../../work_v3 POLARS_MAX_THREADS=8 OMP_WAIT_POLICY=PASSIVE ER_B_CHUNK=250000 ER_S2_LR=0.1 ER_S2_LEAVES=127 ER_S2_ROUNDS=1500 ER_S2_EARLY=50
L=$W/v3_train.log
for st in "stage_b --split train" "train2" "tune"; do echo "=== $st start $(date +%T)" >> $L; $PY -X faulthandler -m er.run $st --run v3 >> $L 2>&1 || { echo "FAILED $st" >> $L; exit 1; }; echo "=== $st done $(date +%T)" >> $L; done
echo "V3_TRAIN_DONE" >> $L
