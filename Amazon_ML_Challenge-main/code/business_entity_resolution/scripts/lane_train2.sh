#!/bin/bash
# fast train lane: train2 (lr 0.1, 127 leaves, early stop 50) -> tune
cd "$(dirname "$0")/../src"; PY=../../../.venv/Scripts/python; L=../../../work/lane_train.log
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 ER_S2_LR=0.1 ER_S2_LEAVES=127 ER_S2_ROUNDS=1500 ER_S2_EARLY=50 ER_WORKERS=12 OMP_WAIT_POLICY=PASSIVE
echo "=== fast train2 start $(date +%T)" >> $L
$PY -X faulthandler -m er.run train2 --run v0 >> $L 2>&1 || { echo "FAILED train2" >> $L; exit 1; }
$PY -m er.run tune --run v0 >> $L 2>&1 || { echo "FAILED tune" >> $L; exit 1; }
echo "LANE_TRAIN_DONE" >> $L
