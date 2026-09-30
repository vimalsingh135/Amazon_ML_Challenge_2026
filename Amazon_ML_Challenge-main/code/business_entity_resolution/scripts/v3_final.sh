#!/bin/bash
# v3 predict once training/tuning and all 6 test stage-B partitions are complete
cd "$(dirname "$0")/../src"; PY=../../../.venv/Scripts/python; W=../../../work_v3
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 ER_WORK=../../../work_v3 POLARS_MAX_THREADS=8 OMP_WAIT_POLICY=PASSIVE ER_B_CHUNK=250000 ER_S2_LR=0.1 ER_S2_LEAVES=127 ER_S2_ROUNDS=1500 ER_S2_EARLY=50
L=$W/v3_final.log; echo "=== waiting $(date +%T)" >> $L
until grep -q V3_TRAIN_DONE $W/v3_train.log 2>/dev/null && [ $(ls -d $W/test_B_*/_SUCCESS 2>/dev/null | wc -l) -ge 6 ]; do sleep 15; done
echo "=== predict start $(date +%T)" >> $L
ER_SUB_NAME=v3 POLARS_MAX_THREADS=16 $PY -X faulthandler -m er.run predict --run v3 >> $L 2>&1; echo "PREDICT_EXIT=$? $(date +%T)" >> $L
