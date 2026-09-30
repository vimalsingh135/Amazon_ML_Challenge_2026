#!/bin/bash
# second v3 test stage-B worker: starts once train2 has finished (frees memory), shares partitions via locks
cd "$(dirname "$0")/../src"; PY=../../../.venv/Scripts/python; W=../../../work_v3; L=$W/v3_testB_w2.log
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 ER_WORK=../../../work_v3 POLARS_MAX_THREADS=8 ER_B_CHUNK=250000 ER_B_MINP1=0.01
echo "=== waiting for train2 $(date +%T)" >> $L
until grep -q "=== train2 done" $W/v3_train.log 2>/dev/null; do sleep 15; done
echo "=== start $(date +%T)" >> $L
$PY -X faulthandler -m er.run stage_b --split test --run v3 >> $L 2>&1 || { echo "FAILED" >> $L; exit 1; }
echo "W2_DONE $(date +%T)" >> $L
