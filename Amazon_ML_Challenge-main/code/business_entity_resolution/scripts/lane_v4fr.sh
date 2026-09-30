#!/bin/bash
# after v4: France self-training (pseudo-labels from v4 test scores) -> retrain -> tune -> gate vs v4 -> output/v4_fr
cd "$(dirname "$0")/../src"; PY=../../../.venv/Scripts/python; W=../../../work_v31; L=$W/lane_v4fr.log
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 ER_WORK=$W ER_N_FIT=700000 POLARS_MAX_THREADS=16 OMP_WAIT_POLICY=PASSIVE \
       ER_S2_LR=0.05 ER_S2_LEAVES=255 ER_S2_ROUNDS=3000 ER_S2_EARLY=100
echo "=== waiting for v4 $(date +%T)" >> $L
until grep -q "V4_DONE\|FAILED" $W/lane_v31.log 2>/dev/null; do sleep 30; done
grep -q "V4_DONE" $W/lane_v31.log || { echo "v4 failed - not running" >> $L; exit 1; }
step() { echo "=== $1 start $(date +%T)" >> $L; shift; "$@" >> $L 2>&1 || { echo "FAILED $(date +%T)" >> $L; exit 1; }; }
step pseudo $PY -m er.selftrain pseudo --run v31 --country France
mkdir -p $W/runs/v4fr
step train2 env ER_PSEUDO=$W/runs/v31/pseudo_France.parquet $PY -X faulthandler -m er.run train2 --run v4fr
step tune $PY -m er.run tune --run v4fr
step gate $PY -m er.compare --a $W/runs/v31 --b $W/runs/v4fr --work-a $W --work-b $W
step predict env ER_SUB_NAME=v4_fr $PY -X faulthandler -m er.run predict --run v4fr
echo "V4FR_DONE $(date +%T)" >> $L
