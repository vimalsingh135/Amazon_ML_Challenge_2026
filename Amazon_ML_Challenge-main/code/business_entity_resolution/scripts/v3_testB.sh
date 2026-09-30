#!/bin/bash
# v3 test stage-B worker for countries $1 (comma list): waits for their stage-1 outputs, p1 floor 0.01
cd "$(dirname "$0")/../src"; PY=../../../.venv/Scripts/python; W=../../../work_v3
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 ER_WORK=../../../work_v3 POLARS_MAX_THREADS=8 OMP_WAIT_POLICY=PASSIVE ER_B_CHUNK=250000 ER_S2_LR=0.1 ER_S2_LEAVES=127 ER_S2_ROUNDS=1500 ER_S2_EARLY=50
export ER_COUNTRIES=$1 ER_B_MINP1=0.01
L=$W/v3_testB_${1//,/_}.log
echo "=== start $(date +%T)" >> $L
for c in ${1//,/ }; do until [ -f $W/test_P_${c}_2.parquet ] && [ -f $W/test_P_${c}_3.parquet ]; do sleep 15; done; echo "=== $c P ready $(date +%T)" >> $L; $PY -X faulthandler -m er.run stage_b --split test --run v3 >> $L 2>&1 || { echo "FAILED $c" >> $L; exit 1; }; done
echo "TESTB_DONE $(date +%T)" >> $L
