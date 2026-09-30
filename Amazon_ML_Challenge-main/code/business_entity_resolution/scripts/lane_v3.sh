#!/bin/bash
# v3: blocking v4 -> full cascade, strictly sequential (one heavy job at a time), own work dir
cd "$(dirname "$0")/../src"; PY=../../../.venv/Scripts/python; L=../../../work_v3/lane_v3.log
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 ER_WORK=../../../work_v3 ER_CAP_PAIR=600 ER_JOIN_BUDGET=10000000 \
       POLARS_MAX_THREADS=16 OMP_WAIT_POLICY=PASSIVE ER_EPS1=0.005 ER_B_CHUNK=250000 \
       ER_S2_LR=0.1 ER_S2_LEAVES=127 ER_S2_ROUNDS=1500 ER_S2_EARLY=50
run() { echo "=== $* start $(date +%T)" >> $L; $PY -X faulthandler -m er.run "$@" --run v3 >> $L 2>&1 || { echo "FAILED $* $(date +%T)" >> $L; exit 1; }; echo "=== $* done $(date +%T)" >> $L; }
run stage_a --split train
run stage_a --split test
run stage1 --split all
run stage_b --split train
run stage_b --split test
run train2
run tune
ER_SUB_NAME=v3 run predict
echo "V3_DONE $(date +%T)" >> $L
