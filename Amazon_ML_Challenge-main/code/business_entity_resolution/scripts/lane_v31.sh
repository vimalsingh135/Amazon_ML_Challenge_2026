#!/bin/bash
# v3.1 overnight: blocking v5 (fs key, xlarge budgets), 700k fit S1, full-strength stage 2; strictly sequential
cd "$(dirname "$0")/../src"; PY=../../../.venv/Scripts/python; W=../../../work_v31; L=$W/lane_v31.log
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 ER_WORK=$W ER_N_FIT=700000 ER_CAP_PAIR=600 ER_JOIN_BUDGET=10000000 \
       ER_TOP_K=120 ER_TOP_NAME=45 ER_TOP_ADDR=60 ER_TOP_PURE=35 ER_TOP_NUM=35 ER_S1_NEG_MOD=10 \
       POLARS_MAX_THREADS=16 OMP_WAIT_POLICY=PASSIVE ER_EPS1=0.005 ER_B_CHUNK=250000 \
       ER_S2_LR=0.05 ER_S2_LEAVES=255 ER_S2_ROUNDS=3000 ER_S2_EARLY=100
run() { echo "=== $* start $(date +%T)" >> $L; $PY -X faulthandler -m er.run "$@" --run v31 >> $L 2>&1 || { echo "FAILED $* $(date +%T)" >> $L; exit 1; }; echo "=== $* done $(date +%T)" >> $L; }
run stage_a --split train
run stage_a --split test
run stage1 --split all
run stage_b --split train
ER_B_MINP1=0.01 run stage_b --split test
run train2
run tune
ER_SUB_NAME=v31 run predict
echo "V31_DONE $(date +%T)" >> $L
