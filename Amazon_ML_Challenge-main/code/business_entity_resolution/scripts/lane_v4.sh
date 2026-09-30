#!/bin/bash
# v4 continuation: France admin-part re-normalisation, then the rest of the cascade (sequential).
# Internal work dir work_v31 / run name v31 (started before the rename); submission -> output/v4
cd "$(dirname "$0")/../src"; PY=../../../.venv/Scripts/python; W=../../../work_v31; L=$W/lane_v31.log
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 ER_WORK=$W ER_N_FIT=700000 ER_CAP_PAIR=600 ER_JOIN_BUDGET=10000000 \
       ER_TOP_K=120 ER_TOP_NAME=45 ER_TOP_ADDR=60 ER_TOP_PURE=35 ER_TOP_NUM=35 ER_S1_NEG_MOD=10 \
       POLARS_MAX_THREADS=16 OMP_WAIT_POLICY=PASSIVE ER_EPS1=0.005 ER_B_CHUNK=250000 \
       ER_S2_LR=0.05 ER_S2_LEAVES=255 ER_S2_ROUNDS=3000 ER_S2_EARLY=100 ER_ADMIN_PARTS=$W/aux/admin_parts.json
running() { powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*er.run $1*' -or \$_.CommandLine -like '*er.admin*' }) { exit 0 } else { exit 1 }"; }
run() { echo "=== $* start $(date +%T)" >> $L; $PY -X faulthandler -m er.run "$@" --run v31 >> $L 2>&1 || { echo "FAILED $* $(date +%T)" >> $L; exit 1; }; echo "=== $* done $(date +%T)" >> $L; }
echo "=== v4 continuation waiting $(date +%T)" >> $L
while running "stage_a --split train"; do sleep 20; done
[ -d $W/train_A_US_3 ] && [ -f $W/train_A_US_3/_SUCCESS ] || { echo "FAILED stage_a train incomplete" >> $L; exit 1; }
[ -s $W/aux/admin_parts.json ] || { echo "FAILED admin_parts.json missing" >> $L; exit 1; }
echo "=== renorm France start $(date +%T)" >> $L
$PY -m er.admin renorm --split test --country France >> $L 2>&1 || { echo "FAILED renorm" >> $L; exit 1; }
echo "=== renorm France done $(date +%T)" >> $L
run stage_a --split test
run stage1 --split all
run stage_b --split train
ER_B_MINP1=0.01 run stage_b --split test
run train2
run tune
ER_SUB_NAME=v4 run predict
echo "V4_DONE $(date +%T)" >> $L
