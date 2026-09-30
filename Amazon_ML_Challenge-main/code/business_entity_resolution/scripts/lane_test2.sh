#!/bin/bash
# test lane (continuation): wait for running test stage A worker -> stage A (completes leftovers)
# -> stage 1 (test) -> stage B (test, memory-capped)
cd "$(dirname "$0")/../src"; PY=../../../.venv/Scripts/python; L=../../../work/lane_test.log
export PYTHONIOENCODING=utf-8 ER_JOIN_BUDGET=5000000 ER_B_CHUNK=${ER_B_CHUNK:-250000} POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-12}
running() { powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*er.run $1*' }) { exit 0 } else { exit 1 }"; }
echo "=== lane_test2 start $(date +%T)" >> $L
while running "stage_a --split test"; do sleep 15; done
for st in stage_a stage1 stage_b; do
  $PY -X faulthandler -m er.run $st --split test --run v0 >> $L 2>&1 || { echo "FAILED $st" >> $L; exit 1; }
  echo "=== $st test done $(date +%T)" >> $L
done
echo "LANE_TEST_DONE" >> $L
