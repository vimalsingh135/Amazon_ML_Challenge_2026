#!/bin/bash
# pipelined test lane: repeatedly run stage 1 + stage B on whatever test partitions are ready,
# overlapping with test stage A of the remaining partitions. Exits when all 6 B partitions exist.
cd "$(dirname "$0")/../src"; PY=../../../.venv/Scripts/python; L=../../../work/lane_test.log; W=../../../work
export PYTHONIOENCODING=utf-8 ER_B_CHUNK=${ER_B_CHUNK:-250000} POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-8} ER_EPS1=${ER_EPS1:-0.01} OMP_WAIT_POLICY=PASSIVE ER_WORKERS=${ER_WORKERS:-10} ER_S1_SINGLE=1
echo "=== lane_test3 (pipelined) start $(date +%T)" >> $L
while true; do
  $PY -X faulthandler -m er.run stage1 --split test --run v0 >> $L 2>&1 || { echo "FAILED stage1" >> $L; exit 1; }
  $PY -X faulthandler -m er.run stage_b --split test --run v0 >> $L 2>&1 || { echo "FAILED stage_b" >> $L; exit 1; }
  n=$(ls -d $W/test_B_*/_SUCCESS 2>/dev/null | wc -l)
  echo "=== $(date +%T) test B partitions done: $n/6" >> $L
  [ "$n" -ge 6 ] && break
  sleep 30
done
echo "LANE_TEST_DONE" >> $L
