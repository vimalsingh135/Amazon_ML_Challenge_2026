#!/bin/bash
# test lane: stage A (remaining partitions) -> stage 1 (test) -> stage B (test)
cd "$(dirname "$0")/../src"; PY=../../../.venv/Scripts/python; L=../../../work/lane_test.log
export PYTHONIOENCODING=utf-8 ER_JOIN_BUDGET=5000000
for st in stage_a stage1 stage_b; do
  $PY -X faulthandler -m er.run $st --split test --run v0 >> $L 2>&1 || { echo "FAILED $st" >> $L; exit 1; }
done
echo "LANE_TEST_DONE" >> $L
