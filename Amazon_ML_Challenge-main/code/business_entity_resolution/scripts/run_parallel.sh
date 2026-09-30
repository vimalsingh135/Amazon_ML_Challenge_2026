#!/bin/bash
# Two-lane orchestration (same computations as `er.run all`, overlapped in time):
#   lane 1: train stage A -> stage 1 (train) -> stage B (train) -> train2 -> tune
#   lane 2: test  stage A
#   then  : stage 1 (test) -> stage B (test) -> predict
# usage: scripts/run_parallel.sh <run-name>
set -u
RUN=${1:-v0}
cd "$(dirname "$0")/../src"
PY=../../../.venv/Scripts/python
[ -x "$PY" ] || PY=../../../.venv/bin/python
LOG=../../../work
export PYTHONIOENCODING=utf-8 ER_JOIN_BUDGET=${ER_JOIN_BUDGET:-10000000}
( for st in stage_a stage1 stage_b; do $PY -X faulthandler -m er.run $st --split train --run $RUN || exit 1; done
  $PY -X faulthandler -m er.run train2 --run $RUN && $PY -m er.run tune --run $RUN ) > $LOG/run_${RUN}_lane1.log 2>&1 &
L1=$!
$PY -X faulthandler -m er.run stage_a --split test --run $RUN > $LOG/run_${RUN}_lane2.log 2>&1 &
L2=$!
wait $L1; E1=$?; wait $L2; E2=$?
echo "lane1=$E1 lane2=$E2" >> $LOG/run_${RUN}_lane1.log
[ $E1 -eq 0 ] && [ $E2 -eq 0 ] || exit 1
for st in stage1 stage_b; do $PY -X faulthandler -m er.run $st --split test --run $RUN >> $LOG/run_${RUN}_final.log 2>&1 || exit 1; done
$PY -m er.run predict --run $RUN >> $LOG/run_${RUN}_final.log 2>&1
echo "exit=$?" >> $LOG/run_${RUN}_final.log
