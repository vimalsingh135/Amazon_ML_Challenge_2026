#!/bin/bash
# Resume a two-lane run after its orchestrator was stopped.
#  - waits for any still-running `er.run stage_a/stage1` workers of this run to finish
#  - lane 1: stage_b (train) -> train2 -> tune
#  - lane 2: stage1 (test) -> stage_b (test)        [after test stage A is complete]
#  - then  : predict (writes output/*.tsv and runs the official validator)
# Completed stages are skipped automatically (checkpoints), so it is safe to rerun.
# usage (Git Bash, from the repo root):  bash code/business_entity_resolution/scripts/resume_parallel.sh v0
set -u
RUN=${1:-v0}
cd "$(dirname "$0")/../src"
PY=../../../.venv/Scripts/python
[ -x "$PY" ] || PY=../../../.venv/bin/python
LOG=../../../work
export PYTHONIOENCODING=utf-8 ER_JOIN_BUDGET=${ER_JOIN_BUDGET:-10000000}

running() { powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*er.run $1*' }) { exit 0 } else { exit 1 }"; }
echo "$(date +%T) waiting for running workers (stage1 train / stage_a test)..."
while running "stage1 --split train"; do sleep 20; done
echo "$(date +%T) train stage1 finished -> lane 1"
( $PY -X faulthandler -m er.run stage_b --split train --run $RUN &&
  $PY -X faulthandler -m er.run train2 --run $RUN &&
  $PY -m er.run tune --run $RUN ) >> $LOG/run_${RUN}_lane1.log 2>&1 &
L1=$!
while running "stage_a --split test"; do sleep 20; done
echo "$(date +%T) test stage A finished -> lane 2"
$PY -X faulthandler -m er.run stage_a --split test --run $RUN >> $LOG/run_${RUN}_lane2.log 2>&1   # completes any leftover partition
( $PY -X faulthandler -m er.run stage1 --split test --run $RUN &&
  $PY -X faulthandler -m er.run stage_b --split test --run $RUN ) >> $LOG/run_${RUN}_lane2.log 2>&1 &
L2=$!
wait $L1; E1=$?; wait $L2; E2=$?
echo "$(date +%T) lane1=$E1 lane2=$E2"
[ $E1 -eq 0 ] && [ $E2 -eq 0 ] || { echo "a lane failed - see work/run_${RUN}_lane*.log"; exit 1; }
$PY -m er.run predict --run $RUN 2>&1 | tee -a $LOG/run_${RUN}_final.log | grep -E "PASS|FAIL|predict"
echo "$(date +%T) done -> output/matching_results.tsv"
