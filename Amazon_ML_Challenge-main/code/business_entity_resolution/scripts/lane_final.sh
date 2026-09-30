#!/bin/bash
# fires predict as soon as the train lane (train2 + tune) and all 6 test stage-B partitions are done
cd "$(dirname "$0")/../src"; PY=../../../.venv/Scripts/python; W=../../../work; L=$W/lane_final.log
export PYTHONIOENCODING=utf-8 ER_SUB_NAME=v2 ER_DECISION=decision_refined.json POLARS_MAX_THREADS=12
echo "=== waiting $(date +%T)" >> $L
until grep -q LANE_TRAIN_DONE $W/lane_train.log 2>/dev/null && [ $(ls -d $W/test_B_*/_SUCCESS 2>/dev/null | wc -l) -ge 6 ]; do sleep 15; done
echo "=== predict start $(date +%T)" >> $L
$PY -X faulthandler -m er.run predict --run v0 >> $L 2>&1; echo "PREDICT_EXIT=$? $(date +%T)" >> $L
