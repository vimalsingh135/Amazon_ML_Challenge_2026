#!/bin/bash
# train lane: stage B (train) -> train2 -> tune
cd "$(dirname "$0")/../src"; PY=../../../.venv/Scripts/python; L=../../../work/lane_train.log
export PYTHONIOENCODING=utf-8 POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-10} ER_B_CHUNK=${ER_B_CHUNK:-250000}
$PY -X faulthandler -m er.run stage_b --split train --run v0 >> $L 2>&1 || { echo "FAILED stage_b" >> $L; exit 1; }
$PY -X faulthandler -m er.run train2 --run v0 >> $L 2>&1 || { echo "FAILED train2" >> $L; exit 1; }
$PY -m er.run tune --run v0 >> $L 2>&1 || { echo "FAILED tune" >> $L; exit 1; }
echo "LANE_TRAIN_DONE" >> $L
