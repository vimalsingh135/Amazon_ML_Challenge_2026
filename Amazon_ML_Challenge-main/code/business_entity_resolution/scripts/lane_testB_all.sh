#!/bin/bash
# single sequential test stage-B worker (all countries), memory-capped
cd "$(dirname "$0")/../src"; PY=../../../.venv/Scripts/python; L=../../../work/lane_testB_all.log
export PYTHONIOENCODING=utf-8 ER_B_CHUNK=250000 POLARS_MAX_THREADS=12
echo "=== start $(date +%T)" >> $L
$PY -X faulthandler -m er.run stage_b --split test --run v0 >> $L 2>&1 || { echo "FAILED" >> $L; exit 1; }
echo "TESTB_DONE $(date +%T)" >> $L
