#!/bin/bash
# submission #1: stage-1 matcher + tuned exact-F0.5 decision, as soon as all 6 test P files exist
cd "$(dirname "$0")/../src"; PY=../../../.venv/Scripts/python; W=../../../work; L=$W/lane_quick.log
export PYTHONIOENCODING=utf-8 POLARS_MAX_THREADS=8
echo "=== waiting for test P $(date +%T)" >> $L
until [ $(ls $W/test_P_*.parquet 2>/dev/null | wc -l) -ge 6 ]; do sleep 10; done
echo "=== quick start $(date +%T)" >> $L
$PY -m er.quick --run v0s1 >> $L 2>&1; echo "QUICK_EXIT=$? $(date +%T)" >> $L
