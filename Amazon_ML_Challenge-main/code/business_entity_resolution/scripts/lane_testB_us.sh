#!/bin/bash
# second stage-B worker dedicated to US test partitions (runs in parallel with lane_test3 via locks)
cd "$(dirname "$0")/../src"; PY=../../../.venv/Scripts/python; L=../../../work/lane_testB_us.log; W=../../../work
export PYTHONIOENCODING=utf-8 ER_B_CHUNK=250000 POLARS_MAX_THREADS=8 ER_COUNTRIES=US
echo "=== start $(date +%T), waiting for US stage-1 outputs" >> $L
until [ -f $W/test_P_US_2.parquet ] && [ -f $W/test_P_US_3.parquet ]; do sleep 20; done
echo "=== US P ready $(date +%T)" >> $L
$PY -X faulthandler -m er.run stage_b --split test --run v0 >> $L 2>&1 || { echo "FAILED" >> $L; exit 1; }
echo "US_B_DONE $(date +%T)" >> $L
