#!/bin/bash
# second stage-B worker dedicated to US test partitions (runs in parallel with lane_test3 via locks)
cd "$(dirname "$0")/../src"; PY=../../../.venv/Scripts/python; L=../../../work/lane_testB_fi.log; W=../../../work
export PYTHONIOENCODING=utf-8 ER_B_CHUNK=250000 POLARS_MAX_THREADS=8 ER_COUNTRIES=France,India
echo "=== start $(date +%T), waiting for FR/IN stage-1 outputs" >> $L
until [ -f $W/test_P_France_2.parquet ] && [ -f $W/test_P_France_3.parquet ]; do sleep 20; done
echo "=== US P ready $(date +%T)" >> $L
$PY -X faulthandler -m er.run stage_b --split test --run v0 >> $L 2>&1 || { echo "FAILED" >> $L; exit 1; }
echo "FI_B_DONE $(date +%T)" >> $L
# India's stage-1 outputs may land after France is done: run stage B again to pick them up
until [ -f $W/test_P_India_2.parquet ] && [ -f $W/test_P_India_3.parquet ]; do sleep 20; done
$PY -X faulthandler -m er.run stage_b --split test --run v0 >> $L 2>&1 || { echo "FAILED India" >> $L; exit 1; }
echo "FI_B_DONE2 $(date +%T)" >> $L
