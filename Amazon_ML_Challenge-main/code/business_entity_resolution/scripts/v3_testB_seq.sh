#!/bin/bash
# waits for the stage-1 worker to exit, then runs test stage B for all countries (p1 floor 0.01)
cd "$(dirname "$0")/../src"; PY=../../../.venv/Scripts/python; W=../../../work_v3; L=$W/v3_testB_seq.log
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 ER_WORK=../../../work_v3 POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-10} ER_B_CHUNK=250000 ER_B_MINP1=0.01
running() { powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*er.run $1*' }) { exit 0 } else { exit 1 }"; }
echo "=== waiting for stage1 $(date +%T)" >> $L
while running "stage1 --split all"; do sleep 20; done
echo "=== test stage B start $(date +%T)" >> $L
$PY -X faulthandler -m er.run stage_b --split test --run v3 >> $L 2>&1 || { echo "FAILED" >> $L; exit 1; }
echo "TESTB_DONE $(date +%T)" >> $L
