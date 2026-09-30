"""Run a Zayaan er.* module (final2 code) with the local work env, e.g.:
python er_run.py SUBNAME er.run predict --run v6ce --scores ce     (SUBNAME -> ER_SUB_NAME, '-' for none)"""
import os
import subprocess
import sys

R = "E:/projects/Amazon ML challenge"
SRC = R + "/.worktrees/final2/code/business_entity_resolution/src"
env = dict(os.environ, ER_WORK=R + "/.worktrees/v6val/work_v6", ER_DATA=R + "/student_resource/dataset",
           ER_OUT=R + "/.worktrees/rep/output", POLARS_MAX_THREADS="6", ER_WORKERS="6", PYTHONUTF8="1",
           PYTHONIOENCODING="utf-8", PYTHONPATH=SRC)
if sys.argv[1] != "-":
    env["ER_SUB_NAME"] = sys.argv[1]
r = subprocess.run([sys.executable, "-m", *sys.argv[2:]], cwd=SRC, env=env)
print("RC", r.returncode, flush=True)
sys.exit(r.returncode)
