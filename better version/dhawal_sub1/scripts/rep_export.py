"""Replicate v6cd inputs locally with Zayaan's final2 code (unchanged): bi-encoder training pairs + dense-kernel data.
Outputs under .worktrees/rep/: bi_data/bi_train.parquet, dense_data/*, ce_data/ce_train.parquet (copied)."""
import os
import shutil
import subprocess
import sys

R = "E:/projects/Amazon ML challenge"
W = R + "/.worktrees/v6val/work_v6"
SRC = R + "/.worktrees/final2/code/business_entity_resolution/src"
REP = R + "/.worktrees/rep"
PY = R + "/.venv/Scripts/python.exe"
env = dict(os.environ, ER_WORK=W, ER_DATA=R + "/student_resource/dataset", POLARS_MAX_THREADS="6", ER_WORKERS="6",
           PYTHONUTF8="1", PYTHONPATH=SRC)
for d in ("bi_data", "dense_data", "ce_data"):
    os.makedirs(f"{REP}/{d}", exist_ok=True)

# candidate keys of the v6ce run in the layout er.dense export reads (fit_oof / val_scores / test_scores)
rd = f"{W}/runs/v6ce_dx"
os.makedirs(rd, exist_ok=True)
for s, d in ((f"{W}/runs/v6/fit_oof.parquet", "fit_oof.parquet"), (f"{W}/runs/v6ce/val_scores_ce.parquet", "val_scores.parquet"),
             (f"{W}/runs/v6ce/test_scores_ce.parquet", "test_scores.parquet")):
    if not os.path.exists(f"{rd}/{d}"):
        shutil.copyfile(s, f"{rd}/{d}")
if not os.path.exists(f"{REP}/ce_data/ce_train.parquet"):
    shutil.copyfile(R + "/.worktrees/v6val/ce_data/ce_train.parquet", f"{REP}/ce_data/ce_train.parquet")


def run(*a):
    print(">>", " ".join(a), flush=True)
    r = subprocess.run([PY, "-m", *a], cwd=SRC, env=env)
    if r.returncode:
        sys.exit(f"FAILED {a} rc={r.returncode}")


if not os.path.exists(f"{REP}/bi_data/bi_train.parquet"):
    run("er.cross", "export-biencoder", "--out", f"{REP}/bi_data")
if not os.path.exists(f"{REP}/dense_data/cand_test.parquet"):
    run("er.dense", "export", "--run", "v6ce_dx", "--out", f"{REP}/dense_data")
for d in ("bi_data", "dense_data", "ce_data"):
    for f in sorted(os.listdir(f"{REP}/{d}")):
        print(d, f, round(os.path.getsize(f"{REP}/{d}/{f}") / 2**20, 1), "MB")
print("EXPORT_DONE", flush=True)
