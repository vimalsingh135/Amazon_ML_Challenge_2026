"""Smoke test for merge_extra.py: tiny fake dense_{fit1,val,test} built from real candidate keys, run the lex merge
into a throwaway run, check it writes scores, then delete it."""
import os
import shutil
import subprocess
import sys

import numpy as np
import polars as pl

R = "E:/projects/Amazon ML challenge"
W = R + "/.worktrees/v6val/work_v6"
D = R + "/.worktrees/rep/_smoke_dense"
os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(0)
src = {"fit1": f"{W}/runs/v6/fit_oof.parquet", "val": f"{W}/runs/v6ce/val_scores_ce.parquet", "test": f"{W}/runs/v6ce/test_scores_ce.parquet"}
for part, f in src.items():
    d = pl.read_parquet(f).select("s1_idx", "src", "t_idx").sample(3000, seed=1)
    n = d.height
    d = d.with_columns(pl.Series("cos", rng.uniform(0.6, 1, n).astype("float32")), pl.Series("rank", rng.integers(1, 11, n).astype("int16")),
                       pl.Series("gap1", rng.uniform(0, 0.2, n).astype("float32")), pl.Series("ce", rng.uniform(0, 1, n).astype("float32")),
                       pl.Series("y", rng.integers(0, 2, n).astype("int8")))
    d.write_parquet(f"{D}/dense_{part}.parquet")
r = subprocess.run([sys.executable, R + "/.worktrees/v6val/tmp/merge_extra.py", D, D + "_lex", "zz_smoke", "lex"],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
print(r.stdout[-1500:], r.stderr[-1500:])
out = f"{W}/runs/zz_smoke/val_scores_ce.parquet"
print("OK" if os.path.exists(out) and r.returncode == 0 else "FAILED", flush=True)
shutil.rmtree(f"{W}/runs/zz_smoke", ignore_errors=True)
shutil.rmtree(D, ignore_errors=True)
shutil.rmtree(D + "_lex", ignore_errors=True)
