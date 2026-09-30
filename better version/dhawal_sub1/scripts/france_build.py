"""France dense variant with Zayaan's recipe: er.dense add-test (score new France pairs with the dense stacker of
<stacker_run>) -> his france_dense_filter logic (same house number AND street fuzzy >= 70) -> v6ce decision
(as restore.py sets for v6cef2) -> er.run predict. Output dir: .worktrees/rep/output/<tag> (France rows are the probe).
Usage: python france_build.py <stacker_run> <dense_france_test.parquet> <tag>"""
import json
import os
import shutil
import subprocess
import sys

import polars as pl
from rapidfuzz import fuzz

R = "E:/projects/Amazon ML challenge"
WT = R + "/.worktrees"
W = WT + "/v6val/work_v6"
SRC = WT + "/final2/code/business_entity_resolution/src"
stack_run, dense_file, tag = sys.argv[1:4]
env = dict(os.environ, ER_WORK=W, ER_DATA=R + "/student_resource/dataset", ER_OUT=WT + "/rep/output",
           POLARS_MAX_THREADS="6", ER_WORKERS="6", PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONPATH=SRC)
run = lambda *a, **kw: subprocess.run([sys.executable, "-m", *a], cwd=SRC, env={**env, **kw}, check=True)

raw, filt = f"v6cef_{tag}", f"v6cef2_{tag}"
if not os.path.exists(f"{W}/runs/{raw}/test_scores_ce.parquet"):
    run("er.dense", "add-test", "--run", stack_run, "--dense", dense_file, "--out-run", raw)

K = ["s1_idx", "src", "t_idx"]
cast = lambda d: d.with_columns(pl.col("s1_idx").cast(pl.UInt32), pl.col("src").cast(pl.UInt8), pl.col("t_idx").cast(pl.UInt32))
base = cast(pl.read_parquet(f"{W}/runs/v6ce/test_scores_ce.parquet"))
new = cast(pl.read_parquet(f"{W}/runs/{raw}/test_scores_ce.parquet")).join(base, on=K, how="anti")
s1 = pl.scan_parquet(f"{W}/test_s1.parquet").filter(pl.col("country") == "France").select(pl.col("idx").cast(pl.UInt32).alias("s1_idx"), "hno", pl.col("street").list.join(" ").alias("st1")).collect()
tg = pl.concat([pl.scan_parquet(f"{W}/test_s{s}.parquet").filter(pl.col("country") == "France").select(pl.col("idx").cast(pl.UInt32).alias("t_idx"), pl.lit(s).cast(pl.UInt8).alias("src"), pl.col("hno").alias("h2"), pl.col("street").list.join(" ").alias("st2")).collect() for s in (2, 3)])
e = new.join(s1, on="s1_idx").join(tg, on=["src", "t_idx"]).filter((pl.col("hno") != "") & (pl.col("hno") == pl.col("h2")))
e = e.with_columns(pl.Series("ss", [fuzz.token_set_ratio(a or "", b or "") for a, b in zip(e["st1"].to_list(), e["st2"].to_list())])).filter(pl.col("ss") >= 70)
keep = e.select(*K, "p")
od = f"{W}/runs/{filt}"
os.makedirs(od, exist_ok=True)
for f in os.listdir(f"{W}/runs/{raw}"):
    if f not in ("test_scores_ce.parquet", "decision.json", "decision_ce.json") and not os.path.exists(f"{od}/{f}"):
        os.link(f"{W}/runs/{raw}/{f}", f"{od}/{f}")
for f in ("decision.json", "decision_ce.json"):   # v6ce decision, as restore.py sets for v6cef2
    shutil.copyfile(f"{W}/runs/v6ce/decision_ce.json", f"{od}/{f}")
pl.concat([base.select(*K, "p"), keep]).write_parquet(f"{od}/test_scores_ce.parquet")
info = {"france_dense_new": new.height, "kept_same_number_same_street": keep.height, "kept_p>0.5": int((keep["p"] > 0.5).sum())}
print(info, flush=True)
json.dump(info, open(f"{WT}/v6val/out/france_build_{tag}.json", "w"))
run("er.run", "predict", "--run", filt, "--scores", "ce", ER_SUB_NAME=filt)
print("OUTPUT", f"{WT}/rep/output/{filt}", flush=True)
