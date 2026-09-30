"""Zayaan's er.dense.merge with the second-opinion cross-encoder added to the dense stacker's features.
Joins ce2_{part}.parquet onto dense_{part}.parquet (adds ce2 and its logit lce2), writes them to <out_dense_dir>,
then calls er.dense.merge unchanged except DENSE_FEATS += ["ce2", "lce2"].
Usage: python merge_ce2.py <dense_dir> <ce2_dir> <out_dense_dir> <out_run>"""
import os
import sys

R = "E:/projects/Amazon ML challenge"
os.environ.setdefault("ER_WORK", R + "/.worktrees/v6val/work_v6")
os.environ.setdefault("ER_DATA", R + "/student_resource/dataset")
os.environ.setdefault("POLARS_MAX_THREADS", "6")
os.environ.setdefault("ER_WORKERS", "6")
sys.path.insert(0, R + "/.worktrees/final2/code/business_entity_resolution/src")

import polars as pl  # noqa: E402

from er import dense  # noqa: E402

dd, cd, od, run = sys.argv[1:5]
os.makedirs(od, exist_ok=True)
K = ["s1_idx", "src", "t_idx"]
cast = lambda d: d.with_columns(pl.col("s1_idx").cast(pl.UInt32), pl.col("src").cast(pl.UInt8), pl.col("t_idx").cast(pl.UInt32))
for part in ("fit1", "val", "test"):
    d = cast(pl.read_parquet(f"{dd}/dense_{part}.parquet"))
    c = cast(pl.read_parquet(f"{cd}/ce2_{part}.parquet")).select(K + ["ce2"])
    d = d.join(c, on=K, how="left")
    miss = int(d["ce2"].null_count())
    assert miss == 0, f"{part}: {miss} pairs without ce2"
    c2 = pl.col("ce2").clip(1e-6, 1 - 1e-6)
    d.with_columns((c2 / (1 - c2)).log().alias("lce2")).write_parquet(f"{od}/dense_{part}.parquet")
dense.DENSE_FEATS = dense.DENSE_FEATS + ["ce2", "lce2"]
print(dense.merge("v6ce", od, run))
