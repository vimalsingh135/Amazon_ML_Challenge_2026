"""Are v6cd's decision FNs dense-added pairs (scored by the small dense stacker) or original v6ce candidates?"""
import json
import os
import sys

R = "E:/projects/Amazon ML challenge"
os.environ.setdefault("ER_WORK", R + "/.worktrees/v6val/work_v6")
os.environ.setdefault("ER_DATA", R + "/student_resource/dataset")
sys.path.insert(0, R + "/.worktrees/final2/code/business_entity_resolution/src")
import polars as pl  # noqa: E402

from er import config as C  # noqa: E402
from er.decide import DecisionParams, select  # noqa: E402
from er.pipeline import id_maps, train_roles  # noqa: E402
from er.prep import load_gt  # noqa: E402

K = ["s1_idx", "src", "t_idx"]
b = json.loads((C.WORK / "runs/v6cd/decision_ce.json").read_text())["best"]
v = pl.read_parquet(C.WORK / "runs/v6cd/val_scores_ce.parquet").select(K + ["p"])
sel = select(v, DecisionParams(margin=b["margin"], m0=b["m0"], empty_bias=b["empty_bias"], coh=b["coh"]))
base = pl.read_parquet(C.WORK / "runs/v6ce/val_scores_ce.parquet").select(K).with_columns(pl.lit("base_v6ce").alias("origin"))
roles = train_roles().filter(pl.col("role") == "val").select(pl.col("idx").alias("s1_idx"), "country")
s1m, tg = id_maps("train")
truth = load_gt().join(s1m.select("s1_idx", "s1"), on="s1").join(tg, on="tid").select(K).join(roles, on="s1_idx")
fn = (truth.join(v, on=K).join(sel, on=K, how="anti").join(base, on=K, how="left")
          .with_columns(pl.col("origin").fill_null("dense_added")))
print(fn.group_by("country", "origin").agg(pl.len().alias("n"), pl.col("p").median().round(3).alias("median_p"),
                                          (pl.col("p") >= 0.2).sum().alias("p>=0.2")).sort("country", "origin"))
dense = v.join(base, on=K, how="anti").join(truth.with_columns(pl.lit(1).alias("y")), on=K, how="left").with_columns(pl.col("y").fill_null(0))
print("dense-added val pairs:", dense.height, "true:", int(dense["y"].sum()), "selected true:",
      int(dense.join(sel, on=K)["y"].sum()))
