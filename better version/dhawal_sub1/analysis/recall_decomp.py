# Run from src/: ER_WORK=../../../work_v4 PYTHONPATH=. python ../scripts/analysis/recall_decomp.py  (per-country recall loss: blocking / stage-1 / model-decision)
import json
import polars as pl
from er import config as C
from er.decide import DecisionParams, select
from er.pipeline import train_roles, labels
W = C.WORK
roles = train_roles().select(pl.col("idx").alias("s1_idx"), "role", "country").filter(pl.col("role") == "val")
lab = labels().join(roles, on="s1_idx")                    # true val pairs (s1_idx, src, t_idx)
A = pl.concat([pl.scan_parquet(f"{W}/train_A_{c}_{s}/*.parquet").select("s1_idx", pl.lit(int(s)).cast(pl.UInt8).alias("src"), "t_idx")
               .join(roles.lazy().select("s1_idx"), on="s1_idx", how="semi").collect() for c in ("US", "India") for s in ("2", "3")])
print("A cols ok", A.height)
val = pl.read_parquet(W / "runs/v5/val_scores.parquet")
b = json.loads((W / "runs/v5/decision.json").read_text())["best"]
sel = select(val, DecisionParams(margin=b["margin"], m0=b["m0"], empty_bias=b["empty_bias"], coh=b["coh"]))
k = ["s1_idx", "src", "t_idx"]
cast = lambda d: d.select(pl.col("s1_idx").cast(pl.UInt32), pl.col("src").cast(pl.UInt8), pl.col("t_idx").cast(pl.UInt32)).with_columns(pl.lit(True).alias("_x"))
lab = lab.with_columns(pl.col("s1_idx").cast(pl.UInt32), pl.col("src").cast(pl.UInt8), pl.col("t_idx").cast(pl.UInt32))
lab = (lab.join(cast(A).unique().rename({"_x": "inA"}), on=k, how="left").join(cast(val).rename({"_x": "inP"}), on=k, how="left")
          .join(cast(sel).rename({"_x": "sel"}), on=k, how="left").fill_null(False))
print(lab.group_by("country").agg(pl.len().alias("true_pairs"), (~pl.col("inA")).mean().alias("miss_blocking"),
      (pl.col("inA") & ~pl.col("inP")).mean().alias("miss_stage1"), (pl.col("inP") & ~pl.col("sel")).mean().alias("miss_model"),
      pl.col("sel").mean().alias("recall")).sort("country"))
# false positives
fp = cast(sel).join(cast(lab.select(k)), on=k, how="anti").join(roles.select(pl.col("s1_idx").cast(pl.UInt32), "country"), on="s1_idx")
print("FP pairs by country", fp.group_by("country").len().sort("country").rows(), "selected", sel.height)
lab.write_parquet(W / "runs/v5/val_recall_decomp.parquet")
