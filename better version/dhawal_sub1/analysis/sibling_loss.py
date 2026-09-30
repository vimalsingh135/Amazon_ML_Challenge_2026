"""Loss anatomy of v6ce on val S1 that have SIBLING businesses (France's dominant structure, 21% of French S1).

Sibling S1 = another S1 of the same country with the same brand (first core token), same street tokens
and same locality but a different house number (Zayaan's definition, README 2026-09-26). India val is
the labelled proxy (7% of India S1). For sibling vs non-sibling S1, per country:
  macro F0.5, and where the loss goes: retrieval miss / decision FN / FP, with each FP and FN split by
  whether the S1 and target house numbers agree (FP at a different number = picked the sibling;
  FN at a different number = noise changed the number and the model trusted it).
Usage: python sibling_loss.py
"""
import json
import os
import sys

ROOT = "E:/projects/Amazon ML challenge"
os.environ.setdefault("ER_WORK", ROOT + "/.worktrees/v6val/work_v6")
os.environ.setdefault("ER_DATA", ROOT + "/student_resource/dataset")
os.environ.setdefault("POLARS_MAX_THREADS", "6")
sys.path.insert(0, ROOT + "/.worktrees/ce/code/business_entity_resolution/src")

import polars as pl  # noqa: E402

from er import config as C  # noqa: E402
from er.decide import DecisionParams, select  # noqa: E402
from er.pipeline import id_maps, train_roles  # noqa: E402
from er.prep import load_gt  # noqa: E402

K = ["s1_idx", "src", "t_idx"]
rd = C.WORK / "runs" / "v6ce"
best = json.loads((rd / "decision_ce.json").read_text())["best"]
val = pl.read_parquet(rd / "val_scores_ce.parquet").select(K + ["p"])
pred = select(val, DecisionParams(margin=best["margin"], m0=best["m0"], empty_bias=best["empty_bias"], coh=best["coh"]))

roles = train_roles().filter(pl.col("role") == "val").select(pl.col("idx").alias("s1_idx"), "country")
s1m, tg = id_maps("train")
truth = load_gt().join(s1m.select("s1_idx", "s1"), on="s1").join(tg, on="tid").select(K).join(roles.select("s1_idx"), on="s1_idx")

# ---- sibling flag over ALL train S1 of a country (siblings may sit in fit, not val) ----------------
s1 = pl.read_parquet(C.WORK / "train_s1.parquet", columns=["idx", "country", "core", "street", "loc", "hno"]).rename({"idx": "s1_idx"})
s1 = s1.with_columns(pl.col("core").list.first().alias("brand"),
                     pl.col("street").list.sort().list.join(" ").alias("st"),
                     pl.col("loc").list.sort().list.join(" ").alias("lc"))
key = ["country", "brand", "st", "lc"]
g = s1.filter(pl.col("brand").is_not_null() & (pl.col("st") != "") & pl.col("hno").is_not_null())
g = g.with_columns(pl.col("hno").n_unique().over(key).alias("n_hno"))
sib = g.filter(pl.col("n_hno") >= 2).select("s1_idx").with_columns(pl.lit(True).alias("sib"))
roles = roles.join(sib, on="s1_idx", how="left").with_columns(pl.col("sib").fill_null(False))
print(roles.group_by("country").agg(pl.col("sib").mean().round(4).alias("sib_share"), pl.len()), flush=True)

# ---- house-number agreement for every pair we look at ------------------------------------------
hn1 = s1.select("s1_idx", pl.col("hno").alias("h1"))
tgt = pl.concat([pl.read_parquet(C.WORK / f"train_s{s}.parquet", columns=["idx", "hno"])
                   .with_columns(pl.lit(s, pl.UInt8).alias("src")) for s in (2, 3)]).rename({"idx": "t_idx", "hno": "h2"})


def hflag(df):
    from rapidfuzz.distance import Levenshtein
    d = df.join(hn1, on="s1_idx", how="left").join(tgt, on=["src", "t_idx"], how="left").with_columns(
        pl.col("h1").fill_null("").str.strip_chars(), pl.col("h2").fill_null("").str.strip_chars())
    ed = [Levenshtein.distance(a, b) for a, b in zip(d["h1"].to_list(), d["h2"].to_list())]
    return (d.with_columns(pl.Series("ed", ed))
             .with_columns(pl.when((pl.col("h1") == "") & (pl.col("h2") == "")).then(pl.lit("both_missing"))
                             .when((pl.col("h1") == "") | (pl.col("h2") == "")).then(pl.lit("one_missing"))
                             .when(pl.col("ed") == 0).then(pl.lit("same"))
                             .when(pl.col("ed") == 1).then(pl.lit("edit1")).otherwise(pl.lit("diff")).alias("hno"))
             .drop("h1", "h2", "ed"))


cand = val.select(K)
T = (truth.join(cand.with_columns(pl.lit(True).alias("in_c")), on=K, how="left")
          .join(pred.with_columns(pl.lit(True).alias("in_p")), on=K, how="left").fill_null(False))
P = pred.join(truth.with_columns(pl.lit(True).alias("is_t")), on=K, how="left").fill_null(False)
miss = hflag(T.filter(~pl.col("in_c"))).with_columns(pl.lit("retrieval_miss").alias("err"))
fn = hflag(T.filter(pl.col("in_c") & ~pl.col("in_p"))).with_columns(pl.lit("decision_fn").alias("err"))
fp = hflag(P.filter(~pl.col("is_t"))).with_columns(pl.lit("fp").alias("err"))
errs = pl.concat([x.select("s1_idx", "err", "hno") for x in (miss, fn, fp)]).join(roles, on="s1_idx")

# ---- macro F0.5 per group ---------------------------------------------------------------------
nt = truth.group_by("s1_idx").len("nt")
tp = P.filter("is_t").group_by("s1_idx").len("tp")
d = (roles.join(nt, on="s1_idx", how="left").join(pred.group_by("s1_idx").len("np"), on="s1_idx", how="left")
          .join(tp, on="s1_idx", how="left").fill_null(0))
d = d.with_columns(pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0).otherwise(
    1.25 * pl.col("tp") / (0.25 * pl.col("nt") + pl.col("np")).clip(lower_bound=1e-9)).alias("f"))
tot_loss = float((1 - d["f"]).sum())
fs = d.group_by("country", "sib").agg(pl.len().alias("n_s1"), pl.col("f").mean().round(5).alias("f05"),
                                      ((1 - pl.col("f")).sum() / tot_loss).round(4).alias("loss_share")).sort("country", "sib")
et = (errs.group_by("country", "sib", "err", "hno").len("n")
          .join(roles.group_by("country", "sib").len("n_s1"), on=["country", "sib"])
          .with_columns((1000 * pl.col("n") / pl.col("n_s1")).round(2).alias("per_1k_s1")).drop("n_s1")
          .sort("country", "sib", "err", "hno"))
pl.Config.set_tbl_rows(60)
print(fs)
print(et)
open(ROOT + "/.worktrees/v6val/out/sibling_loss.json", "w").write(
    json.dumps({"f": fs.to_dicts(), "errors": et.to_dicts()}, indent=1, default=str))
