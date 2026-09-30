"""Is v6ce's stacked p mis-calibrated for pairs where ONE side lacks a house number?

sibling_loss.py: 73% (India) / 81% (US) of decision FNs have a house number on one side only. The decision
layer is Bayes-optimal only if p is calibrated inside that slice. Here, for val candidates the decision did
NOT select, true-rate vs mean p by (house-number status x exact-name x p bin). A slice with true-rate
well above p (and above the ~0.79 F0.5 break-even) is free F0.5; then the gain of adding it is measured
with the exact macro metric, cross-fitted (rule chosen on one S1 half, scored on the other).
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
from rapidfuzz import fuzz, process  # noqa: E402

from er import config as C  # noqa: E402
from er.decide import DecisionParams, select  # noqa: E402
from er.pipeline import id_maps, train_roles  # noqa: E402
from er.prep import load_gt  # noqa: E402

K = ["s1_idx", "src", "t_idx"]
rd = C.WORK / "runs" / "v6ce"
best = json.loads((rd / "decision_ce.json").read_text())["best"]
PRM = DecisionParams(margin=best["margin"], m0=best["m0"], empty_bias=best["empty_bias"], coh=best["coh"])
val = pl.read_parquet(rd / "val_scores_ce.parquet").select(K + ["p"])
pred = select(val, PRM)
roles = train_roles().filter(pl.col("role") == "val").select(pl.col("idx").alias("s1_idx"), "country")
s1m, tg = id_maps("train")
truth = load_gt().join(s1m.select("s1_idx", "s1"), on="s1").join(tg, on="tid").select(K).join(roles.select("s1_idx"), on="s1_idx")

c1 = ["idx", "name_norm", "hno"]
s1 = pl.read_parquet(C.WORK / "train_s1.parquet", columns=c1).rename({"idx": "s1_idx", "name_norm": "n1", "hno": "h1"})
tt = pl.concat([pl.read_parquet(C.WORK / f"train_s{s}.parquet", columns=c1).with_columns(pl.lit(s, pl.UInt8).alias("src"))
                for s in (2, 3)]).rename({"idx": "t_idx", "name_norm": "n2", "hno": "h2"})
d = (val.join(pred.with_columns(pl.lit(True).alias("sel")), on=K, how="left")
        .join(truth.with_columns(pl.lit(1).alias("y")), on=K, how="left")
        .with_columns(pl.col("sel").fill_null(False), pl.col("y").fill_null(0))
        .join(roles, on="s1_idx").join(s1, on="s1_idx", how="left").join(tt, on=["src", "t_idx"], how="left")
        .with_columns(pl.col("n1", "n2", "h1", "h2").fill_null("").str.strip_chars()))
d = d.with_columns(pl.Series("sn", process.cpdist(d["n1"].to_list(), d["n2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)))
d = d.with_columns(
    pl.when((pl.col("h1") == "") & (pl.col("h2") == "")).then(pl.lit("both_missing"))
      .when((pl.col("h1") == "") | (pl.col("h2") == "")).then(pl.lit("one_missing"))
      .when(pl.col("h1") == pl.col("h2")).then(pl.lit("same")).otherwise(pl.lit("differ")).alias("hno"),
    pl.when(pl.col("sn") >= 100).then(pl.lit("name=100")).when(pl.col("sn") >= 90).then(pl.lit("name90")).otherwise(pl.lit("name<90")).alias("nm"),
    pl.col("p").cut([0.2, 0.4, 0.6, 0.8]).alias("pbin"),
    pl.col("p").rank("ordinal", descending=True).over("s1_idx").alias("rk"),
    pl.col("p").rank("ordinal", descending=True).over("src", "t_idx").alias("trk"))
un = d.filter(~pl.col("sel") & (pl.col("p") >= 0.2))
tab = (un.group_by("hno", "nm", "pbin").agg(pl.len().alias("n"), pl.col("p").mean().round(3).alias("mean_p"),
                                             pl.col("y").mean().round(3).alias("true_rate"))
         .filter(pl.col("n") >= 50).with_columns((pl.col("true_rate") - pl.col("mean_p")).round(3).alias("gap"))
         .sort("gap", descending=True))
pl.Config.set_tbl_rows(80)
print(tab)

# ---- cross-fitted rule: add unselected pairs of a slice (target not selected by another S1) ------
nt = truth.group_by("s1_idx").len("nt")


def macro(sel, s1s):
    tp = sel.join(truth, on=K).group_by("s1_idx").len("tp")
    x = (s1s.join(nt, on="s1_idx", how="left").join(sel.group_by("s1_idx").len("np"), on="s1_idx", how="left")
            .join(tp, on="s1_idx", how="left").fill_null(0))
    f = pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0).otherwise(
        1.25 * pl.col("tp") / (0.25 * pl.col("nt") + pl.col("np")).clip(lower_bound=1e-9))
    return float(x.with_columns(f.alias("f"))["f"].mean())


taken = pred.select("src", "t_idx").unique()
free = un.join(taken, on=["src", "t_idx"], how="anti").filter(pl.col("trk") == 1)
rules = [(h, n, lo) for h in ("one_missing", "both_missing") for n in ("name=100", "name90") for lo in (0.3, 0.4, 0.5, 0.6)]
fold = lambda df: df.with_columns(((pl.col("s1_idx") // 7) % 2).alias("fold"))
rs, fr, pr = fold(roles), fold(free), fold(pred.join(roles.select("s1_idx"), on="s1_idx"))
out = {}
for f in (0, 1):
    tr_s1, te_s1 = rs.filter(pl.col("fold") != f).drop("fold"), rs.filter(pl.col("fold") == f).drop("fold")
    base_tr = pr.filter(pl.col("fold") != f).select(K)
    scores = {}
    for r in rules:
        add = fr.filter((pl.col("fold") != f) & (pl.col("hno") == r[0]) & (pl.col("nm") == r[1]) & (pl.col("p") >= r[2]) & (pl.col("rk") <= 3)).select(K)
        scores[r] = macro(pl.concat([base_tr, add]), tr_s1)
    b0 = macro(base_tr, tr_s1)
    rbest = max(scores, key=scores.get)
    base_te = pr.filter(pl.col("fold") == f).select(K)
    add = fr.filter((pl.col("fold") == f) & (pl.col("hno") == rbest[0]) & (pl.col("nm") == rbest[1]) & (pl.col("p") >= rbest[2]) & (pl.col("rk") <= 3)).select(K)
    out[f"fold{f}"] = {"rule": list(rbest), "train_gain": round(scores[rbest] - b0, 6),
                       "test_base": round(macro(base_te, te_s1), 6), "test_rule": round(macro(pl.concat([base_te, add]), te_s1), 6),
                       "n_added": add.height, "added_true": int(add.join(truth, on=K).height)}
    print(f, out[f"fold{f}"], flush=True)
open(ROOT + "/.worktrees/v6val/out/calib_hno.json", "w").write(json.dumps({"slices": tab.to_dicts(), "rule": out}, indent=1, default=str))
