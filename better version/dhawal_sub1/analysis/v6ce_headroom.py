"""Loss anatomy + oracle ceilings for a run's validation predictions (default: v6ce).

Where does the remaining macro-F0.5 go? For every val S1:
    T = true targets, C = candidate pairs the run scored, P = pairs the decision selected.
    retrieval miss = T \\ C (never scored -> only better blocking/pruning can fix)
    decision FN    = (T & C) \\ P (scored but not selected -> better scores/decision can fix)
    FP             = P \\ T
Oracles (same candidates):
    dec_perfect = T & C        -> ceiling of any scoring/decision improvement WITHOUT new recall
    no_fp       = P & T        -> value of removing every false positive
    all_fn      = P | (T & C)  -> value of recovering every decision FN
Usage (ER_WORK / ER_DATA set, src on PYTHONPATH):  python v6ce_headroom.py [run] [out.json]
"""
import json
import sys

import numpy as np
import polars as pl

from er import config as C
from er.decide import DecisionParams, select, select_exact
from er.pipeline import id_maps, train_roles
from er.prep import load_gt

RUN = sys.argv[1] if len(sys.argv) > 1 else "v6ce"
OUT = sys.argv[2] if len(sys.argv) > 2 else None
K = ["s1_idx", "src", "t_idx"]

rd = C.WORK / "runs" / RUN
best = json.loads((rd / "decision.json").read_text())["best"]
val = pl.read_parquet(rd / "val_scores.parquet").select(K + ["p"])
fn = select_exact if best.get("kind") == "exact" else select
prm = DecisionParams(margin=best["margin"], m0=best["m0"], empty_bias=best["empty_bias"], coh=best.get("coh", 0.0))
pred = fn(val, prm).select(K)

roles = train_roles().filter(pl.col("role") == "val").select(pl.col("idx").alias("s1_idx"), "country")
s1_ids, tg_ids = id_maps("train")
truth = (load_gt().join(s1_ids.select("s1_idx", "s1"), on="s1")
         .join(tg_ids, on="tid").select(K).join(roles.select("s1_idx"), on="s1_idx"))
cand = val.select(K)


def flag(df, other, name):
    return df.join(other.with_columns(pl.lit(True).alias(name)), on=K, how="left").with_columns(
        pl.col(name).fill_null(False))


T = flag(flag(truth, cand, "in_c"), pred, "in_p")
P = flag(pred, truth, "is_t")


def macro(pairs_sel, name):
    """macro F0.5 over all val S1 for a selected-pair set."""
    tp = pairs_sel.join(truth, on=K, how="inner").group_by("s1_idx").len("tp")
    np_ = pairs_sel.group_by("s1_idx").len("np")
    nt = truth.group_by("s1_idx").len("nt")
    d = (roles.join(nt, on="s1_idx", how="left").join(np_, on="s1_idx", how="left")
              .join(tp, on="s1_idx", how="left").fill_null(0))
    f = pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0).otherwise(
        1.25 * pl.col("tp") / (0.25 * pl.col("nt") + pl.col("np")).clip(lower_bound=1e-9))
    d = d.with_columns(f.alias("f"))
    res = {"all": round(float(d["f"].mean()), 5)}
    for (c,), g in d.group_by("country"):
        res[c] = round(float(g["f"].mean()), 5)
    return res, d


cur, dcur = macro(pred, "current")
tc = T.filter("in_c").select(K)
rep = {"run": RUN, "decision": best,
       "current": cur,
       "oracle_dec_perfect(T&C)": macro(tc, "dec")[0],
       "oracle_no_fp(P&T)": macro(P.filter("is_t").select(K), "nofp")[0],
       "oracle_all_fn(P|T&C)": macro(pl.concat([pred, tc]).unique(), "allfn")[0]}

n_t = T.height
rep["true_links"] = n_t
rep["retrieval_miss"] = int((~T["in_c"]).sum())
rep["decision_fn"] = int((T["in_c"] & ~T["in_p"]).sum())
rep["fp"] = int((~P["is_t"]).sum())
by_c = T.join(roles, on="s1_idx").group_by("country").agg(
    pl.len().alias("true"), (~pl.col("in_c")).sum().alias("retrieval_miss"),
    (pl.col("in_c") & ~pl.col("in_p")).sum().alias("decision_fn"))
rep["by_country_links"] = by_c.sort("country").to_dicts()

# how fixable are the decision FNs / FPs? -> stacked-p distribution
fnp = T.filter(pl.col("in_c") & ~pl.col("in_p")).join(val, on=K)["p"].to_numpy()
fpp = P.filter(~pl.col("is_t")).join(val, on=K)["p"].to_numpy()
edges = [0, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0001]
rep["decision_fn_p_hist"] = dict(zip([f"{a}-{b}" for a, b in zip(edges, edges[1:])],
                                     np.histogram(fnp, edges)[0].tolist()))
rep["fp_p_hist"] = dict(zip([f"{a}-{b}" for a, b in zip(edges, edges[1:])],
                            np.histogram(fpp, edges)[0].tolist()))

# where the F0.5 loss sits, per S1 category (share of the total 1-F0.5 loss)
dd = dcur.join(T.group_by("s1_idx").agg((~pl.col("in_c")).sum().alias("miss"),
                                        (pl.col("in_c") & ~pl.col("in_p")).sum().alias("dfn")),
               on="s1_idx", how="left").join(
    P.group_by("s1_idx").agg((~pl.col("is_t")).sum().alias("nfp")), on="s1_idx", how="left").fill_null(0)
dd = dd.with_columns((1 - pl.col("f")).alias("loss"))
cat = (pl.when(pl.col("loss") <= 1e-12).then(pl.lit("perfect"))
       .when((pl.col("nt") == 0) & (pl.col("np") > 0)).then(pl.lit("singleton_FP"))
       .when((pl.col("np") == 0) & (pl.col("nt") > 0)).then(pl.lit("predicted_empty"))
       .when((pl.col("miss") > 0) & (pl.col("dfn") == 0) & (pl.col("nfp") == 0)).then(pl.lit("retrieval_only"))
       .when((pl.col("miss") == 0) & (pl.col("dfn") > 0) & (pl.col("nfp") == 0)).then(pl.lit("decision_fn_only"))
       .when((pl.col("miss") == 0) & (pl.col("dfn") == 0) & (pl.col("nfp") > 0)).then(pl.lit("fp_only"))
       .otherwise(pl.lit("mixed")))
tot = float(dd["loss"].sum())
rep["loss_share_by_category"] = {r["cat"]: {"s1": r["s1"], "loss_share": round(r["loss"] / tot, 4)}
                                 for r in dd.with_columns(cat.alias("cat")).group_by("cat").agg(
                                     pl.len().alias("s1"), pl.col("loss").sum()).sort("loss", descending=True)
                                 .iter_rows(named=True)}
print(json.dumps(rep, indent=1))
if OUT:
    open(OUT, "w").write(json.dumps(rep, indent=1))
