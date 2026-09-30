"""Cluster-coherence re-scorer experiment on v6ce val scores (2-fold cross-fitted over val S1).

True matches of one S1 are duplicates of each other, so a candidate target that closely resembles the
S1's OTHER confident candidates (same or other source) is more likely true; one that resembles none of
them is suspect. The pair model and the CE only compare S1<->target, never target<->target, so this is
evidence v6ce does not use.

Per candidate r of S1 s (paired with s's top-8 other candidates j by p):
  wsim_*   = sum_j p_j * sim(r, j) / sum_j p_j          (soft cluster agreement; name, addr, combined)
  asim_*   = max sim(r, j) over anchors (p_j >= 0.8)     (-1 when s has no other anchor)
  strong   = max_j min(p_j, sim_c(r, j) / 100)
  top1sim  = sim_c(r, top-1 candidate), top1p
  n_close  = #j with sim_c >= 90 and p_j >= 0.5
Control model = same GBDT on p-only features (lp, rank, n_cand, pmax, psum) so the gain is not just
recalibration. Decision params (ratio kind) are tuned on one half's OOF p and scored on the other.
Usage: python coherence_exp.py  (paths set below)
"""
import json
import os
import sys

ROOT = "E:/projects/Amazon ML challenge"
os.environ.setdefault("ER_WORK", ROOT + "/.worktrees/v6val/work_v6")
os.environ.setdefault("ER_DATA", ROOT + "/student_resource/dataset")
os.environ.setdefault("POLARS_MAX_THREADS", "6")
sys.path.insert(0, ROOT + "/.worktrees/ce/code/business_entity_resolution/src")

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from rapidfuzz import fuzz, process  # noqa: E402

from er import config as C  # noqa: E402
from er.decide import DecisionParams, select  # noqa: E402
from er.pipeline import id_maps, train_roles  # noqa: E402
from er.prep import load_gt  # noqa: E402

RUN = os.environ.get("COH_RUN", "v6ce")
OUT = os.environ.get("COH_OUT", ROOT + "/.worktrees/v6val/out/coherence_exp.json")
K = ["s1_idx", "src", "t_idx"]
TOPJ = 8

val = pl.read_parquet(C.WORK / "runs" / RUN / "val_scores.parquet").select(K + ["p"])
roles = train_roles().filter(pl.col("role") == "val").select(pl.col("idx").alias("s1_idx"), "country")
s1m, tg = id_maps("train")
truth = (load_gt().join(s1m.select("s1_idx", "s1"), on="s1").join(tg, on="tid").select(K)
         .join(roles.select("s1_idx"), on="s1_idx"))
val = val.join(truth.with_columns(pl.lit(1).alias("y")), on=K, how="left").with_columns(pl.col("y").fill_null(0))
print("val rows", val.height, "pos", int(val["y"].sum()), flush=True)

# ---- target texts -----------------------------------------------------------------------------
tx = []
for s in (2, 3):
    ids = val.filter(pl.col("src") == s)["t_idx"].unique()
    t = (pl.scan_parquet(C.WORK / f"train_s{s}.parquet").select("idx", "name_norm", "addr_norm", "hno")
           .filter(pl.col("idx").is_in(ids.implode())).collect())
    tx.append(t.with_columns(pl.lit(s, pl.UInt8).alias("src"), pl.col("idx").alias("t_idx")).drop("idx"))
tx = pl.concat(tx).with_columns(pl.col("name_norm").fill_null(""), pl.col("addr_norm").fill_null(""))

# ---- p-only features ---------------------------------------------------------------------------
pc = pl.col("p").clip(1e-6, 1 - 1e-6)
val = val.with_columns(
    (pc / (1 - pc)).log().alias("lp"),
    pl.col("p").rank("ordinal", descending=True).over("s1_idx").cast(pl.Int32).alias("rank"),
    pl.len().over("s1_idx").cast(pl.Int32).alias("n_cand"),
    pl.col("p").max().over("s1_idx").alias("pmax"),
    pl.col("p").sum().over("s1_idx").alias("psum"),
    pl.col("p").rank("ordinal", descending=True).over("src", "t_idx").cast(pl.Int32).alias("t_rank"),
    pl.len().over("src", "t_idx").cast(pl.Int32).alias("t_n"))

# ---- target<->target pairs within each S1 --------------------------------------------------------
v = val.select("s1_idx", "src", "t_idx", "p", "rank").join(tx, on=["src", "t_idx"], how="left")
v = v.with_columns(pl.col("name_norm").fill_null(""), pl.col("addr_norm").fill_null(""))
other = v.filter(pl.col("rank") <= TOPJ).rename({c: c + "_j" for c in v.columns if c != "s1_idx"})
pr = v.join(other, on="s1_idx").filter((pl.col("src") != pl.col("src_j")) | (pl.col("t_idx") != pl.col("t_idx_j")))
print("t-t pairs", pr.height, flush=True)
sn = np.empty(pr.height, np.float32)
sa = np.empty(pr.height, np.float32)
B = 2_000_000
for i in range(0, pr.height, B):
    c = pr.slice(i, B)
    sn[i:i + B] = process.cpdist(c["name_norm"].to_list(), c["name_norm_j"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    sa[i:i + B] = process.cpdist(c["addr_norm"].to_list(), c["addr_norm_j"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
pr = pr.select("s1_idx", "src", "t_idx", "p_j", "rank_j",
               (pl.col("addr_norm") == "").alias("na"), (pl.col("addr_norm_j") == "").alias("na_j"),
               ((pl.col("hno") == pl.col("hno_j")) & pl.col("hno").is_not_null()).alias("hno_eq")).with_columns(
    pl.Series("sn", sn), pl.Series("sa", sa))
# an empty address on either side carries no evidence -> fall back to name only
pr = pr.with_columns(pl.when(pl.col("na") | pl.col("na_j")).then(pl.col("sn")).otherwise(0.6 * pl.col("sn") + 0.4 * pl.col("sa")).alias("sc"))
anc = pl.col("p_j") >= 0.8
agg = pr.group_by(K).agg(
    ((pl.col("p_j") * pl.col("sn")).sum() / pl.col("p_j").sum()).alias("wsim_n"),
    ((pl.col("p_j") * pl.col("sa")).sum() / pl.col("p_j").sum()).alias("wsim_a"),
    ((pl.col("p_j") * pl.col("sc")).sum() / pl.col("p_j").sum()).alias("wsim_c"),
    pl.col("sn").filter(anc).max().fill_null(-1).alias("asim_n"),
    pl.col("sa").filter(anc).max().fill_null(-1).alias("asim_a"),
    pl.col("sc").filter(anc).max().fill_null(-1).alias("asim_c"),
    pl.min_horizontal(pl.col("p_j"), pl.col("sc") / 100).max().alias("strong"),
    pl.col("sc").filter(pl.col("rank_j") == 1).first().fill_null(-1).alias("top1sim"),
    ((pl.col("sc") >= 90) & (pl.col("p_j") >= 0.5)).sum().alias("n_close"),
    (pl.col("hno_eq") & anc).any().alias("hno_anchor"))
val = val.join(agg, on=K, how="left").with_columns(
    pl.col("wsim_n", "wsim_a", "wsim_c", "asim_n", "asim_a", "asim_c", "strong", "top1sim").fill_null(-1),
    pl.col("n_close").fill_null(0), pl.col("hno_anchor").fill_null(False).cast(pl.Int8))
del pr, sn, sa

F0 = ["lp", "src", "rank", "n_cand", "pmax", "psum", "t_rank", "t_n"]
F1 = F0 + ["wsim_n", "wsim_a", "wsim_c", "asim_n", "asim_a", "asim_c", "strong", "top1sim", "n_close", "hno_anchor"]
val = val.with_columns(((pl.col("s1_idx") // 7) % 2).alias("fold"))
PRM = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, seed=42, verbose=-1, num_threads=6)


def oof(feats):
    out = np.zeros(val.height, np.float32)
    fo = val["fold"].to_numpy()
    X = val.select(feats).to_numpy().astype(np.float32)
    y = val["y"].to_numpy()
    for f in (0, 1):
        m = lgb.train(PRM, lgb.Dataset(X[fo != f], y[fo != f]), 500)
        out[fo == f] = m.predict(X[fo == f])
        if feats is F1 and f == 0:
            imp = dict(zip(feats, m.feature_importance("gain").round(0).tolist()))
            print("gain", imp, flush=True)
    return out


nt = truth.group_by("s1_idx").len("nt")


def macro(sel, s1s):
    tp = sel.join(truth, on=K).group_by("s1_idx").len("tp")
    d = (s1s.join(nt, on="s1_idx", how="left").join(sel.group_by("s1_idx").len("np"), on="s1_idx", how="left")
            .join(tp, on="s1_idx", how="left").fill_null(0))
    f = pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0).otherwise(
        1.25 * pl.col("tp") / (0.25 * pl.col("nt") + pl.col("np")).clip(lower_bound=1e-9))
    d = d.with_columns(f.alias("f"))
    return {"all": float(d["f"].mean()), **{c: float(g["f"].mean()) for (c,), g in d.group_by("country")}}


GRID = [DecisionParams(margin=mg, m0=m0, empty_bias=eb, coh=1.0)
        for mg in (0.0, 0.05) for m0 in (0.0, 0.2, 0.5) for eb in (1.5, 2.0, 2.5, 3.0)]
fold_s1 = roles.with_columns(((pl.col("s1_idx") // 7) % 2).alias("fold"))


def cross_eval(col):
    """tune decision on one fold, score the other; returns the pooled macro over all val S1."""
    parts = []
    for f in (0, 1):
        tr_s1 = fold_s1.filter(pl.col("fold") != f)
        te_s1 = fold_s1.filter(pl.col("fold") == f)
        dtr = val.filter(pl.col("fold") != f).select(K + [pl.col(col).alias("p")])
        dte = val.filter(pl.col("fold") == f).select(K + [pl.col(col).alias("p")])
        best = max(GRID, key=lambda g: macro(select(dtr, g), tr_s1)["all"])
        parts.append((select(dte, best), te_s1, best))
    sel = pl.concat([p[0] for p in parts])
    r = macro(sel, roles)
    r["params"] = [vars(p[2]) for p in parts]
    return r, sel


rep = {"run": RUN}
val = val.with_columns(pl.Series("p_ctrl", oof(F0)), pl.Series("p_coh", oof(F1)))
for col in ("p", "p_ctrl", "p_coh"):
    r, sel = cross_eval(col)
    rep[col] = r
    print(col, json.dumps({k: (round(x, 5) if isinstance(x, float) else x) for k, x in r.items() if k != "params"}), flush=True)

# paired bootstrap p_coh vs p (per-S1 F differences)
def per_s1(col):
    r, sel = cross_eval(col)
    tp = sel.join(truth, on=K).group_by("s1_idx").len("tp")
    d = (roles.join(nt, on="s1_idx", how="left").join(sel.group_by("s1_idx").len("np"), on="s1_idx", how="left")
             .join(tp, on="s1_idx", how="left").fill_null(0).sort("s1_idx"))
    return d.with_columns(pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0).otherwise(
        1.25 * pl.col("tp") / (0.25 * pl.col("nt") + pl.col("np")).clip(lower_bound=1e-9)).alias("f"))["f"].to_numpy()


diff = per_s1("p_coh") - per_s1("p")
rng = np.random.default_rng(0)
bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(1000)]
rep["delta_coh_vs_p"] = {"mean": float(diff.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
print("delta", rep["delta_coh_vs_p"], flush=True)
open(OUT, "w").write(json.dumps(rep, indent=1))
