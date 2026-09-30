"""Where does France lose F0.5? Label-free diagnostic of a test submission (default: the LB-0.982 file).

France scores ~0.948 on the LB vs ~0.987 for US/India at the same matches-per-S1 (~3.36), so the loss
is most likely precision. For every selected (S1, target) pair we bucket name/address similarity
(token_set_ratio on the normalised fields, house-number agreement). The same buckets on v6ce's VAL
selections, where labels exist, give the precision of each bucket for US/India. Applying those
precisions to France's bucket mix estimates France's precision *if it erred like US/India*; a France
bucket much larger than in US/India marks where the unseen country's errors concentrate.
Usage: python france_diag.py [submission(.zip|.tsv|dir)]
"""
import io
import json
import os
import sys
import zipfile

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

SUB = sys.argv[1] if len(sys.argv) > 1 else ROOT + "/matching_results_0.982_on_unstop"
OUT = ROOT + "/.worktrees/v6val/out/france_diag.json"
COLS = ["idx", "entity_id", "name_norm", "addr_norm", "hno", "country"]
K = ["s1_idx", "src", "t_idx"]


def sims(pairs):
    """pairs has name_norm/addr_norm/hno (S1) and *_t (target) -> adds sn, sa, hno_eq, bucket."""
    pairs = pairs.with_columns(pl.col("name_norm", "addr_norm", "name_norm_t", "addr_norm_t").fill_null(""))
    sn = process.cpdist(pairs["name_norm"].to_list(), pairs["name_norm_t"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    sa = process.cpdist(pairs["addr_norm"].to_list(), pairs["addr_norm_t"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    pairs = pairs.with_columns(pl.Series("sn", sn), pl.Series("sa", sa),
                               ((pl.col("hno") == pl.col("hno_t")) & pl.col("hno").is_not_null()).alias("hno_eq"),
                               ((pl.col("addr_norm") == "") | (pl.col("addr_norm_t") == "")).alias("na"))
    nb = pl.when(pl.col("sn") >= 90).then(pl.lit("n90")).when(pl.col("sn") >= 70).then(pl.lit("n70")).otherwise(pl.lit("n<70"))
    ab = (pl.when(pl.col("na")).then(pl.lit("a-null")).when(pl.col("sa") >= 80).then(pl.lit("a80"))
            .when(pl.col("sa") >= 50).then(pl.lit("a50")).otherwise(pl.lit("a<50")))
    return pairs.with_columns(pl.concat_str([nb, ab], separator="|").alias("bucket"))


def texts(split, s):
    return pl.read_parquet(C.WORK / f"{split}_s{s}.parquet", columns=COLS)


# ---- test submission ---------------------------------------------------------------------------
path = os.path.join(SUB, "matching_results.tsv") if os.path.isdir(SUB) else SUB
raw = (zipfile.ZipFile(path).read(next(n for n in zipfile.ZipFile(path).namelist() if n.endswith("matching_results.tsv")))
       if zipfile.is_zipfile(path) else open(path, "rb").read())
d = pl.read_csv(io.BytesIO(raw), separator="\t", quote_char=None, infer_schema=False)
d = d.rename({d.columns[0]: "s1", d.columns[1]: "m"})
sel = d.with_columns(pl.col("m").fill_null("").str.split(",")).explode("m").filter(pl.col("m") != "")
s1 = texts("test", 1).drop("idx").rename({"entity_id": "s1"})
tg = pl.concat([texts("test", s).drop("idx", "country") for s in (2, 3)]).rename(
    {"entity_id": "m", "name_norm": "name_norm_t", "addr_norm": "addr_norm_t", "hno": "hno_t"})
tp = sims(sel.join(s1, on="s1").join(tg, on="m"))
print("test pairs", tp.height, flush=True)

# ---- val selections with labels ---------------------------------------------------------------
rd = C.WORK / "runs" / "v6ce"
best = json.loads((rd / "decision_ce.json").read_text())["best"]
vp = select(pl.read_parquet(rd / "val_scores_ce.parquet"),
            DecisionParams(margin=best["margin"], m0=best["m0"], empty_bias=best["empty_bias"], coh=best["coh"]))
s1m, tgm = id_maps("train")
truth = load_gt().join(s1m.select("s1_idx", "s1"), on="s1").join(tgm, on="tid").select(K).with_columns(pl.lit(1).alias("y"))
vp = vp.join(truth, on=K, how="left").with_columns(pl.col("y").fill_null(0))
v1 = texts("train", 1).drop("entity_id").rename({"idx": "s1_idx"})
vt = pl.concat([texts("train", s).drop("entity_id", "country").with_columns(pl.lit(s, pl.UInt8).alias("src")) for s in (2, 3)]).rename(
    {"idx": "t_idx", "name_norm": "name_norm_t", "addr_norm": "addr_norm_t", "hno": "hno_t"})
vv = sims(vp.join(v1, on="s1_idx").join(vt, on=["src", "t_idx"]))
print("val pairs", vv.height, "precision", round(float(vv["y"].mean()), 5), flush=True)

# ---- compare -----------------------------------------------------------------------------------
prec = vv.group_by("bucket").agg(pl.col("y").mean().alias("val_prec"), pl.len().alias("val_n"))
mix = tp.group_by("country", "bucket").len("n").with_columns((pl.col("n") / pl.col("n").sum().over("country")).alias("share"))
tab = mix.pivot(on="country", index="bucket", values="share").join(prec, on="bucket", how="left").sort("val_prec")
est = mix.join(prec, on="bucket", how="left").group_by("country").agg(
    (pl.col("share") * pl.col("val_prec").fill_null(0.5)).sum().round(5).alias("est_precision_if_like_val"))
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(10)
print(tab)
print(est)
q = tp.group_by("country").agg(pl.col("sn").median().alias("med_name"), pl.col("sa").median().alias("med_addr"),
                               pl.col("na").mean().round(4).alias("null_addr"), pl.col("hno_eq").mean().round(4).alias("hno_eq"))
print(q)
open(OUT, "w").write(json.dumps({"mix_vs_val_prec": tab.to_dicts(), "est": est.to_dicts(), "q": q.to_dicts()}, indent=1, default=str))
