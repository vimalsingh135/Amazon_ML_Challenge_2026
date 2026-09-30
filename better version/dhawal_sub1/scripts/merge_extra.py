"""Zayaan's er.dense.merge with extra dense-stacker features (his code otherwise unchanged).
  lex : name / address token-set similarity (normalised fields from the s-files), house-number status, name length ratio
  ce2 : second-opinion cross-encoder score + logit (ce2_{part}.parquet from the almc-ce2-score kernel)
The dense stacker still trains on FIT fold-1 pairs only (labels from ground truth), so val stays untouched for gating.
Usage: python merge_extra.py <dense_dir> <out_dense_dir> <out_run> <lex|ce2|lex+ce2> [ce2_dir]"""
import os
import sys

R = "E:/projects/Amazon ML challenge"
os.environ.setdefault("ER_WORK", R + "/.worktrees/v6val/work_v6")
os.environ.setdefault("ER_DATA", R + "/student_resource/dataset")
os.environ.setdefault("POLARS_MAX_THREADS", "6")
os.environ.setdefault("ER_WORKERS", "6")
sys.path.insert(0, R + "/.worktrees/final2/code/business_entity_resolution/src")

import polars as pl  # noqa: E402
from rapidfuzz import fuzz, process  # noqa: E402

from er import config as C  # noqa: E402
from er import dense  # noqa: E402

dd, od, run, mode = sys.argv[1:5]
ce2_dir = sys.argv[5] if len(sys.argv) > 5 else None
os.makedirs(od, exist_ok=True)
K = ["s1_idx", "src", "t_idx"]
cast = lambda d: d.with_columns(pl.col("s1_idx").cast(pl.UInt32), pl.col("src").cast(pl.UInt8), pl.col("t_idx").cast(pl.UInt32))
COLS = ["idx", "name_norm", "addr_norm", "hno"]


def texts(split):
    s1 = pl.read_parquet(C.WORK / f"{split}_s1.parquet", columns=COLS).rename(
        {"idx": "s1_idx", "name_norm": "n1", "addr_norm": "a1", "hno": "h1"}).with_columns(pl.col("s1_idx").cast(pl.UInt32))
    tg = pl.concat([pl.read_parquet(C.WORK / f"{split}_s{s}.parquet", columns=COLS).with_columns(pl.lit(s, pl.UInt8).alias("src"))
                    for s in (2, 3)]).rename({"idx": "t_idx", "name_norm": "n2", "addr_norm": "a2", "hno": "h2"}).with_columns(pl.col("t_idx").cast(pl.UInt32))
    return s1, tg


def lex(d, split):
    s1, tg = texts(split)
    x = d.select(K).join(s1, on="s1_idx", how="left").join(tg, on=["src", "t_idx"], how="left").with_columns(
        pl.col("n1", "n2", "a1", "a2", "h1", "h2").fill_null("").str.strip_chars())
    sn = process.cpdist(x["n1"].to_list(), x["n2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    sa = process.cpdist(x["a1"].to_list(), x["a2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    from rapidfuzz.distance import Levenshtein
    ed = process.cpdist(x["h1"].to_list(), x["h2"].to_list(), scorer=Levenshtein.distance, workers=-1)
    x = x.with_columns(pl.Series("lx_name", sn), pl.Series("lx_addr", sa), pl.Series("_ed", ed)).with_columns(
        pl.when((pl.col("h1") == "") & (pl.col("h2") == "")).then(0)
          .when((pl.col("h1") == "") | (pl.col("h2") == "")).then(1)
          .when(pl.col("_ed") == 0).then(2).when(pl.col("_ed") == 1).then(3).otherwise(4).cast(pl.Float32).alias("lx_hno"),
        (pl.min_horizontal(pl.col("n1").str.len_chars(), pl.col("n2").str.len_chars())
         / pl.max_horizontal(pl.col("n1").str.len_chars(), pl.col("n2").str.len_chars(), pl.lit(1))).cast(pl.Float32).alias("lx_lenr"),
        (pl.col("a1").eq("") | pl.col("a2").eq("")).cast(pl.Float32).alias("lx_anull"))
    return d.join(x.select(K + LEX), on=K, how="left")


LEX = ["lx_name", "lx_addr", "lx_hno", "lx_lenr", "lx_anull"]
extra = []
for part in ("fit1", "val", "test"):
    d = cast(pl.read_parquet(f"{dd}/dense_{part}.parquet"))
    if "lex" in mode:
        d = lex(d, "test" if part == "test" else "train")
    if "ce2" in mode:
        c = cast(pl.read_parquet(f"{ce2_dir}/ce2_{part}.parquet")).select(K + ["ce2"])
        d = d.join(c, on=K, how="left")
        assert d["ce2"].null_count() == 0, f"{part}: pairs without ce2"
        c2 = pl.col("ce2").clip(1e-6, 1 - 1e-6)
        d = d.with_columns((c2 / (1 - c2)).log().alias("lce2"))
    d.write_parquet(f"{od}/dense_{part}.parquet")
    print(part, d.height, flush=True)
if "lex" in mode:
    extra += LEX
if "ce2" in mode:
    extra += ["ce2", "lce2"]
dense.DENSE_FEATS = dense.DENSE_FEATS + extra
print("features", dense.DENSE_FEATS, flush=True)
print(dense.merge("v6ce", od, run))
