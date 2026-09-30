"""Probe B on top of LB-0.988: remove France pairs that are in the same building (same house number, address
token_set >= 80) with name token_set in [LO, HI). US/India untouched. Candidate file = 0.988's matches (subset rule holds).
Usage: python fr_band_probe.py [LO=70] [HI=90]  -> .worktrees/rep/output/probe_fr_band{LO}_{HI}/"""
import io
import os
import sys
import zipfile

import polars as pl
from rapidfuzz import fuzz, process

LO, HI = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (70.0, 90.0)
R = "E:/projects/Amazon ML challenge"
W = R + "/.worktrees/v6val/work_v6"
OUT = f"{R}/.worktrees/rep/output/probe_fr_band{int(LO)}_{int(HI)}"
os.makedirs(OUT, exist_ok=True)
raw = open(R + "/matching_results_0.988_balaji", "rb").read()
if raw[:2] == b"PK":
    z = zipfile.ZipFile(io.BytesIO(raw))
    raw = z.read(next(n for n in z.namelist() if n.endswith(".tsv")))
sub = pl.read_csv(io.BytesIO(raw), separator="\t", quote_char=None, infer_schema=False)
cols = sub.columns
sub = sub.rename({cols[0]: "s1", cols[1]: "m"}).with_columns(pl.col("m").fill_null(""))
pairs = sub.with_columns(pl.col("m").str.split(",")).explode("m").filter(pl.col("m") != "").select("s1", pl.col("m").alias("t"))
S = pl.read_parquet(f"{W}/test_s1.parquet", columns=["entity_id", "name_norm", "addr_norm", "hno", "country"]).filter(pl.col("country") == "France").select(
    pl.col("entity_id").alias("s1"), pl.col("name_norm").fill_null("").alias("n1"), pl.col("addr_norm").fill_null("").alias("a1"),
    pl.col("hno").fill_null("").str.strip_chars().alias("h1"))
T = pl.concat([pl.read_parquet(f"{W}/test_s{s}.parquet", columns=["entity_id", "name_norm", "addr_norm", "hno"]) for s in (2, 3)]).select(
    pl.col("entity_id").alias("t"), pl.col("name_norm").fill_null("").alias("n2"), pl.col("addr_norm").fill_null("").alias("a2"),
    pl.col("hno").fill_null("").str.strip_chars().alias("h2"))
x = pairs.join(S, on="s1").join(T, on="t")
x = x.with_columns(pl.Series("sn", process.cpdist(x["n1"].to_list(), x["n2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)),
                   pl.Series("sa", process.cpdist(x["a1"].to_list(), x["a2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)))
drop = x.filter((pl.col("h1") != "") & (pl.col("h1") == pl.col("h2")) & (pl.col("sa") >= 80) & (pl.col("sn") >= LO) & (pl.col("sn") < HI)).select("s1", "t")
keep = pairs.join(drop, on=["s1", "t"], how="anti")
m = keep.group_by("s1", maintain_order=True).agg(pl.col("t").str.join(","))
out = sub.select("s1").join(m.rename({"t": "m"}), on="s1", how="left").with_columns(pl.col("m").fill_null(""))
out.rename({"s1": cols[0], "m": cols[1]}).write_csv(f"{OUT}/matching_results.tsv", separator="\t", quote_style="never")
sub.rename({"s1": cols[0], "m": "candidate_entity_ids"}).write_csv(f"{OUT}/candidate_pairs.tsv", separator="\t", quote_style="never")
fr_s1 = S.height
print(f"dropped {drop.height} France pairs (name {LO}-{HI}, same building); France pairs {x.height} -> {x.height - drop.height}; "
      f"France matches/S1 {x.height / fr_s1:.3f} -> {(x.height - drop.height) / fr_s1:.3f}", flush=True)
