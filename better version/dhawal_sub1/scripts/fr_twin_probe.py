"""Probe on top of LB-0.988: France 'twin' rule. A France pair (s1, t) whose S1 and target are in the same building
(same house number, address token_set >= 80) but whose names differ (token_set < 90) is removed only if ANOTHER
France S1 in the same building (same house number + same normalised street) matches t's name much better (>= 95):
t then belongs to that twin business, not to s1 (each target belongs to at most one S1). US/India rows untouched.
Writes .worktrees/rep/output/probe_fr_twin/matching_results.tsv; candidate file = the 0.988 matches (valid: matches
are a subset). Prints counts + samples."""
import io
import os
import zipfile

import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

R = "E:/projects/Amazon ML challenge"
W = R + "/.worktrees/v6val/work_v6"
OUT = R + "/.worktrees/rep/output/probe_fr_twin"
os.makedirs(OUT, exist_ok=True)
raw = open(R + "/matching_results_0.988_balaji", "rb").read()
if raw[:2] == b"PK":
    z = zipfile.ZipFile(io.BytesIO(raw))
    raw = z.read(next(n for n in z.namelist() if n.endswith(".tsv")))
sub = pl.read_csv(io.BytesIO(raw), separator="\t", quote_char=None, infer_schema=False)
cols = sub.columns
sub = sub.rename({cols[0]: "s1", cols[1]: "m"}).with_columns(pl.col("m").fill_null(""))
pairs = sub.with_columns(pl.col("m").str.split(",")).explode("m").filter(pl.col("m") != "").select("s1", pl.col("m").alias("t"))

S = pl.read_parquet(f"{W}/test_s1.parquet", columns=["entity_id", "name_norm", "addr_norm", "hno", "street", "country"]).filter(pl.col("country") == "France")
S = S.select(pl.col("entity_id").alias("s1"), pl.col("name_norm").fill_null("").alias("n1"), pl.col("addr_norm").fill_null("").alias("a1"),
             pl.col("hno").fill_null("").str.strip_chars().alias("h1"), pl.col("street").list.sort().list.join(" ").fill_null("").alias("st1"))
T = pl.concat([pl.read_parquet(f"{W}/test_s{s}.parquet", columns=["entity_id", "name_norm", "addr_norm", "hno"]) for s in (2, 3)]).select(
    pl.col("entity_id").alias("t"), pl.col("name_norm").fill_null("").alias("n2"), pl.col("addr_norm").fill_null("").alias("a2"),
    pl.col("hno").fill_null("").str.strip_chars().alias("h2"))
x = pairs.join(S, on="s1").join(T, on="t")
x = x.with_columns(pl.Series("sn", process.cpdist(x["n1"].to_list(), x["n2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)),
                   pl.Series("sa", process.cpdist(x["a1"].to_list(), x["a2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)))
cand = x.filter((pl.col("h1") != "") & (pl.col("h1") == pl.col("h2")) & (pl.col("sa") >= 80) & (pl.col("sn") < 90))
print("France pairs", x.height, "same-building & name<90:", cand.height, flush=True)
# twins: other France S1 with the same house number and street
bld = S.filter((pl.col("h1") != "") & (pl.col("st1") != "")).select("s1", "h1", "st1", "n1")
tw = (cand.select("s1", "t", "n2", "h1", "st1").join(bld.rename({"s1": "s1b", "n1": "nb"}), on=["h1", "st1"])
          .filter(pl.col("s1b") != pl.col("s1")))
tw = tw.with_columns(pl.Series("sb", process.cpdist(tw["n2"].to_list(), tw["nb"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)))
best = tw.group_by("s1", "t").agg(pl.col("sb").max().alias("twin_best"), pl.col("s1b").get(pl.col("sb").arg_max()).alias("twin"))
cand = cand.join(best, on=["s1", "t"], how="left").with_columns(pl.col("twin_best").fill_null(-1))
drop = cand.filter(pl.col("twin_best") >= 95)
print("have a same-building twin:", int((cand["twin_best"] >= 0).sum()), " twin name >= 95 -> DROP:", drop.height, flush=True)
with pl.Config(fmt_str_lengths=40, tbl_width_chars=200, tbl_rows=12):
    print(drop.select("sn", "twin_best", "n1", "n2").sample(min(12, drop.height), seed=0))
keep = pairs.join(drop.select("s1", "t"), on=["s1", "t"], how="anti")
m = keep.group_by("s1", maintain_order=True).agg(pl.col("t").str.join(","))
out = sub.select("s1").join(m.rename({"t": "m"}), on="s1", how="left").with_columns(pl.col("m").fill_null(""))
out.rename({"s1": cols[0], "m": cols[1]}).write_csv(f"{OUT}/matching_results.tsv", separator="\t", quote_style="never")
cand_out = sub.rename({"s1": cols[0], "m": "candidate_entity_ids"})
cand_out.write_csv(f"{OUT}/candidate_pairs.tsv", separator="\t", quote_style="never")
print("pairs", pairs.height, "->", keep.height, "| rows", out.height, flush=True)
