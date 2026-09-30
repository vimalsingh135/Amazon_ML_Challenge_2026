"""What is left in the LB-0.988 France of the pattern Balaji removed (same house number & address, different name)?
Name-similarity buckets x house-number status of France pairs in 0.988, plus the same table for the removed pairs."""
import io
import os
import zipfile

import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

R = "E:/projects/Amazon ML challenge"
W = R + "/.worktrees/v6val/work_v6"


def load(path):
    if zipfile.is_zipfile(path):
        z = zipfile.ZipFile(path)
        data = z.read(next(n for n in z.namelist() if n.endswith(".tsv")))
    else:
        data = open(path, "rb").read()
    d = pl.read_csv(io.BytesIO(data), separator="\t", quote_char=None, infer_schema=False)
    d = d.rename({d.columns[0]: "s1", d.columns[1]: "m"})
    return d.with_columns(pl.col("m").fill_null("").str.split(",")).explode("m").filter(pl.col("m") != "").select("s1", pl.col("m").alias("t"))


s1 = pl.read_parquet(f"{W}/test_s1.parquet", columns=["entity_id", "name_norm", "addr_norm", "hno", "country"]).filter(
    pl.col("country") == "France").rename({"entity_id": "s1", "name_norm": "n1", "addr_norm": "a1", "hno": "h1"})
tg = pl.concat([pl.read_parquet(f"{W}/test_s{s}.parquet", columns=["entity_id", "name_norm", "addr_norm", "hno"]) for s in (2, 3)]).rename(
    {"entity_id": "t", "name_norm": "n2", "addr_norm": "a2", "hno": "h2"})
b88 = load(R + "/matching_results_0.988_balaji").join(s1.select("s1"), on="s1")
b86 = load(R + "/matching_results_0.986_on_unstop").join(s1.select("s1"), on="s1")
rem = b86.join(b88, on=["s1", "t"], how="anti")


def prof(p, name):
    x = p.join(s1, on="s1").join(tg, on="t").with_columns(pl.col("n1", "n2", "a1", "a2", "h1", "h2").fill_null("").str.strip_chars())
    x = x.with_columns(pl.Series("sn", process.cpdist(x["n1"].to_list(), x["n2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)),
                       pl.Series("sa", process.cpdist(x["a1"].to_list(), x["a2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)),
                       pl.Series("ed", process.cpdist(x["h1"].to_list(), x["h2"].to_list(), scorer=Levenshtein.distance, workers=-1)))
    x = x.with_columns(
        pl.when(pl.col("sn") >= 100).then(pl.lit("a:100")).when(pl.col("sn") >= 90).then(pl.lit("b:90-99")).when(pl.col("sn") >= 80).then(pl.lit("c:80-89"))
          .when(pl.col("sn") >= 70).then(pl.lit("d:70-79")).when(pl.col("sn") >= 50).then(pl.lit("e:50-69")).otherwise(pl.lit("f:<50")).alias("name"),
        (((pl.col("h1") != "") & (pl.col("ed") == 0)) & (pl.col("sa") >= 80)).alias("same_bldg"))
    t = x.group_by("name", "same_bldg").len("n").sort("name", "same_bldg")
    print(f"\n== {name}: {x.height} France pairs")
    print(t)
    return x


pl.Config.set_tbl_rows(30)
prof(rem, "REMOVED by Balaji (0.986 -> 0.988)")
x = prof(b88, "KEPT in 0.988")
x.filter(pl.col("same_bldg") & (pl.col("sn") < 90)).select("s1", "t", "sn", "n1", "n2").write_parquet(R + "/.worktrees/v6val/out/fr_left_samebldg_name_lt90.parquet")
