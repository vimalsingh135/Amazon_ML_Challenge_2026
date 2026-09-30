"""Do Balaji's removed France pairs line up with v6ce's stacked p? Compare p of removed vs kept same-building,
name 70-89 France pairs (v6ce test_scores_ce, the 0.986 France base)."""
import io
import zipfile

import polars as pl
from rapidfuzz import fuzz, process

R = "E:/projects/Amazon ML challenge"
W = R + "/.worktrees/v6val/work_v6"


def load(path):
    raw = open(path, "rb").read()
    if raw[:2] == b"PK":
        z = zipfile.ZipFile(io.BytesIO(raw))
        raw = z.read(next(n for n in z.namelist() if n.endswith(".tsv")))
    d = pl.read_csv(io.BytesIO(raw), separator="\t", quote_char=None, infer_schema=False)
    d = d.rename({d.columns[0]: "s1", d.columns[1]: "m"})
    return d.with_columns(pl.col("m").fill_null("").str.split(",")).explode("m").filter(pl.col("m") != "").select("s1", pl.col("m").alias("t"))


s1 = pl.read_parquet(f"{W}/test_s1.parquet", columns=["entity_id", "idx", "name_norm", "addr_norm", "hno", "country"]).filter(pl.col("country") == "France")
s1 = s1.select(pl.col("entity_id").alias("s1"), pl.col("idx").alias("s1_idx"), pl.col("name_norm").fill_null("").alias("n1"),
               pl.col("addr_norm").fill_null("").alias("a1"), pl.col("hno").fill_null("").alias("h1"))
tg = pl.concat([pl.read_parquet(f"{W}/test_s{s}.parquet", columns=["entity_id", "idx", "name_norm", "addr_norm", "hno"]).with_columns(pl.lit(s, pl.UInt8).alias("src"))
                for s in (2, 3)]).select(pl.col("entity_id").alias("t"), pl.col("idx").alias("t_idx"), "src", pl.col("name_norm").fill_null("").alias("n2"),
                                         pl.col("addr_norm").fill_null("").alias("a2"), pl.col("hno").fill_null("").alias("h2"))
b86 = load(R + "/matching_results_0.986_on_unstop").join(s1.select("s1"), on="s1")
b88 = load(R + "/matching_results_0.988_balaji").join(s1.select("s1"), on="s1")
x = b86.join(b88.with_columns(pl.lit(True).alias("kept")), on=["s1", "t"], how="left").with_columns(pl.col("kept").fill_null(False))
x = x.join(s1, on="s1").join(tg, on="t")
x = x.with_columns(pl.Series("sn", process.cpdist(x["n1"].to_list(), x["n2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)),
                   pl.Series("sa", process.cpdist(x["a1"].to_list(), x["a2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)))
x = x.filter((pl.col("h1") != "") & (pl.col("h1") == pl.col("h2")) & (pl.col("sa") >= 80) & (pl.col("sn") < 90))
sc = pl.concat([pl.read_parquet(f"{W}/runs/v6ce/test_scores_ce.parquet"), pl.read_parquet(f"{W}/runs/v6ce/test_scores_ce.parquet").head(0)])
x = x.join(sc.select(pl.col("s1_idx").cast(pl.UInt32), pl.col("src").cast(pl.UInt8), pl.col("t_idx").cast(pl.UInt32), "p"),
           on=["s1_idx", "src", "t_idx"], how="left")
x = x.with_columns(pl.col("p").cut([0.5, 0.7, 0.8, 0.9, 0.95, 0.99]).alias("pbin"),
                   pl.col("sn").cut([50, 70, 80, 90]).alias("nbin"))
print("p missing (dense-added pairs):", x["p"].null_count(), "of", x.height)
t = x.group_by("nbin", "pbin").agg(pl.len().alias("n"), (~pl.col("kept")).sum().alias("removed")).with_columns(
    (pl.col("removed") / pl.col("n")).round(3).alias("removed_share")).sort("nbin", "pbin")
pl.Config.set_tbl_rows(60)
print(t)
