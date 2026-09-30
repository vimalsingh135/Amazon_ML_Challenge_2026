"""House-number status of every selected pair in the LB-0.986 file, per country (both numbers present: same / 1-edit / diff)."""
import io
import zipfile

import polars as pl
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

R = "E:/projects/Amazon ML challenge"
W = R + "/.worktrees/v6val/work_v6"
z = zipfile.ZipFile(R + "/matching_results_0.986_on_unstop")
d = pl.read_csv(io.BytesIO(z.read(z.namelist()[0])), separator="\t", quote_char=None, infer_schema=False)
d = d.rename({d.columns[0]: "s1", d.columns[1]: "m"})
p = d.with_columns(pl.col("m").fill_null("").str.split(",")).explode("m").filter(pl.col("m") != "")
s1 = pl.read_parquet(f"{W}/test_s1.parquet", columns=["entity_id", "hno", "country"]).rename({"entity_id": "s1", "hno": "h1"})
tg = pl.concat([pl.read_parquet(f"{W}/test_s{s}.parquet", columns=["entity_id", "hno"]) for s in (2, 3)]).rename({"entity_id": "m", "hno": "h2"})
x = p.join(s1, on="s1").join(tg, on="m").with_columns(pl.col("h1", "h2").fill_null("").str.strip_chars())
x = x.with_columns(pl.Series("ed", process.cpdist(x["h1"].to_list(), x["h2"].to_list(), scorer=Levenshtein.distance, workers=-1)))
x = x.with_columns(pl.when((pl.col("h1") == "") | (pl.col("h2") == "")).then(pl.lit("missing")).when(pl.col("ed") == 0).then(pl.lit("same"))
                     .when(pl.col("ed") == 1).then(pl.lit("1edit")).otherwise(pl.lit("diff")).alias("hn"))
t = x.group_by("country", "hn").len("n").with_columns((pl.col("n") / pl.col("n").sum().over("country")).round(4).alias("share")).sort("country", "hn")
pl.Config.set_tbl_rows(20)
print(t)
