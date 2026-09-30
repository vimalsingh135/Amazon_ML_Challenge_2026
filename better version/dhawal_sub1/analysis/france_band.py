"""Is France's F0.5 gap a recall hole in the medium-address-similarity pairs?

france_diag.py showed the LB-0.982 France selections are almost all near-identical pairs (name>=90 &
addr>=80: 80%, house-number agreement 91%), with 5-60x fewer "addr 50-80" selections than US/India,
where those pairs are 99.8% precise on val. Here, on v6ce's scored TEST pairs, per country:
  * how many pairs per S1 fall in each similarity bucket (are the France pairs there at all?), and
  * their stacked p distribution (are France's scored but under-rated?).
If France has as many name>=90 & addr 50-80 pairs per S1 as US/India but much lower p, the pair model
is under-scoring French address variants -> fix normalisation / adapt; if it has far fewer, blocking
misses them -> fix retrieval.
"""
import json
import os
import sys

ROOT = "E:/projects/Amazon ML challenge"
os.environ.setdefault("ER_WORK", ROOT + "/.worktrees/v6val/work_v6")
os.environ.setdefault("POLARS_MAX_THREADS", "6")
sys.path.insert(0, ROOT + "/.worktrees/ce/code/business_entity_resolution/src")

import polars as pl  # noqa: E402
from rapidfuzz import fuzz, process  # noqa: E402

from er import config as C  # noqa: E402

N_S1 = int(os.environ.get("FB_N_S1", "40000"))   # sampled S1 per country
COLS = ["idx", "name_norm", "addr_norm", "hno", "business_address"]

sc = pl.read_parquet(C.WORK / "runs" / "v6ce" / "test_scores_ce.parquet").select("s1_idx", "src", "t_idx", "p")
s1 = pl.read_parquet(C.WORK / "test_s1.parquet", columns=COLS + ["country"]).rename({"idx": "s1_idx"})
pick = pl.concat([g.sample(min(N_S1, g.height), seed=0) for _, g in s1.group_by("country")]).select("s1_idx")
sc = sc.join(pick, on="s1_idx")
tg = pl.concat([pl.read_parquet(C.WORK / f"test_s{s}.parquet", columns=COLS).with_columns(pl.lit(s, pl.UInt8).alias("src"))
                for s in (2, 3)]).rename({"idx": "t_idx", "name_norm": "name_t", "addr_norm": "addr_t", "hno": "hno_t",
                                          "business_address": "raw_t"})
d = sc.join(s1, on="s1_idx").join(tg, on=["src", "t_idx"])
d = d.with_columns(pl.col("name_norm", "addr_norm", "name_t", "addr_t").fill_null(""))
d = d.with_columns(pl.Series("sn", process.cpdist(d["name_norm"].to_list(), d["name_t"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)),
                   pl.Series("sa", process.cpdist(d["addr_norm"].to_list(), d["addr_t"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)))
nb = pl.when(pl.col("sn") >= 90).then(pl.lit("n90")).when(pl.col("sn") >= 70).then(pl.lit("n70")).otherwise(pl.lit("n<70"))
ab = (pl.when((pl.col("addr_norm") == "") | (pl.col("addr_t") == "")).then(pl.lit("a-null")).when(pl.col("sa") >= 80).then(pl.lit("a80"))
        .when(pl.col("sa") >= 50).then(pl.lit("a50")).otherwise(pl.lit("a<50")))
d = d.with_columns(pl.concat_str([nb, ab], separator="|").alias("bucket"),
                   ((pl.col("hno") == pl.col("hno_t")) & pl.col("hno").is_not_null()).alias("hno_eq"))
n1 = pick.join(s1.select("s1_idx", "country"), on="s1_idx").group_by("country").len("n_s1")
tab = (d.group_by("country", "bucket").agg(pl.len().alias("n"), pl.col("p").median().round(3).alias("p_med"),
                                           (pl.col("p") >= 0.8).mean().round(3).alias("p>=.8"),
                                           pl.col("hno_eq").mean().round(3).alias("hno_eq"))
         .join(n1, on="country").with_columns((pl.col("n") / pl.col("n_s1")).round(4).alias("per_s1")).drop("n_s1"))
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(10)
foc = tab.filter(pl.col("bucket").is_in(["n90|a80", "n90|a50", "n70|a80", "n70|a50", "n<70|a80", "n90|a<50"]))
print(foc.pivot(on="country", index="bucket", values="per_s1"))
print(foc.pivot(on="country", index="bucket", values="p_med"))
print(foc.pivot(on="country", index="bucket", values="p>=.8"))
print(foc.pivot(on="country", index="bucket", values="hno_eq"))
ex = d.filter((pl.col("country") == "France") & (pl.col("bucket") == "n90|a50"))
ex = ex.sample(min(15, ex.height), seed=1).select("p", "sa", "hno_eq", "business_address", "raw_t")
with pl.Config(fmt_str_lengths=70, tbl_width_chars=220):
    print(ex)
open(ROOT + "/.worktrees/v6val/out/france_band.json", "w").write(json.dumps(tab.to_dicts(), indent=1, default=str))
