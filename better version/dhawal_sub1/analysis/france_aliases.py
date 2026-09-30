# Run from src/: python ../scripts/analysis/france_aliases.py  (label-free France alias mining from v3 confident matches -> research_data/france_aliases.json)
import json
import polars as pl
from er.aliases import mine_token_aliases
W3 = "../../../work_v3"
c = pl.read_parquet(f"{W3}/test_s1.parquet", columns=["idx", "country", "core", "atoks"]).filter(pl.col("country") == "France")
t = pl.read_parquet(f"{W3}/runs/v3/test_scores.parquet").select("s1_idx", "src", "t_idx", "p").join(c.select(pl.col("idx").alias("s1_idx")), on="s1_idx")
t = t.with_columns((pl.col("p") >= pl.col("p").max().over("src", "t_idx")).alias("best")).filter((pl.col("p") >= 0.97) & pl.col("best"))
print("France pseudo-positive pairs:", t.height)
tg = pl.concat([pl.read_parquet(f"{W3}/test_s{s}.parquet", columns=["idx", "core", "atoks"]).with_columns(pl.lit(s).cast(t["src"].dtype).alias("src")) for s in (2, 3)])
d = (t.join(c.select(pl.col("idx").alias("s1_idx"), "core", "atoks"), on="s1_idx")
      .join(tg.rename({"idx": "t_idx", "core": "core_2", "atoks": "atoks_2"}).with_columns(pl.col("t_idx").cast(t["t_idx"].dtype)), on=["src", "t_idx"]))
vocab = dict(c["core"].explode().drop_nulls().value_counts().iter_rows())
avocab = dict(c["atoks"].explode().drop_nulls().value_counts().iter_rows())
ca = mine_token_aliases(d["core"], d["core_2"], min_count=5, min_purity=0.6, s1_vocab_freq=vocab)
aa = mine_token_aliases(d["atoks"], d["atoks_2"], min_count=8, min_purity=0.6, s1_vocab_freq=avocab)
cur_c = json.load(open(f"{W3}/aux/core_alias.json")); cur_a = json.load(open(f"{W3}/aux/addr_alias.json"))
new_c = {k: v for k, v in ca.items() if k not in cur_c}; new_a = {k: v for k, v in aa.items() if k not in cur_a}
print("France core aliases:", len(ca), "new vs current aux:", len(new_c), list(new_c.items())[:40])
print("France addr aliases:", len(aa), "new:", len(new_a), list(new_a.items())[:40])
# how many France target records would change
print("share of France pseudo-target cores touched by new core aliases:", float(d["core_2"].list.eval(pl.element().is_in(list(new_c))).list.any().mean()))
print("share touched by new addr aliases:", float(d["atoks_2"].list.eval(pl.element().is_in(list(new_a))).list.any().mean()))
json.dump({"core": new_c, "addr": new_a}, open("../../../research_data/france_aliases.json", "w"), indent=0)
