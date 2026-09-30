"""Hard-negative triplets for bi-encoder v2: for each FIT positive pair in bi_train.parquet (Zayaan's export), add the
highest-oof WRONG candidate of the same S1 and source from runs/v6/fit_oof.parquet (the pair the pipeline found most
confusing). Text format = er.cross._attach ("name | address"). Output: rep2/bi_hn_data/bi_train_hn.parquet (a, b, c)."""
import os
import sys

R = "E:/projects/Amazon ML challenge"
os.environ.setdefault("ER_WORK", R + "/.worktrees/v6val/work_v6")
sys.path.insert(0, R + "/.worktrees/final2/code/business_entity_resolution/src")
import json  # noqa: E402

import polars as pl  # noqa: E402

from er.cross import _texts  # noqa: E402

W = R + "/.worktrees"
out = f"{W}/rep2/bi_hn_data"
os.makedirs(out, exist_ok=True)
pos = pl.read_parquet(f"{W}/rep/bi_data/bi_train.parquet").select("s1_idx", "src", "t_idx", "a", "b")
oof = pl.read_parquet(f"{W}/v6val/work_v6/runs/v6/fit_oof.parquet").with_columns(pl.col("src").cast(pl.UInt8))
neg = (oof.filter(pl.col("y") == 0).sort("oof", descending=True).group_by("s1_idx", "src").first()
          .select("s1_idx", "src", pl.col("t_idx").alias("n_idx"), pl.col("oof").alias("neg_oof")))
d = pos.with_columns(pl.col("src").cast(pl.UInt8)).join(neg, on=["s1_idx", "src"], how="inner")
tx = _texts("train")
d = pl.concat([d.filter(pl.col("src") == s).join(tx[str(s)].rename({"idx": "n_idx", "t": "c"}), on="n_idx") for s in (2, 3)])
d = d.sample(fraction=1.0, shuffle=True, seed=0)
d.select("a", "b", "c", "neg_oof").write_parquet(f"{out}/bi_train_hn.parquet")
json.dump({"title": "almc-bi-hn-data", "id": "codelearner00/almc-bi-hn-data", "licenses": [{"name": "CC0-1.0"}]},
          open(f"{out}/dataset-metadata.json", "w"))
print("positives", pos.height, "triplets", d.height, "neg_oof median", float(d["neg_oof"].median()))
