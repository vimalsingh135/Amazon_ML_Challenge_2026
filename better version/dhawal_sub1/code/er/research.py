"""Unseen-country research data (India plays France): streamed export of stage-B features for Kaggle CPU kernels.

python -m er.research export --run v4fr --out <dir>      (ER_WORK=<work dir>)
Writes fit_US/part_*.parquet (US FIT S1 with fold), val_India/part_*.parquet (India VAL S1), each row
(s1_idx, src, t_idx, y, fold, <stage-2 features>), plus val_India_s1.parquet (all India val S1 idx, incl. S1
without candidates) and features.json. One partition file at a time, so peak memory stays small.
"""
from __future__ import annotations

import argparse
import json
import os

import polars as pl

from . import config as C
from .pipeline import _files, _parts, labels, train_roles

KEYS = ["s1_idx", "src", "t_idx"]


def export(run: str, out: str) -> dict:
    feats = json.loads((C.WORK / "runs" / run / "summary.json").read_text())["features"]
    roles = train_roles().select(pl.col("idx").cast(pl.UInt32).alias("s1_idx"), "role", "fold", "country")
    lab = labels().select(pl.col("s1_idx").cast(pl.UInt32), pl.col("src").cast(pl.UInt8), pl.col("t_idx").cast(pl.UInt32)).with_columns(pl.lit(1, pl.Int8).alias("y"))
    sets = {"fit_US": roles.filter((pl.col("country") == "US") & (pl.col("role") == "fit")),
            "val_India": roles.filter((pl.col("country") == "India") & (pl.col("role") == "val"))}
    cols = [c for c in feats if c not in KEYS]
    n = {k: 0 for k in sets}
    for name, r in sets.items():
        os.makedirs(f"{out}/{name}", exist_ok=True)
        ctry = "US" if name == "fit_US" else "India"
        i = 0
        for d_ in _parts("train", "B"):
            if f"_B_{ctry}_" not in d_.name:
                continue
            for f in _files(d_):
                d = (pl.read_parquet(f, columns=[c for c in KEYS + cols if c in pl.read_parquet_schema(f)])
                       .with_columns(pl.col("s1_idx").cast(pl.UInt32), pl.col("src").cast(pl.UInt8), pl.col("t_idx").cast(pl.UInt32))
                       .join(r.select("s1_idx", "fold"), on="s1_idx").join(lab, on=KEYS, how="left")
                       .with_columns(pl.col("y").fill_null(0), *[pl.col(c).cast(pl.Float32) for c in cols if c in pl.read_parquet_schema(f)]))
                if d.height:
                    d.write_parquet(f"{out}/{name}/part_{i:05d}.parquet")
                    i += 1
                    n[name] += d.height
    sets["val_India"].select("s1_idx").write_parquet(f"{out}/val_India_s1.parquet")
    lab.join(sets["val_India"].select("s1_idx"), on="s1_idx").select(KEYS).write_parquet(f"{out}/val_India_truth.parquet")
    json.dump({"features": feats}, open(f"{out}/features.json", "w"))
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["export"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(export(a.run, a.out))
