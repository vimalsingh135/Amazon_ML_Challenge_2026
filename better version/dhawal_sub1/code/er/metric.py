"""Exact re-implementation of the challenge metric (macro F0.5 over Source-1 entities)."""
from __future__ import annotations

import polars as pl

BETA2 = 0.25


def macro_f05(pred: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series) -> dict:
    """pred/truth: long pairs (s1, tid). s1_ids: every evaluated Source-1 id.

    Singleton rules: empty/empty = 1, anything else involving an empty side = 0.
    """
    base = pl.DataFrame({"s1": s1_ids}).unique()
    t = truth.join(base, on="s1").unique()
    p = pred.join(base, on="s1").unique()
    tp = t.join(p, on=["s1", "tid"]).group_by("s1").len("tp")
    nt = t.group_by("s1").len("nt")
    np_ = p.group_by("s1").len("np")
    df = (base.join(nt, on="s1", how="left").join(np_, on="s1", how="left")
              .join(tp, on="s1", how="left").fill_null(0))
    prec = pl.col("tp") / pl.col("np")
    rec = pl.col("tp") / pl.col("nt")
    f = (1 + BETA2) * prec * rec / (BETA2 * prec + rec)
    df = df.with_columns(
        pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0)
          .when((pl.col("nt") == 0) | (pl.col("np") == 0) | (pl.col("tp") == 0)).then(0.0)
          .otherwise(f).alias("f"),
        pl.when(pl.col("np") > 0).then(prec).otherwise(None).alias("p"),
        pl.when(pl.col("nt") > 0).then(rec).otherwise(None).alias("r"),
    )
    return {"f05": df["f"].mean(), "prec": df["p"].mean(), "rec": df["r"].mean(),
            "n": df.height, "singleton_f": df.filter(pl.col("nt") == 0)["f"].mean()}
