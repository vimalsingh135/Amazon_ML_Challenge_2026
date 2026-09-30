"""Load raw TSVs and write normalised Parquet tables (one per split x source)."""
from __future__ import annotations

import logging
from multiprocessing import Pool

import polars as pl

from . import config as C
from .normalize import normalize_record

log = logging.getLogger(__name__)

SCHEMA = {
    "name_norm": pl.Utf8, "core": pl.List(pl.Utf8), "legal": pl.List(pl.Utf8),
    "skel": pl.List(pl.Utf8), "addr_norm": pl.Utf8, "atoks": pl.List(pl.Utf8),
    "nums": pl.List(pl.Utf8), "hno": pl.Utf8, "state": pl.Utf8,
    "loc": pl.List(pl.Utf8), "street": pl.List(pl.Utf8),
}


def read_tsv(path) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                       encoding="utf8-lossy")


def _norm_chunk(df: pl.DataFrame) -> pl.DataFrame:
    out = [normalize_record(n, a, c) for n, a, c in
           zip(df["business_name"], df["business_address"], df["country"])]
    cols = list(zip(*out)) if out else [[] for _ in SCHEMA]
    norm = pl.DataFrame({k: list(v) for k, v in zip(SCHEMA, cols)}, schema=SCHEMA)
    return pl.concat([df, norm], how="horizontal")


def normalize_df(df: pl.DataFrame, workers: int = C.WORKERS, chunk: int = 25000) -> pl.DataFrame:
    """Normalise in worker processes; each worker builds its own DataFrame chunk."""
    chunks = [df.slice(i, chunk) for i in range(0, df.height, chunk)]
    with Pool(workers) as pool:
        return pl.concat(pool.map(_norm_chunk, chunks))


def prep_split(split: str) -> None:
    for src in ("1", "2", "3"):
        out = C.WORK / f"{split}_s{src}.parquet"
        if out.exists():
            continue
        df = read_tsv(C.DATA / split / f"{split}_source{src}.tsv")
        df = df.with_columns(pl.col("country").fill_null(""),
                             pl.int_range(pl.len(), dtype=pl.UInt32).alias("idx"))
        normalize_df(df).write_parquet(out)
        log.info("prep %s s%s rows=%d", split, src, df.height)


def load_gt() -> pl.DataFrame:
    """Ground truth as long pairs (s1, tid)."""
    gt = read_tsv(C.DATA / "train" / "train_ground_truth.tsv")
    return (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
              .explode("matched_entity_ids")
              .filter(pl.col("matched_entity_ids") != "")
              .select(pl.col("source1_entity_id").alias("s1"),
                      pl.col("matched_entity_ids").alias("tid")))
