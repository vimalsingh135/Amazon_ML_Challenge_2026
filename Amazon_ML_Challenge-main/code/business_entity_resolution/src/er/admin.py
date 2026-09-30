"""Label-free learning of administrative address parts for countries without a state lexicon.

Unseen countries (France) carry region / department parts that US/India lexicons do not know.
Left in place they occupy the locality slots used by blocking keys and distort locality features
(S1 writes the region, S2/S3 often the department or only the city). Two structural rules:

  R1 hierarchy : a part implied (P(part | city) >= 0.3) by >= 4 distinct frequent parts is an
                 administrative area above the city level (regions).
  R2 source-specific : a frequent S2/S3 part that never appears in the clean S1 reference and is
                 not a spelling variant of any S1 part (edit ratio < 80) is an admin variant
                 (departments replacing the region).

python -m er.admin learn --split test --out work_v31/aux/admin_parts.json
python -m er.admin renorm --split test --country France      (ER_WORK=<work dir>, ER_ADMIN_PARTS set)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import polars as pl
from rapidfuzz import fuzz, process

from . import config as C
from .aliases import norm_part
from .normalize import STATE_MAPS

log = logging.getLogger(__name__)


def _parts(split: str, src: str, country: str) -> pl.DataFrame:
    d = (pl.scan_parquet(C.WORK / f"{split}_s{src}.parquet").filter(pl.col("country") == country)
           .select("business_address").collect().with_row_index("r"))
    return (d.select("r", pl.col("business_address").fill_null("").str.split(",").alias("p")).explode("p")
             .with_columns(pl.col("p").map_elements(norm_part, return_dtype=pl.Utf8))
             .filter((pl.col("p") != "") & ~pl.col("p").str.contains(r"\d")).unique())


def learn_admin_parts(split: str, country: str, min_freq: int = 1000, implied_by: int = 4,
                      p_implied: float = 0.3, variant_ratio: float = 80.0) -> list[str]:
    per_src = {s: _parts(split, s, country).with_columns((pl.col("r").cast(pl.Int64) * 4 + int(s)).alias("r"))
               for s in ("1", "2", "3")}
    parts = pl.concat(per_src.values())
    freq = parts.group_by("p").len("f")
    big = freq.filter(pl.col("f") >= min_freq)
    co = (parts.join(big.select("p"), on="p").rename({"p": "y"})
               .join(parts.rename({"p": "x"}), on="r").filter(pl.col("x") != pl.col("y"))
               .group_by("x", "y").len("n").join(freq.rename({"p": "y", "f": "fy"}), on="y")
               .filter(pl.col("n") / pl.col("fy") >= p_implied))
    r1 = set(co.group_by("x").len("k").filter(pl.col("k") >= implied_by)["x"].to_list())
    f1 = per_src["1"].group_by("p").len("f1")
    f23 = pl.concat([per_src["2"], per_src["3"]]).group_by("p").len("f23")
    s1_parts = f1.filter(pl.col("f1") >= 50)["p"].to_list()
    cand = (f23.filter(pl.col("f23") >= min_freq).join(f1, on="p", how="left").with_columns(pl.col("f1").fill_null(0))
               .filter(pl.col("f1") <= 0.01 * pl.col("f23")))
    r2 = set()
    for p in cand["p"].to_list():
        best = process.extractOne(p, s1_parts, scorer=fuzz.ratio) if s1_parts else None
        if best is None or best[1] < variant_ratio:
            r2.add(p)
    out = sorted(r1 | r2)
    log.info("admin parts %s: R1=%s R2=%s", country, sorted(r1), sorted(r2))
    return out


def learn(split: str, out: Path) -> dict:
    countries = pl.scan_parquet(C.WORK / f"{split}_s1.parquet").select("country").unique().collect()["country"].to_list()
    res = {c: learn_admin_parts(split, c) for c in countries if c not in STATE_MAPS}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return res


def renorm(split: str, country: str) -> None:
    """Re-normalise one country's rows (other rows untouched, row order / idx preserved) and replace
    the split files atomically (breaks hard links, so other work dirs keep their originals)."""
    from .prep import normalize_df
    for src in ("1", "2", "3"):
        path = C.WORK / f"{split}_s{src}.parquet"
        df = pl.read_parquet(path)
        m = df["country"] == country
        raw = df.filter(m).select("entity_id", "business_name", "business_address", "country", "idx")
        new = normalize_df(raw).select(df.columns)
        out = pl.concat([df.filter(~m), new]).sort("idx")
        assert out.height == df.height and (out["idx"] == df["idx"]).all()
        tmp = path.with_suffix(".tmp")
        out.write_parquet(tmp)
        os.replace(tmp, path)
        log.info("renorm %s s%s: %d %s rows", split, src, new.height, country)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["learn", "renorm"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--country", default="France")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if a.what == "learn":
        print(json.dumps(learn(a.split, Path(a.out) if a.out else C.WORK / "aux" / "admin_parts.json"), indent=1))
    else:
        renorm(a.split, a.country)
