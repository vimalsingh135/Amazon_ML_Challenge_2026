"""Learned back-transliteration dictionary for Indic-script business names.

Indian S2/S3 records often write an English business name phonetically in a Brahmic script
("प्रीमियर बिजनेस" = Premier Business). er.text romanises it ("priimiyar bijnes"), but the
romanisation is not the English spelling, so name keys and similarities miss the pair
(v5: 9.4% blocking miss on Indic-script targets vs 2.5% for Latin script).

The dictionary is learned from the provided training pairs only. For a true pair whose target
name is Indic-script and whose normalised core has the same number of tokens as the S1 core,
tokens are aligned by position. A romanised token maps to its most frequent English partner
when seen >= 2 times with >= 70% agreement. Only FIT S1 are used, so the validation set stays
clean. The dictionary is applied only to names that contain Indic script (er.normalize).

python -m er.translit learn          (ER_WORK=<work dir>) -> <work>/aux/translit.json
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict

import polars as pl

from . import config as C

log = logging.getLogger(__name__)
INDIC = r"[ऀ-෿]"          # Devanagari .. Sinhala blocks (all Brahmic scripts used in India)


def learn(min_count: int = 2, min_share: float = 0.7) -> dict[str, str]:
    from .pipeline import labels, train_roles
    fit = train_roles().filter(pl.col("role") == "fit").select(pl.col("idx").cast(pl.UInt32).alias("s1_idx"))
    lab = labels().select(pl.col("s1_idx").cast(pl.UInt32), pl.col("src").cast(pl.UInt8),
                          pl.col("t_idx").cast(pl.UInt32)).join(fit, on="s1_idx")
    s1 = pl.read_parquet(C.WORK / "train_s1.parquet", columns=["idx", "core"]).select(
        pl.col("idx").cast(pl.UInt32).alias("s1_idx"), pl.col("core").alias("c1"))
    tg = pl.concat([pl.read_parquet(C.WORK / f"train_s{s}.parquet", columns=["idx", "business_name", "core"]).select(
        pl.lit(s).cast(pl.UInt8).alias("src"), pl.col("idx").cast(pl.UInt32).alias("t_idx"), "business_name",
        pl.col("core").alias("c2")) for s in (2, 3)])
    d = (lab.join(tg, on=["src", "t_idx"]).filter(pl.col("business_name").fill_null("").str.contains(INDIC))
            .join(s1, on="s1_idx").filter(pl.col("c1").list.len() == pl.col("c2").list.len()))
    cnt: dict[str, Counter] = defaultdict(Counter)
    for a, b in zip(d["c2"].to_list(), d["c1"].to_list()):
        for x, y in zip(a, b):
            if x != y:
                cnt[x][y] += 1
    dic = {}
    for x, c in cnt.items():
        y, n = c.most_common(1)[0]
        tot = sum(c.values())
        if tot >= min_count and n / tot >= min_share:
            dic[x] = y
    log.info("translit dictionary: %d entries from %d aligned Indic pairs", len(dic), d.height)
    return dic


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["learn"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    out = a.out or str(C.WORK / "aux" / "translit.json")
    dic = learn()
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(dic, fh, indent=0, sort_keys=True)
    print(json.dumps({"entries": len(dic), "out": out}))
