"""Build a small, realistic dataset (real records) for end-to-end smoke/regression tests.

python -m er.minidata OUT_DIR --n 3000
Train: n sampled S1 per country with all their true matches, plus 3x distractor targets.
Test : n sampled S1 per country (incl. France) plus a random target sample of the same country.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from . import config as C
from .prep import read_tsv


def build(out: Path, n: int = 3000, seed: int = 0) -> None:
    for split in ("train", "test"):
        (out / split).mkdir(parents=True, exist_ok=True)
        s1 = read_tsv(C.DATA / split / f"{split}_source1.tsv")
        s1 = pl.concat([g.sample(min(n, g.height), seed=seed) for _, g in s1.group_by("country", maintain_order=True)])
        s1.write_csv(out / split / f"{split}_source1.tsv", separator="\t", quote_style="never")
        if split == "train":
            gt = read_tsv(C.DATA / "train" / "train_ground_truth.tsv").join(
                s1.select(pl.col("entity_id").alias("source1_entity_id")), on="source1_entity_id")
            gt.write_csv(out / "train" / "train_ground_truth.tsv", separator="\t", quote_style="never")
            want = set(gt["matched_entity_ids"].fill_null("").str.split(",").explode().to_list()) - {""}
        for src in ("2", "3"):
            t = read_tsv(C.DATA / split / f"{split}_source{src}.tsv")
            keep = t.filter(pl.col("entity_id").is_in(list(want))) if split == "train" else t.head(0)
            dist = t.join(keep.select("entity_id"), on="entity_id", how="anti")
            dist = pl.concat([g.sample(min(3 * n, g.height), seed=seed) for _, g in dist.group_by("country", maintain_order=True)])
            pl.concat([keep, dist]).sample(fraction=1.0, shuffle=True, seed=seed).write_csv(
                out / split / f"{split}_source{src}.tsv", separator="\t", quote_style="never")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=3000)
    a = ap.parse_args()
    build(Path(a.out), a.n)
