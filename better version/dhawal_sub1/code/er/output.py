"""Submission writers with hard invariants (checked before and after writing)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import polars as pl


def id_lists(pairs: pl.DataFrame, s1_ids: pl.Series, col: str) -> pl.DataFrame:
    """pairs (s1, tid) -> one row per S1 in s1_ids, sorted unique comma-joined ids ('' if none)."""
    g = (pairs.unique(["s1", "tid"]).sort("s1", "tid")
              .group_by("s1", maintain_order=True).agg(pl.col("tid").str.join(",").alias(col)))
    out = (pl.DataFrame({"s1": s1_ids}).join(g, on="s1", how="left")
             .with_columns(pl.col(col).fill_null("")).sort("s1")
             .rename({"s1": "source1_entity_id"}))
    return out


def check_invariants(match: pl.DataFrame, cand: pl.DataFrame, s1_ids: pl.Series) -> None:
    """Fail loudly on anything the scorer would reject or that signals a pipeline bug."""
    n = s1_ids.n_unique()
    for name, df in (("matching", match), ("candidate", cand)):
        assert df.height == n, f"{name}: {df.height} rows != {n} test S1"
        assert df["source1_entity_id"].n_unique() == n, f"{name}: duplicate S1 rows"
    for name, pairs in (("matching", match), ("candidate", cand)):
        col = pairs.columns[1]
        ex = pairs.select(pl.col(col).str.split(",")).explode(col).filter(pl.col(col) != "")[col]
        assert ex.str.contains(r"^S[23]-").all(), f"{name}: non S2/S3 id present"
    mm = match.select("source1_entity_id", pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
    mm = mm.filter(pl.col("matched_entity_ids") != "")
    cc = cand.select("source1_entity_id", pl.col("candidate_entity_ids").str.split(",")).explode("candidate_entity_ids")
    missing = mm.join(cc, left_on=["source1_entity_id", "matched_entity_ids"],
                      right_on=["source1_entity_id", "candidate_entity_ids"], how="anti")
    assert missing.height == 0, f"{missing.height} matches are not in candidate_pairs"
    per_s1 = mm.group_by("source1_entity_id").agg(pl.len().alias("n"),
                                                  pl.col("matched_entity_ids").n_unique().alias("u"))
    assert (per_s1["n"] == per_s1["u"]).all(), "duplicate id within a list"


def write_submission(matches: pl.DataFrame, cands: pl.DataFrame, s1_ids: pl.Series, out_dir: Path,
                     validator: Path | None = None, test_dir: Path | None = None) -> None:
    """matches/cands: long (s1, tid). Writes matching_results.tsv + candidate_pairs.tsv."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cands = pl.concat([cands.select("s1", "tid"), matches.select("s1", "tid")]).unique()
    m = id_lists(matches, s1_ids, "matched_entity_ids")
    c = id_lists(cands, s1_ids, "candidate_entity_ids")
    check_invariants(m, c, s1_ids)
    m.write_csv(out_dir / "matching_results.tsv", separator="\t", quote_style="never")
    c.write_csv(out_dir / "candidate_pairs.tsv", separator="\t", quote_style="never")
    if validator is not None:
        r = subprocess.run([sys.executable, str(validator), "--matching", str(out_dir / "matching_results.tsv"),
                            "--candidate", str(out_dir / "candidate_pairs.tsv"), "--test-dir", str(test_dir)],
                           capture_output=True, text=True)
        print(r.stdout[-2000:])
        if r.returncode != 0:
            raise RuntimeError("official validator FAILED:\n" + r.stdout[-3000:])
