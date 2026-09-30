"""Per-country submission assembly: each country's rows come from the version that is best for it.

US / India are chosen on labelled validation (er.compare). France has no labels anywhere, so it is chosen
on leaderboard evidence. The LB decomposition (test mix US 38.3% / India 46.8% / France 15.0%, US/India
test ~ validation, verified by sum-p calibration) implies France F0.5 ~0.930 for v3 vs ~0.902 for v5.
v5 added ~60k weaker-name France pairs (median name similarity 84 vs 100), consistent with
co-located French businesses (17.5% of France S1 share an address vs 4-6% in US/India) being merged.

Both files of a country come from the same version, so candidate_pairs.tsv stays the exact candidate
set that version's model scored for those S1 rows.

python -m er.hybrid --base v6 --override France=v3 --out v6h
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import polars as pl

from . import config as C

FILES = (("matching_results.tsv", "matched_entity_ids"), ("candidate_pairs.tsv", "candidate_entity_ids"))


def assemble(base: str, override: dict[str, str], out: str) -> dict:
    s1 = (pl.read_parquet(C.WORK / "test_s1.parquet", columns=["entity_id", "country"])
            .rename({"entity_id": "source1_entity_id"}))
    od = C.OUT / out
    od.mkdir(parents=True, exist_ok=True)
    info = {}
    for fname, col in FILES:
        b = pl.read_csv(C.OUT / base / fname, separator="\t", infer_schema=False)
        order = b.select("source1_entity_id")
        parts = [b.join(s1, on="source1_entity_id").filter(~pl.col("country").is_in(list(override)))]
        for country, ver in override.items():
            o = pl.read_csv(C.OUT / ver / fname, separator="\t", infer_schema=False)
            parts.append(o.join(s1, on="source1_entity_id").filter(pl.col("country") == country))
        m = pl.concat([p.select("source1_entity_id", col) for p in parts])
        assert m.height == b.height == s1.height and m["source1_entity_id"].n_unique() == m.height
        m = order.join(m, on="source1_entity_id", maintain_order="left")     # keep the base row order
        m.write_csv(od / fname, separator="\t", quote_style="never")
        info[fname] = m.height
    r = subprocess.run([sys.executable, str(C.ROOT / "student_resource" / "utils" / "validate_submission.py"),
                        "--matching", str(od / "matching_results.tsv"), "--candidate", str(od / "candidate_pairs.tsv"),
                        "--test-dir", str(C.DATA / "test")], capture_output=True, text=True, encoding="utf-8")
    info["validator"] = (r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else r.returncode
    return info


def from_selection(base: str, country: str, sel_path: str, out: str) -> dict:
    """output/<out> = output/<base> with <country>'s matching rows rebuilt from a selected-pairs parquet
    (s1_idx, src, t_idx in C.WORK's test idx space, same candidate set as <base>; candidate_pairs.tsv copied)."""
    import shutil
    sel = pl.read_parquet(sel_path).select(pl.col("s1_idx").cast(pl.UInt32), pl.col("src").cast(pl.UInt8), pl.col("t_idx").cast(pl.UInt32))
    s1 = (pl.scan_parquet(C.WORK / "test_s1.parquet").filter(pl.col("country") == country)
            .select(pl.col("idx").cast(pl.UInt32).alias("s1_idx"), pl.col("entity_id").alias("source1_entity_id")).collect())
    tg = pl.concat([pl.scan_parquet(C.WORK / f"test_s{k}.parquet").select(pl.col("idx").cast(pl.UInt32).alias("t_idx"), pl.lit(k).cast(pl.UInt8).alias("src"),
                    pl.col("entity_id").alias("tid")).collect().join(sel.select("src", "t_idx").unique(), on=["src", "t_idx"], how="semi") for k in (2, 3)])
    m = (sel.join(s1, on="s1_idx").join(tg, on=["src", "t_idx"]).group_by("source1_entity_id")
            .agg(pl.col("tid").sort().str.join(",").alias("new")))
    b = pl.read_csv(C.OUT / base / "matching_results.tsv", separator="\t", infer_schema=False)
    rows = s1.select("source1_entity_id").join(m, on="source1_entity_id", how="left").with_columns(pl.col("new").fill_null(""))
    b = (b.join(rows, on="source1_entity_id", how="left")
          .with_columns(pl.when(pl.col("source1_entity_id").is_in(s1["source1_entity_id"].implode())).then(pl.col("new"))
                        .otherwise(pl.col("matched_entity_ids")).alias("matched_entity_ids")).drop("new"))
    od = C.OUT / out
    od.mkdir(parents=True, exist_ok=True)
    b.write_csv(od / "matching_results.tsv", separator="\t", quote_style="never")
    shutil.copyfile(C.OUT / base / "candidate_pairs.tsv", od / "candidate_pairs.tsv")
    return {"country_rows": s1.height, "rows_with_matches": m.height, "pairs": sel.height}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--from-selection", default=None, help="COUNTRY=path/to/selected_pairs.parquet")
    ap.add_argument("--override", action="append", default=[], help="COUNTRY=VERSION, repeatable")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.from_selection:
        c, path = a.from_selection.split("=", 1)
        print(from_selection(a.base, c, path, a.out))
    else:
        print(assemble(a.base, dict(x.split("=", 1) for x in a.override), a.out))
