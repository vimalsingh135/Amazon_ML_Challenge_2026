"""Compare a candidate matching_results.tsv against the LB-0.982 reference submission.

Test has no labels, so this is structural: per country (from test_s1), mean matches per S1, empty
rate, pair overlap (both / ref-only / candidate-only) and exact set agreement. The true structure is
country-independent (~3.46 matches/S1, ~5.6% singletons), so a country far from it is suspicious.
Accepts a .tsv, a zip holding one, or a directory containing matching_results.tsv.
Usage: python compare_to_ref.py CANDIDATE [REF] [TEST_S1_PARQUET]
"""
import io
import json
import os
import sys
import zipfile

import polars as pl

ROOT = "E:/projects/Amazon ML challenge"
CAND = sys.argv[1]
REF = sys.argv[2] if len(sys.argv) > 2 else ROOT + "/matching_results_0.982_on_unstop"
TS1 = sys.argv[3] if len(sys.argv) > 3 else ROOT + "/.worktrees/v6val/work_v6/test_s1.parquet"


def load(path):
    if os.path.isdir(path):
        path = os.path.join(path, "matching_results.tsv")
    if zipfile.is_zipfile(path):
        z = zipfile.ZipFile(path)
        name = next(n for n in z.namelist() if n.endswith("matching_results.tsv"))
        data = z.read(name)
    else:
        data = open(path, "rb").read()
    d = pl.read_csv(io.BytesIO(data), separator="\t", quote_char=None, infer_schema=False)
    d = d.rename({d.columns[0]: "s1", d.columns[1]: "m"})
    pairs = (d.with_columns(pl.col("m").fill_null("").str.split(","))
              .explode("m").filter(pl.col("m") != "").select("s1", pl.col("m").alias("t")))
    sets = d.select("s1", pl.col("m").fill_null("").str.split(",").list.sort().list.join(",").alias("k"))
    return d.height, pairs, sets


n_c, pc, sc = load(CAND)
n_r, pr, sr = load(REF)
cty = pl.read_parquet(TS1, columns=["entity_id", "country"]).rename({"entity_id": "s1"})
rep = {"candidate": CAND, "reference": REF, "rows": {"candidate": n_c, "reference": n_r}}

per = []
for name, p in (("ref", pr), ("cand", pc)):
    cnt = cty.join(p.group_by("s1").len("n"), on="s1", how="left").with_columns(pl.col("n").fill_null(0))
    per.append(cnt.group_by("country").agg(pl.col("n").mean().round(3).alias(f"{name}_matches"),
                                           (pl.col("n") == 0).mean().round(4).alias(f"{name}_empty")))
struct = per[0].join(per[1], on="country").sort("country")

j = (pr.with_columns(pl.lit(1).alias("r")).join(pc.with_columns(pl.lit(1).alias("c")), on=["s1", "t"], how="full",
                                                coalesce=True).join(cty, on="s1", how="left"))
ov = j.group_by("country").agg((pl.col("r").is_not_null() & pl.col("c").is_not_null()).sum().alias("both"),
                               (pl.col("r").is_not_null() & pl.col("c").is_null()).sum().alias("ref_only"),
                               (pl.col("r").is_null() & pl.col("c").is_not_null()).sum().alias("cand_only")).sort("country")
agree = (sr.rename({"k": "kr"}).join(sc.rename({"k": "kc"}), on="s1").join(cty, on="s1")
           .group_by("country").agg((pl.col("kr") == pl.col("kc")).mean().round(4).alias("identical_sets"))).sort("country")
out = struct.join(ov, on="country").join(agree, on="country")
rep["by_country"] = out.to_dicts()
rep["total"] = {"ref_pairs": pr.height, "cand_pairs": pc.height,
                "both": int(out["both"].sum()), "ref_only": int(out["ref_only"].sum()),
                "cand_only": int(out["cand_only"].sum())}
pl.Config.set_tbl_cols(12)
print(out)
print(json.dumps(rep["total"], indent=1))
