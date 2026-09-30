"""Submission #1 clean: US/India rows from our v6cd (sub1_usin) + France rows exactly from LB-0.988. Light row swap +
official validator (no --check-ids: every id comes from files the Kaggle kernel already strict-validated)."""
import io
import os
import subprocess
import sys
import zipfile

import polars as pl

R = "E:/projects/Amazon ML challenge"
S = R + "/.worktrees/rep/sub1_out"
OUT = R + "/.worktrees/rep/output/SUB1_ours_usin_fr0988"
os.makedirs(OUT, exist_ok=True)


def rd(path, col):
    raw = open(path, "rb").read()
    if raw[:2] == b"PK":
        z = zipfile.ZipFile(io.BytesIO(raw))
        raw = z.read(next(n for n in z.namelist() if n.endswith(".tsv")))
    d = pl.read_csv(io.BytesIO(raw), separator="\t", quote_char=None, infer_schema=False)
    return d.rename({d.columns[0]: "s1", d.columns[1]: col}).with_columns(pl.col(col).fill_null(""))


fr = pl.read_parquet(R + "/.worktrees/v6val/work_v6/test_s1.parquet", columns=["entity_id", "country"]).filter(
    pl.col("country") == "France").select(pl.col("entity_id").alias("s1"))
ours, b88 = rd(f"{S}/sub1_usin_matching_results.zip", "m"), rd(R + "/matching_results_0.988_balaji", "m")
cand = rd(f"{S}/sub1_usin_candidate_pairs.zip", "c")
m = ours.select("s1").join(pl.concat([ours.join(fr, on="s1", how="anti"), b88.join(fr, on="s1", how="semi")]), on="s1", how="left")
# candidates: sub1_usin candidates U final matches (France 0.988 rows may lack the 174 additions -> still a superset)
c = (cand.join(m, on="s1", how="left").with_columns(
        pl.concat_list(pl.col("c").str.split(","), pl.col("m").fill_null("").str.split(","))
          .list.eval(pl.element().filter(pl.element() != "")).list.unique(maintain_order=True).list.join(",").alias("c")).select("s1", "c"))
m.rename({"s1": "source1_entity_id", "m": "matched_entity_ids"}).write_csv(f"{OUT}/matching_results.tsv", separator="\t", quote_style="never")
c.rename({"s1": "source1_entity_id", "c": "candidate_entity_ids"}).write_csv(f"{OUT}/candidate_pairs.tsv", separator="\t", quote_style="never")
v = subprocess.run([sys.executable, R + "/student_resource/utils/validate_submission.py", "--matching", f"{OUT}/matching_results.tsv",
                    "--candidate", f"{OUT}/candidate_pairs.tsv", "--test-dir", R + "/student_resource/dataset/test"],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
print(v.stdout[-500:])
if "PASS" in v.stdout:
    for f in ("matching_results", "candidate_pairs"):
        with zipfile.ZipFile(f"{OUT}/{f}.zip", "w", zipfile.ZIP_DEFLATED) as zz:
            zz.write(f"{OUT}/{f}.tsv", f"{f}.tsv")
    pb = lambda d: d.with_columns(pl.col("m").str.split(",")).explode("m").filter(pl.col("m") != "")
    a, b = pb(m), pb(b88)
    print("vs 0.988: added", a.join(b, on=["s1", "m"], how="anti").height, "removed", b.join(a, on=["s1", "m"], how="anti").height)
    print("READY", OUT)
