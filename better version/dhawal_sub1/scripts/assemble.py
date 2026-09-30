"""Final assembly: US/India rows from OUR predicted run + France rows exactly as in the LB-0.986 file.
France's candidate list = our France candidates  U  the 0.986 France matches (matches must be a subset of
candidates; his France dense pairs are not in our candidate set). Writes matching_results.tsv +
candidate_pairs.tsv and runs the official validator with --check-ids, then zips both files.
Usage: python assemble.py <our_output_dir> <out_name> [ref_countries, default France]
  e.g. probe "0.986 US/India + our France":  assemble.py <our_dir> probe_fr US,India"""
import io
import os
import subprocess
import sys
import zipfile

import polars as pl

R = "E:/projects/Amazon ML challenge"
OURS, NAME = sys.argv[1], sys.argv[2]
REF = R + "/matching_results_0.986_on_unstop"
OUT = f"{R}/.worktrees/rep/output/{NAME}"
os.makedirs(OUT, exist_ok=True)


def rd(data, col):
    d = pl.read_csv(io.BytesIO(data), separator="\t", quote_char=None, infer_schema=False)
    return d.rename({d.columns[0]: "s1", d.columns[1]: col}).with_columns(pl.col(col).fill_null(""))


z = zipfile.ZipFile(REF)
ref = rd(z.read(next(n for n in z.namelist() if n.endswith("matching_results.tsv"))), "m")
om = rd(open(f"{OURS}/matching_results.tsv", "rb").read(), "m")
oc = rd(open(f"{OURS}/candidate_pairs.tsv", "rb").read(), "c")
cty = pl.read_parquet(R + "/.worktrees/v6val/work_v6/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
assert om.height == ref.height == oc.height == cty.height, (om.height, ref.height, oc.height, cty.height)

REFC = (sys.argv[3] if len(sys.argv) > 3 else "France").split(",")
fr = cty.filter(pl.col("country").is_in(REFC)).select("s1")   # rows taken from the 0.986 file
m = pl.concat([om.join(fr, on="s1", how="anti"), ref.join(fr, on="s1", how="semi")])
# candidates of those rows: union of our candidate ids and his matched ids, deduplicated, order kept
c_fr = (oc.join(fr, on="s1", how="semi").join(ref.rename({"m": "rm"}), on="s1", how="left")
          .with_columns(pl.concat_list(pl.col("c").str.split(","), pl.col("rm").fill_null("").str.split(","))
                          .list.eval(pl.element().filter(pl.element() != "")).list.unique(maintain_order=True)
                          .list.join(",").alias("c")).select("s1", "c"))
c = pl.concat([oc.join(fr, on="s1", how="anti"), c_fr])
order = om.select("s1").with_row_index("i")
m = order.join(m, on="s1").sort("i").drop("i")
c = order.join(c, on="s1").sort("i").drop("i")
m.rename({"s1": "source1_entity_id", "m": "matched_entity_ids"}).write_csv(f"{OUT}/matching_results.tsv", separator="\t", quote_style="never")
c.rename({"s1": "source1_entity_id", "c": "candidate_entity_ids"}).write_csv(f"{OUT}/candidate_pairs.tsv", separator="\t", quote_style="never")
print("rows", m.height, f"rows from 0.986 ({','.join(REFC)}):", fr.height, flush=True)
v = subprocess.run([sys.executable, R + "/student_resource/utils/validate_submission.py", "--matching", f"{OUT}/matching_results.tsv",
                    "--candidate", f"{OUT}/candidate_pairs.tsv", "--test-dir", R + "/student_resource/dataset/test", "--check-ids"],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
print(v.stdout[-1500:], v.stderr[-800:], flush=True)
if "PASS" in v.stdout:
    for f in ("matching_results", "candidate_pairs"):
        with zipfile.ZipFile(f"{OUT}/{f}.zip", "w", zipfile.ZIP_DEFLATED) as zz:
            zz.write(f"{OUT}/{f}.tsv", f"{f}.tsv")
    print("ZIPPED", OUT, flush=True)
