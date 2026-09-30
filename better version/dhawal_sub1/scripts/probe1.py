"""Submission #1 (direction check): LB-0.988 file + OUR France K=20 dense additions.
US/India rows = 0.988 exactly. France = 0.988 France + pairs selected in our v6cef2_k20 run (Zayaan's same-number +
street>=70 filter, v6ce decision) that are neither in 0.986 France nor removed by Balaji's category-swap rule; then
Balaji's france_rule_apply.py over the whole file (so no swap pair comes back). Label-free check: Balaji's pifit on the
added pairs (share false). Candidates = our run's candidates U final matches. Strict validator + zip.
Usage: python probe1.py <our_output_dir (v6cef2_k20)>"""
import io
import os
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import polars as pl

R = "E:/projects/Amazon ML challenge"
T = R + "/.worktrees/v6val/tmp"
OUT = R + "/.worktrees/rep/output/sub1_fr_k20"
OURS = sys.argv[1]
USIN = sys.argv[2] if len(sys.argv) > 2 else None   # optional: our output dir whose US/India rows replace 0.988's
os.makedirs(OUT, exist_ok=True)


def rd(path, col="m"):
    raw = open(path, "rb").read()
    if raw[:2] == b"PK":
        z = zipfile.ZipFile(io.BytesIO(raw))
        raw = z.read(next(n for n in z.namelist() if n.endswith(".tsv")))
    d = pl.read_csv(io.BytesIO(raw), separator="\t", quote_char=None, infer_schema=False)
    return d.rename({d.columns[0]: "s1", d.columns[1]: col}).with_columns(pl.col(col).fill_null(""))


def pairs(d, col="m"):
    return d.with_columns(pl.col(col).str.split(",")).explode(col).filter(pl.col(col) != "").select("s1", pl.col(col).alias("t"))


cty = pl.read_parquet(R + "/.worktrees/v6val/work_v6/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
fr = cty.filter(pl.col("country") == "France").select("s1")
b88 = rd(R + "/matching_results_0.988_balaji")
if USIN:   # US/India rows from our run, France rows from 0.988
    ours_ui = rd(f"{USIN}/matching_results.tsv").join(fr, on="s1", how="anti")
    b88 = b88.select("s1").join(pl.concat([ours_ui, b88.join(fr, on="s1", how="semi")]), on="s1", how="left").with_columns(pl.col("m").fill_null(""))
    OUT = OUT + "_usin"
    os.makedirs(OUT, exist_ok=True)
    print("US/India rows taken from", USIN, flush=True)
p88, p86 = pairs(b88), pairs(rd(R + "/matching_results_0.986_on_unstop"))
pour = pairs(rd(f"{OURS}/matching_results.tsv")).join(fr, on="s1", how="semi")
balaji_removed = p86.join(p88, on=["s1", "t"], how="anti")
add = (pour.join(p86, on=["s1", "t"], how="anti").join(balaji_removed, on=["s1", "t"], how="anti")
           .join(p88, on=["s1", "t"], how="anti"))
# a target may belong to at most one S1: drop additions whose target is already used in 0.988
add = add.join(p88.select("t").unique(), on="t", how="anti").unique(["t"], keep="first")
print("France additions (before Balaji rule):", add.height, flush=True)
allp = pl.concat([p88, add])
m = allp.group_by("s1", maintain_order=True).agg(pl.col("t").str.join(","))
out = b88.select("s1").join(m.rename({"t": "m"}), on="s1", how="left").with_columns(pl.col("m").fill_null(""))
pre = f"{OUT}/pre_rule.tsv"
out.rename({"s1": "source1_entity_id", "m": "matched_entity_ids"}).write_csv(pre, separator="\t", quote_style="never")
subprocess.run([sys.executable, f"{T}/france_rule_apply.py", "--in", pre, "--out", f"{OUT}/matching_results.tsv",
                "--test-dir", R + "/student_resource/dataset/test"], check=True, cwd=T)
final = pairs(rd(f"{OUT}/matching_results.tsv"))
added = final.join(p88, on=["s1", "t"], how="anti")
removed = p88.join(final, on=["s1", "t"], how="anti")
print("vs 0.988: added", added.height, "removed", removed.height, flush=True)

# Balaji's pifit (label-free share of FALSE among the added pairs), on France S1
fin_fr = final.join(fr, on="s1", how="semi")
nsel = fr.join(fin_fr.group_by("s1").len("n"), on="s1", how="left").fill_null(0).to_pandas().set_index("s1")["n"]
flag = added.join(fr, on="s1", how="semi").group_by("s1").len("k").to_pandas().set_index("s1")["k"]
if len(flag):
    K = 9
    ref = nsel.drop(flag.index).clip(upper=K).value_counts(normalize=True).reindex(range(K + 1), fill_value=0)
    obs = nsel.loc[flag.index].clip(upper=K).value_counts(normalize=True).reindex(range(K + 1), fill_value=0)
    k = int(round(flag.mean()))
    sb = ref * ref.index
    sb = sb / sb.sum()
    alt = ref.shift(k, fill_value=0)
    da, do = alt - sb, obs - sb
    pi = float((da * do).sum() / (da * da).sum())
    print(pd.DataFrame({"obs": obs, "null(true)": sb, "alt(FP)": alt}).loc[0:7].round(4).T.to_string())
    print(f"PIFIT share FALSE among added France pairs: {pi:.3f}  (Balaji's removal rule scored 0.94)", flush=True)

# candidates = our candidates U final matches
oc = rd(f"{OURS}/candidate_pairs.tsv", "c")
c = (oc.join(final.group_by("s1").agg(pl.col("t").str.join(",").alias("fm")), on="s1", how="left")
       .with_columns(pl.concat_list(pl.col("c").str.split(","), pl.col("fm").fill_null("").str.split(","))
                       .list.eval(pl.element().filter(pl.element() != "")).list.unique(maintain_order=True).list.join(",").alias("c"))
       .select("s1", "c"))
c = b88.select("s1").join(c, on="s1", how="left").with_columns(pl.col("c").fill_null(""))
c.rename({"s1": "source1_entity_id", "c": "candidate_entity_ids"}).write_csv(f"{OUT}/candidate_pairs.tsv", separator="\t", quote_style="never")
os.remove(pre)
v = subprocess.run([sys.executable, R + "/student_resource/utils/validate_submission.py", "--matching", f"{OUT}/matching_results.tsv",
                    "--candidate", f"{OUT}/candidate_pairs.tsv", "--test-dir", R + "/student_resource/dataset/test", "--check-ids"],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
print(v.stdout[-700:], flush=True)
if "PASS" in v.stdout:
    for f in ("matching_results", "candidate_pairs"):
        with zipfile.ZipFile(f"{OUT}/{f}.zip", "w", zipfile.ZIP_DEFLATED) as zz:
            zz.write(f"{OUT}/{f}.tsv", f"{f}.tsv")
    print("READY", OUT, flush=True)
