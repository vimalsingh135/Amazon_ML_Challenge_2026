"""Kaggle CPU kernel: build submission #1 (direction check) without touching the laptop.
  A) our v6cd US/India test predictions (Zayaan's er.decide.select + er.output.write_submission, tuned decision_ce.json)
  B) sub1_usin = our v6cd US/India + (0.988 France + our France K=20 additions) -> Balaji's category-swap rule
     sub1_fr   = 0.988 US/India + (0.988 France + K=20 additions) -> rule               (spare, France-only probe)
  C) Balaji's pifit (label-free share FALSE of added France pairs), official validator --check-ids, zips.
Inputs: dataset almc-sub1-data (code, slim id maps, v6cd scores, France K20 rows, 0.986/0.988, rule files),
        competition data satwiksps/amazon-ml-challenge-2026 (raw test TSVs for the validator / rule)."""
import glob
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import polars as pl

g = lambda p: sorted(glob.glob(f"/kaggle/input/**/{p}", recursive=True))[0]
D = os.path.dirname(g("sub1_manifest.json"))
TEST_DIR = os.path.dirname(g("test_source1.tsv"))
W, O = "/kaggle/working/work", "/kaggle/working/out"
os.makedirs(f"{W}/runs/v6cd", exist_ok=True)
os.makedirs(O, exist_ok=True)
for f in ("test_s1.parquet", "test_s2.parquet", "test_s3.parquet"):
    shutil.copyfile(f"{D}/{f}", f"{W}/{f}")
for f in ("test_scores_ce.parquet", "decision_ce.json"):
    shutil.copyfile(f"{D}/v6cd_{f}", f"{W}/runs/v6cd/{f}")
os.environ.update(ER_WORK=W, ER_OUT=O, ER_DATA=os.path.dirname(TEST_DIR))
sys.path.insert(0, f"{D}/code")
from er.decide import DecisionParams, select  # noqa: E402
from er.output import write_submission  # noqa: E402


def id_maps(split):   # same as er.pipeline.id_maps (inlined: er.pipeline needs rapidfuzz, absent offline)
    s1 = pl.read_parquet(f"{W}/{split}_s1.parquet", columns=["idx", "entity_id", "country"]).rename({"idx": "s1_idx", "entity_id": "s1"})
    tg = pl.concat([pl.read_parquet(f"{W}/{split}_s{s}.parquet", columns=["idx", "entity_id"]).rename({"idx": "t_idx", "entity_id": "tid"})
                    .with_columns(pl.lit(int(s), pl.UInt8).alias("src")) for s in ("2", "3")])
    return s1, tg

# ---- A) our v6cd test predictions -------------------------------------------------------------
best = json.load(open(f"{W}/runs/v6cd/decision_ce.json"))["best"]
d = pl.read_parquet(f"{W}/runs/v6cd/test_scores_ce.parquet").select("s1_idx", "src", "t_idx", "p")
sel = select(d, DecisionParams(margin=best["margin"], m0=best["m0"], empty_bias=best["empty_bias"], coh=best.get("coh", 0.0)))
s1_ids, tg_ids = id_maps("test")
to_ids = lambda x: (x.join(s1_ids.select("s1_idx", "s1"), on="s1_idx").join(tg_ids, on=["t_idx", "src"]).select("s1", "tid"))
write_submission(to_ids(sel), to_ids(d.select("s1_idx", "src", "t_idx")), s1_ids["s1"], __import__("pathlib").Path(f"{O}/v6cd"))
print("v6cd predicted:", sel.height, "pairs", flush=True)
del d, sel


# ---- helpers ------------------------------------------------------------------------------------
def rd(path, col="m"):
    raw = open(path, "rb").read()
    if raw[:2] == b"PK":
        z = zipfile.ZipFile(io.BytesIO(raw))
        raw = z.read(next(n for n in z.namelist() if n.endswith(".tsv")))
    x = pl.read_csv(io.BytesIO(raw), separator="\t", quote_char=None, infer_schema=False)
    return x.rename({x.columns[0]: "s1", x.columns[1]: col}).with_columns(pl.col(col).fill_null(""))


pairs = lambda x, col="m": x.with_columns(pl.col(col).str.split(",")).explode(col).filter(pl.col(col) != "").select("s1", pl.col(col).alias("t"))
fr = pl.read_parquet(f"{W}/test_s1.parquet", columns=["entity_id", "country"]).filter(pl.col("country") == "France").select(pl.col("entity_id").alias("s1"))
b88 = rd(g("matching_results_0.988_balaji*"))
p88, p86 = pairs(b88), pairs(rd(g("matching_results_0.986_on_unstop*")))
k20 = pl.read_parquet(f"{D}/fr_k20_matches.parquet")                      # s1, t (France rows of our v6cef2_k20 run)
rem_balaji = p86.join(p88, on=["s1", "t"], how="anti")
add = (k20.join(p86, on=["s1", "t"], how="anti").join(rem_balaji, on=["s1", "t"], how="anti").join(p88, on=["s1", "t"], how="anti"))
add = add.join(p88.select("t").unique(), on="t", how="anti").unique(["t"], keep="first")
print("France K20 additions before rule:", add.height, flush=True)


def build(name, base_rows):
    """base_rows: s1, m for all S1 (US/India choice) with 0.988 France rows; adds K20 France pairs, rule, pifit, validator."""
    out = f"{O}/{name}"
    os.makedirs(out, exist_ok=True)
    allp = pl.concat([pairs(base_rows), add])
    m = allp.group_by("s1", maintain_order=True).agg(pl.col("t").str.join(","))
    rows = base_rows.select("s1").join(m.rename({"t": "m"}), on="s1", how="left").with_columns(pl.col("m").fill_null(""))
    pre = f"{out}/pre.tsv"
    rows.rename({"s1": "source1_entity_id", "m": "matched_entity_ids"}).write_csv(pre, separator="\t", quote_style="never")
    subprocess.run([sys.executable, f"{D}/france_rule_apply.py", "--in", pre, "--out", f"{out}/matching_results.tsv", "--test-dir", TEST_DIR], check=True)
    os.remove(pre)
    final = pairs(rd(f"{out}/matching_results.tsv"))
    added, removed = final.join(p88, on=["s1", "t"], how="anti"), p88.join(final, on=["s1", "t"], how="anti")
    rep = {"added_vs_0988": added.height, "removed_vs_0988": removed.height,
           "added_france": added.join(fr, on="s1", how="semi").height}
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
        rep["pifit_share_false_added_france"] = float((da * do).sum() / (da * da).sum())
    cty = pl.read_parquet(f"{W}/test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
    cnt = cty.join(final.group_by("s1").len("k"), on="s1", how="left").with_columns(pl.col("k").fill_null(0))
    rep["structure"] = {c: {"matches_per_s1": round(float(x["k"].mean()), 4), "empty": round(float((x["k"] == 0).mean()), 4)} for (c,), x in cnt.group_by("country")}
    # candidates: our v6cd candidates (US/India) / 0.988-era France candidates -> union with final matches
    oc = rd(f"{O}/v6cd/candidate_pairs.tsv", "c")
    c = (rows.select("s1").join(oc, on="s1", how="left").with_columns(pl.col("c").fill_null(""))
             .join(final.group_by("s1").agg(pl.col("t").str.join(",").alias("fm")), on="s1", how="left")
             .with_columns(pl.concat_list(pl.col("c").str.split(","), pl.col("fm").fill_null("").str.split(","))
                             .list.eval(pl.element().filter(pl.element() != "")).list.unique(maintain_order=True).list.join(",").alias("c"))
             .select("s1", "c"))
    c.rename({"s1": "source1_entity_id", "c": "candidate_entity_ids"}).write_csv(f"{out}/candidate_pairs.tsv", separator="\t", quote_style="never")
    v = subprocess.run([sys.executable, f"{D}/validate_submission.py", "--matching", f"{out}/matching_results.tsv",
                        "--candidate", f"{out}/candidate_pairs.tsv", "--test-dir", TEST_DIR, "--check-ids"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    rep["validator"] = "PASS" if "PASS" in v.stdout else v.stdout[-800:]
    for f in ("matching_results", "candidate_pairs"):
        with zipfile.ZipFile(f"/kaggle/working/{name}_{f}.zip", "w", zipfile.ZIP_DEFLATED) as zz:
            zz.write(f"{out}/{f}.tsv", f"{f}.tsv")
    shutil.rmtree(out)
    print(name, json.dumps(rep, indent=1), flush=True)
    json.dump(rep, open(f"/kaggle/working/{name}_report.json", "w"), indent=1)


ours = rd(f"{O}/v6cd/matching_results.tsv")
base_usin = b88.select("s1").join(pl.concat([ours.join(fr, on="s1", how="anti"), b88.join(fr, on="s1", how="semi")]), on="s1", how="left").with_columns(pl.col("m").fill_null(""))
build("sub1_usin", base_usin)
build("sub1_fr", b88)
shutil.rmtree(O)
shutil.rmtree(W)
print("DONE", flush=True)
