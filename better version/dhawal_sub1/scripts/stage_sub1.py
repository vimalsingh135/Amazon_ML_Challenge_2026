"""Stage the almc-sub1-data dataset (light: column selects + file copies only)."""
import json
import os
import shutil

import polars as pl

R = "E:/projects/Amazon ML challenge"
WT = R + "/.worktrees"
W = WT + "/v6val/work_v6"
D = WT + "/rep/sub1_data"
os.makedirs(D, exist_ok=True)
shutil.copytree(WT + "/final2/code/business_entity_resolution/src/er", f"{D}/code/er", dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__"))
pl.read_parquet(f"{W}/test_s1.parquet", columns=["idx", "entity_id", "country"]).write_parquet(f"{D}/test_s1.parquet")
for s in (2, 3):
    pl.read_parquet(f"{W}/test_s{s}.parquet", columns=["idx", "entity_id"]).write_parquet(f"{D}/test_s{s}.parquet")
for f in ("test_scores_ce.parquet", "decision_ce.json"):
    shutil.copyfile(f"{W}/runs/v6cd/{f}", f"{D}/v6cd_{f}")
fr = pl.read_parquet(f"{W}/test_s1.parquet", columns=["entity_id", "country"]).filter(pl.col("country") == "France").select(pl.col("entity_id").alias("s1"))
m = pl.read_csv(WT + "/rep/output/v6cef2_k20/matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)
m = m.rename({m.columns[0]: "s1", m.columns[1]: "m"}).join(fr, on="s1", how="semi")
(m.with_columns(pl.col("m").fill_null("").str.split(",")).explode("m").filter(pl.col("m") != "")
  .select("s1", pl.col("m").alias("t")).write_parquet(f"{D}/fr_k20_matches.parquet"))
for src, dst in ((R + "/matching_results_0.988_balaji", "matching_results_0.988_balaji"),
                 (R + "/matching_results_0.986_on_unstop", "matching_results_0.986_on_unstop"),
                 (WT + "/v6val/tmp/france_rule_apply.py", "france_rule_apply.py"),
                 (WT + "/v6val/tmp/fr_vocab300.json", "fr_vocab300.json"),
                 (R + "/student_resource/utils/validate_submission.py", "validate_submission.py")):
    shutil.copyfile(src, f"{D}/{dst}")
json.dump({"what": "submission #1 inputs"}, open(f"{D}/sub1_manifest.json", "w"))
json.dump({"title": "almc-sub1-data", "id": "dhawal2209/almc-sub1-data", "licenses": [{"name": "CC0-1.0"}]},
          open(f"{D}/dataset-metadata.json", "w"))
print({f: round(os.path.getsize(f"{D}/{f}") / 2**20, 1) for f in os.listdir(D) if os.path.isfile(f"{D}/{f}")})
