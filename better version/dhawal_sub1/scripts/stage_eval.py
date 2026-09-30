"""Stage almc-eval-data (light: column selects + copies) for accounts 3 and 4 (hard-linked copies)."""
import json
import os
import shutil

import polars as pl

R = "E:/projects/Amazon ML challenge"
WT = R + "/.worktrees"
W = WT + "/v6val/work_v6"
D = WT + "/rep3/eval_data"
os.makedirs(f"{D}/work", exist_ok=True)
shutil.copytree(WT + "/final2/code/business_entity_resolution/src/er", f"{D}/code/er", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
for split in ("train", "test"):
    pl.read_parquet(f"{W}/{split}_s1.parquet", columns=["idx", "entity_id", "country"]).write_parquet(f"{D}/work/{split}_s1.parquet")
    for s in (2, 3):
        pl.read_parquet(f"{W}/{split}_s{s}.parquet", columns=["idx", "entity_id"]).write_parquet(f"{D}/work/{split}_s{s}.parquet")
shutil.copyfile(f"{W}/train_roles.parquet", f"{D}/work/train_roles.parquet")
for run, files in (("v6ce", ("val_scores.parquet", "val_scores_ce.parquet", "test_scores_ce.parquet", "stage2_f0.lgb", "stage2_f1.lgb",
                             "calibrators.pkl", "summary.json", "decision_ce.json", "decision.json")),
                  ("v6cd", ("val_scores.parquet", "val_scores_ce.parquet", "decision_ce.json", "decision.json", "summary.json",
                            "calibrators.pkl", "stage2_f0.lgb", "stage2_f1.lgb"))):
    os.makedirs(f"{D}/runs_{run}", exist_ok=True)
    for f in files:
        shutil.copyfile(f"{W}/runs/{run}/{f}", f"{D}/runs_{run}/{f}")
json.dump({"variants": [[10, 0.95], [10, 0.99], [15, 0.97], [20, 0.95], [20, 0.97], [20, 0.99]]}, open(f"{D}/eval_manifest.json", "w"))
json.dump({"title": "almc-eval-data", "id": "dhawal2209000/almc-eval-data", "licenses": [{"name": "CC0-1.0"}]}, open(f"{D}/dataset-metadata.json", "w"))
# account 4 copy (K=50 superset -> add K=30/50 variants), hard links
D4 = WT + "/rep4/eval_data"
for root, _, fs in os.walk(D):
    rel = os.path.relpath(root, D)
    os.makedirs(os.path.join(D4, rel), exist_ok=True)
    for f in fs:
        if f not in ("dataset-metadata.json", "eval_manifest.json") and not os.path.exists(os.path.join(D4, rel, f)):
            os.link(os.path.join(root, f), os.path.join(D4, rel, f))
json.dump({"variants": [[30, 0.97], [50, 0.97], [50, 0.99], [20, 0.99]]}, open(f"{D4}/eval_manifest.json", "w"))
json.dump({"title": "almc-eval-data", "id": "dhawal22092004/almc-eval-data", "licenses": [{"name": "CC0-1.0"}]}, open(f"{D4}/dataset-metadata.json", "w"))
tot = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(D) for f in fs) / 2**20
print("staged eval data", round(tot, 1), "MB")
