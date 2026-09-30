"""France dense K=20 (LB probe for later days) on account 1: Zayaan's er.dense.export restricted to France for the
test texts, cand_france = v6ce test candidate keys of France S1, and his fr_dense.py with K=20 and the bi-model
lookup pinned to bi_model/ (two attached models both contain model.safetensors)."""
import json
import os
import sys

R = "E:/projects/Amazon ML challenge"
os.environ.setdefault("ER_WORK", R + "/.worktrees/v6val/work_v6")
os.environ.setdefault("ER_DATA", R + "/student_resource/dataset")
sys.path.insert(0, R + "/.worktrees/final2/code/business_entity_resolution/src")
import polars as pl  # noqa: E402

from er import dense  # noqa: E402

W = R + "/.worktrees"
out = f"{W}/rep/fr_data"
os.makedirs(out, exist_ok=True)
if not os.path.exists(f"{out}/cand_france.parquet"):
    print(dense.export("v6ce_dx", out, countries=("France",)))
    fr = pl.read_parquet(f"{out}/test_s1.parquet").select(pl.col("idx").alias("s1_idx"))
    pl.read_parquet(f"{out}/cand_test.parquet").join(fr, on="s1_idx", how="semi").write_parquet(f"{out}/cand_france.parquet")
    for f in os.listdir(out):   # keep only what fr_dense needs
        if f not in ("test_s1.parquet", "test_s2.parquet", "test_s3.parquet", "cand_france.parquet"):
            os.remove(f"{out}/{f}")
json.dump({"title": "almc-fr-data", "id": "dhawal2209/almc-fr-data", "licenses": [{"name": "CC0-1.0"}]}, open(f"{out}/dataset-metadata.json", "w"))
code = open(f"{W}/final2/code/business_entity_resolution/scripts/kaggle/fr_dense/fr_dense.py", encoding="utf-8").read()
for a, b in (("K, TAU, T0 = 10, 0.5908203125, time.time()", "K, TAU, T0 = 20, 0.5908203125, time.time()   # variant: top-20"),
             ('bdir = os.path.dirname(g("model.safetensors"))', 'bdir = os.path.dirname(g("bi_model/model.safetensors"))')):
    assert a in code, a
    code = code.replace(a, b)
k = f"{W}/rep/k_frdense"
os.makedirs(k, exist_ok=True)
open(f"{k}/fr_dense.py", "w", encoding="utf-8").write(code)
json.dump({"id": "dhawal2209/almc-fr-dense-k20", "title": "almc-fr-dense-k20", "code_file": "fr_dense.py", "language": "python",
           "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
           "dataset_sources": ["dhawal2209/almc-fr-data", "dhawal2209/almc-bi-model"], "competition_sources": [],
           "kernel_sources": ["dhawal2209/almc-ce-train"]}, open(f"{k}/kernel-metadata.json", "w"), indent=1)
print({f: pl.read_parquet(f"{out}/{f}").height for f in os.listdir(out) if f.endswith(".parquet")})
