"""Stage the K=20 dense variant for account 3 (dhawal2209000): copy dense data, write dataset + kernel metadata,
and a dense.py identical to account 1's except K = 20."""
import json
import os
import shutil

R = "E:/projects/Amazon ML challenge/.worktrees"
src, dst = f"{R}/rep/dense_data", f"{R}/rep3/dense_data"
os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    if f.endswith(".parquet") and not os.path.exists(f"{dst}/{f}"):
        shutil.copyfile(f"{src}/{f}", f"{dst}/{f}")
json.dump({"title": "almc-dense-data", "id": "dhawal2209000/almc-dense-data", "licenses": [{"name": "CC0-1.0"}]},
          open(f"{dst}/dataset-metadata.json", "w"))
kd = f"{R}/rep3/k_dense"
os.makedirs(kd, exist_ok=True)
code = open(f"{R}/rep/k_dense/dense.py", encoding="utf-8").read()
assert "K, T0 = 10, time.time()" in code
open(f"{kd}/dense.py", "w", encoding="utf-8").write(code.replace("K, T0 = 10, time.time()", "K, T0 = 20, time.time()   # variant: top-20"))
json.dump({"id": "dhawal2209000/almc-dense-k20", "title": "almc-dense-k20", "code_file": "dense.py", "language": "python",
           "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": False,
           "machine_shape": "NvidiaTeslaT4",
           "dataset_sources": ["dhawal2209000/almc-dense-data", "dhawal2209000/almc-bi-model", "dhawal2209000/almc-ce-model"],
           "competition_sources": [], "kernel_sources": []}, open(f"{kd}/kernel-metadata.json", "w"), indent=1)
print("staged", sorted(os.listdir(dst)), "and", sorted(os.listdir(kd)))
