"""More France dense variants for free slots: acct1 K=50 (bi v1), acct2 K=20 with bi v2 (hard negatives, from the
almc-bi2-train kernel output). Same fr_dense.py as the K=20 run, only K / sources changed."""
import json
import os

W = "E:/projects/Amazon ML challenge/.worktrees"
code20 = open(f"{W}/rep/k_frdense/fr_dense.py", encoding="utf-8").read()
a = "K, TAU, T0 = 20, 0.5908203125, time.time()   # variant: top-20"
assert a in code20


def kernel(folder, kid, code, datasets, kernels):
    os.makedirs(folder, exist_ok=True)
    open(f"{folder}/fr_dense.py", "w", encoding="utf-8").write(code)
    json.dump({"id": kid, "title": kid.split("/")[1], "code_file": "fr_dense.py", "language": "python", "kernel_type": "script",
               "is_private": True, "enable_gpu": True, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
               "dataset_sources": datasets, "competition_sources": [], "kernel_sources": kernels},
              open(f"{folder}/kernel-metadata.json", "w"), indent=1)


kernel(f"{W}/rep/k_frdense50", "dhawal2209/almc-fr-dense-k50",
       code20.replace(a, "K, TAU, T0 = 50, 0.5908203125, time.time()   # variant: top-50"),
       ["dhawal2209/almc-fr-data", "dhawal2209/almc-bi-model"], ["dhawal2209/almc-ce-train"])
# account 2: France data (hard link) + bi v2 via kernel output + CE model dataset already on account 2
d = f"{W}/rep2/fr_data"
os.makedirs(d, exist_ok=True)
for f in os.listdir(f"{W}/rep/fr_data"):
    if f.endswith(".parquet") and not os.path.exists(f"{d}/{f}"):
        os.link(f"{W}/rep/fr_data/{f}", f"{d}/{f}")
json.dump({"title": "almc-fr-data", "id": "codelearner00/almc-fr-data", "licenses": [{"name": "CC0-1.0"}]}, open(f"{d}/dataset-metadata.json", "w"))
kernel(f"{W}/rep2/k_frdense_bi2", "codelearner00/almc-fr-dense-bi2", code20,
       ["codelearner00/almc-fr-data", "codelearner00/almc-ce-model"], ["codelearner00/almc-bi2-train"])
print("staged")
