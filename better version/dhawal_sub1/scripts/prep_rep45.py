"""Stage account 4 (dhawal22092004: dense K=50 keep 99%) and account 5 (dhwalkhatri12873: second CE, e5-base).
Data files are hard-linked (no extra disk). Kernel code = Zayaan's scripts with only the stated constants changed."""
import json
import os

W = "E:/projects/Amazon ML challenge/.worktrees"


def link_dir(src, dst, ds_id):
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if f.endswith(".parquet") and not os.path.exists(f"{dst}/{f}"):
            os.link(f"{src}/{f}", f"{dst}/{f}")
    json.dump({"title": ds_id.split("/")[1], "id": ds_id, "licenses": [{"name": "CC0-1.0"}]}, open(f"{dst}/dataset-metadata.json", "w"))


def kernel(folder, kid, code_file, code, datasets, kernels=(), internet=False):
    os.makedirs(folder, exist_ok=True)
    open(f"{folder}/{code_file}", "w", encoding="utf-8").write(code)
    json.dump({"id": kid, "title": kid.split("/")[1], "code_file": code_file, "language": "python", "kernel_type": "script",
               "is_private": True, "enable_gpu": True, "enable_internet": internet, "machine_shape": "NvidiaTeslaT4",
               "dataset_sources": list(datasets), "competition_sources": [], "kernel_sources": list(kernels)},
              open(f"{folder}/kernel-metadata.json", "w"), indent=1)


# account 4: K=50 superset (same as the account-3 script, K changed)
link_dir(f"{W}/rep/dense_data", f"{W}/rep4/dense_data", "dhawal22092004/almc-dense-data")
k20 = open(f"{W}/rep3/k_dense/dense.py", encoding="utf-8").read()
assert "K, T0 = 20, time.time()" in k20
kernel(f"{W}/rep4/k_dense", "dhawal22092004/almc-dense-k50", "dense.py",
       k20.replace("K, T0 = 20, time.time()   # variant: top-20", "K, T0 = 50, time.time()   # variant: top-50"),
       ["dhawal22092004/almc-dense-data", "dhawal22092004/almc-bi-model", "dhawal22092004/almc-ce-model"])

# account 5: second cross-encoder, different backbone (multilingual-e5-base, MIT), same data and recipe
link_dir(f"{W}/rep/ce_data", f"{W}/rep5/ce_data", "dhwalkhatri12873/almc-ce-data")
ce = open(f"{W}/final2/code/business_entity_resolution/scripts/kaggle/train/ce_train.py", encoding="utf-8").read()
old = 'MODEL, MAXLEN, BS, LR, EPOCHS, SEED = "xlm-roberta-base", 128, 64, 2e-5, 2, 0'
assert old in ce
kernel(f"{W}/rep5/k_ce2", "dhwalkhatri12873/almc-ce2-train", "ce_train.py",
       ce.replace(old, 'MODEL, MAXLEN, BS, LR, EPOCHS, SEED = "intfloat/multilingual-e5-base", 128, 64, 2e-5, 2, 0   # second opinion'),
       ["dhwalkhatri12873/almc-ce-data"], internet=True)
print("staged rep4 + rep5")
