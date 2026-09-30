"""Stage account 2 (codelearner00): bi-encoder v2 with hard negatives + its own dense K=20/keep-99% run.
bi2 kernel = Zayaan's bi_train.py with (anchor, positive, hard negative) triplets; dense kernel = the account-3 K=20
script, chained on the new bi model via kernel_sources. Data: dense text data and the CE model (hard links)."""
import json
import os
import shutil

W = "E:/projects/Amazon ML challenge/.worktrees"
R2 = f"{W}/rep2"


def meta(folder, ds_id):
    json.dump({"title": ds_id.split("/")[1], "id": ds_id, "licenses": [{"name": "CC0-1.0"}]}, open(f"{folder}/dataset-metadata.json", "w"))


def link_tree(src, dst):
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for f in files:
            if f == "dataset-metadata.json":
                continue
            t = os.path.join(dst, rel, f)
            if not os.path.exists(t):
                os.link(os.path.join(root, f), t)


def kernel(folder, kid, code_file, code, datasets, kernels=(), internet=False):
    os.makedirs(folder, exist_ok=True)
    open(f"{folder}/{code_file}", "w", encoding="utf-8").write(code)
    json.dump({"id": kid, "title": kid.split("/")[1], "code_file": code_file, "language": "python", "kernel_type": "script",
               "is_private": True, "enable_gpu": True, "enable_internet": internet, "machine_shape": "NvidiaTeslaT4",
               "dataset_sources": list(datasets), "competition_sources": [], "kernel_sources": list(kernels)},
              open(f"{folder}/kernel-metadata.json", "w"), indent=1)


# bi-encoder v2 (hard negatives)
bi = open(f"{W}/final2/code/business_entity_resolution/scripts/kaggle/bi/bi_train.py", encoding="utf-8").read()
old_read = 'd = pd.read_parquet(glob.glob("/kaggle/input/**/bi_train.parquet", recursive=True)[0])'
old_ex = 'ex = [InputExample(texts=["query: " + a, "query: " + b]) for a, b in zip(d.a, d.b)]'
assert old_read in bi and old_ex in bi
bi = bi.replace(old_read, 'd = pd.read_parquet(glob.glob("/kaggle/input/**/bi_train_hn.parquet", recursive=True)[0])   # v2: hard-negative triplets')
bi = bi.replace(old_ex, 'ex = [InputExample(texts=["query: " + a, "query: " + b, "query: " + c]) for a, b, c in zip(d.a, d.b, d.c)]')
kernel(f"{R2}/k_bi2", "codelearner00/almc-bi2-train", "bi_train.py", bi, ["codelearner00/almc-bi-hn-data"], internet=True)

# dense text data + CE model for account 2
link_tree(f"{W}/rep/dense_data", f"{R2}/dense_data")
meta(f"{R2}/dense_data", "codelearner00/almc-dense-data")
link_tree(f"{W}/rep/ce_model_ds", f"{R2}/ce_model_ds")
meta(f"{R2}/ce_model_ds", "codelearner00/almc-ce-model")

# dense run with bi v2 (same K=20 / keep-99% script as account 3; bi model comes from the bi2 kernel output)
k20 = open(f"{W}/rep3/k_dense/dense.py", encoding="utf-8").read()
kernel(f"{R2}/k_dense_bi2", "codelearner00/almc-dense-bi2", "dense.py", k20,
       ["codelearner00/almc-dense-data", "codelearner00/almc-ce-model"], ["codelearner00/almc-bi2-train"])
print("staged account 2")
