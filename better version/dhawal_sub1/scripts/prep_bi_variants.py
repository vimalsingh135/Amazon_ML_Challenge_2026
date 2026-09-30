"""Bi-encoder variants on the free slots of accounts 3 and 4 (same hard-negative triplets as account 2's bi2):
  acct3 dhawal2209000 : multilingual-e5-base (MIT), 1 epoch, batch 64
  acct4 dhawal22092004: multilingual-e5-small, 2 epochs
Code = the account-2 bi2 script (Zayaan's bi_train.py + triplets) with only MODEL/BS/EPOCHS changed."""
import json
import os

W = "E:/projects/Amazon ML challenge/.worktrees"
base = open(f"{W}/rep2/k_bi2/bi_train.py", encoding="utf-8").read()
old = 'MODEL, BS, EPOCHS, K = "intfloat/multilingual-e5-small", 128, 1, 50'
assert old in base
for user, rep, kid, new in (("dhawal2209000", "rep3", "almc-bi3-base", 'MODEL, BS, EPOCHS, K = "intfloat/multilingual-e5-base", 64, 1, 50   # variant: e5-base'),
                            ("dhawal22092004", "rep4", "almc-bi4-2ep", 'MODEL, BS, EPOCHS, K = "intfloat/multilingual-e5-small", 128, 2, 50   # variant: 2 epochs')):
    d = f"{W}/{rep}/bi_hn_data"
    os.makedirs(d, exist_ok=True)
    if not os.path.exists(f"{d}/bi_train_hn.parquet"):
        os.link(f"{W}/rep2/bi_hn_data/bi_train_hn.parquet", f"{d}/bi_train_hn.parquet")
    json.dump({"title": "almc-bi-hn-data", "id": f"{user}/almc-bi-hn-data", "licenses": [{"name": "CC0-1.0"}]}, open(f"{d}/dataset-metadata.json", "w"))
    k = f"{W}/{rep}/k_{kid}"
    os.makedirs(k, exist_ok=True)
    open(f"{k}/bi_train.py", "w", encoding="utf-8").write(base.replace(old, new))
    json.dump({"id": f"{user}/{kid}", "title": kid, "code_file": "bi_train.py", "language": "python", "kernel_type": "script",
               "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
               "dataset_sources": [f"{user}/almc-bi-hn-data"], "competition_sources": [], "kernel_sources": []},
              open(f"{k}/kernel-metadata.json", "w"), indent=1)
    print("staged", user, kid)
