"""Account 5 (dhwalkhatri12873): CE-large = Zayaan's ce_train.py with xlm-roberta-large (MIT, 560M), 1 epoch,
batch 32, lr 1e-5 (large models diverge at base-model settings). Same ce_train.parquet (FIT fold 0)."""
import json
import os

W = "E:/projects/Amazon ML challenge/.worktrees"
ce = open(f"{W}/final2/code/business_entity_resolution/scripts/kaggle/train/ce_train.py", encoding="utf-8").read()
old = 'MODEL, MAXLEN, BS, LR, EPOCHS, SEED = "xlm-roberta-base", 128, 64, 2e-5, 2, 0'
assert old in ce
ce = ce.replace(old, 'MODEL, MAXLEN, BS, LR, EPOCHS, SEED = "xlm-roberta-large", 128, 32, 1e-5, 1, 0   # CE-large variant')
k = f"{W}/rep5/k_celarge"
os.makedirs(k, exist_ok=True)
open(f"{k}/ce_train.py", "w", encoding="utf-8").write(ce)
json.dump({"id": "dhwalkhatri12873/almc-celarge-train", "title": "almc-celarge-train", "code_file": "ce_train.py",
           "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
           "machine_shape": "NvidiaTeslaT4", "dataset_sources": ["dhwalkhatri12873/almc-ce-data"], "competition_sources": [],
           "kernel_sources": []}, open(f"{k}/kernel-metadata.json", "w"), indent=1)
print("staged", os.listdir(k))
