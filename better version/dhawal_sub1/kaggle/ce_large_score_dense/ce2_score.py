"""Kaggle GPU kernel: second-opinion cross-encoder (multilingual-e5-base, trained with Zayaan's ce_train.py recipe)
scores the dense-retrieval pairs. Same text/tokenisation/scoring loop as dense.py's cross-encoder step.
Inputs: dense pairs dataset (dense_{fit1,val,test}.parquet), dense text data (train/test_s{1,2,3}.parquet), CE2 model.
Output: /kaggle/working/ce3_{part}.parquet with s1_idx, src, t_idx, ce2."""
import glob
import os
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
inp = lambda pat: glob.glob(f"/kaggle/input/**/{pat}", recursive=True)[0]
D = os.path.dirname(inp("train_roles.parquet"))
P = os.path.dirname(inp("dense_val.parquet"))
cdir = os.path.dirname(inp("model/config.json"))
tok = AutoTokenizer.from_pretrained(cdir)
ce = AutoModelForSequenceClassification.from_pretrained(cdir).half().cuda().eval()
txt = {}
for split in ("train", "test"):
    txt[(split, 1)] = pl.read_parquet(f"{D}/{split}_s1.parquet", columns=["idx", "text"]).rename({"idx": "s1_idx", "text": "a"}).with_columns(pl.col("s1_idx").cast(pl.UInt32))
    for s in (2, 3):
        txt[(split, s)] = pl.read_parquet(f"{D}/{split}_s{s}.parquet", columns=["idx", "text"]).rename({"idx": "t_idx", "text": "b"}).with_columns(pl.col("t_idx").cast(pl.UInt32), pl.lit(s).cast(pl.UInt8).alias("src"))
for part in ("val", "fit1", "test"):
    split = "test" if part == "test" else "train"
    d = pl.read_parquet(f"{P}/dense_{part}.parquet").select(pl.col("s1_idx").cast(pl.UInt32), pl.col("src").cast(pl.UInt8), pl.col("t_idx").cast(pl.UInt32))
    d = d.join(txt[(split, 1)], on="s1_idx")
    d = pl.concat([d.filter(pl.col("src") == s).join(txt[(split, s)], on=["t_idx", "src"]) for s in (2, 3)])
    order = np.argsort((d["a"].str.len_chars() + d["b"].str.len_chars()).to_numpy())
    A, B = d["a"].to_numpy()[order], d["b"].to_numpy()[order]
    sc = []
    with torch.no_grad():
        for i in range(0, d.height, 512):
            e = tok(list(A[i:i + 512]), list(B[i:i + 512]), truncation=True, max_length=128, padding=True, return_tensors="pt")
            sc.append(torch.sigmoid(ce(**{k: x.cuda() for k, x in e.items()}).logits.float().squeeze(-1)).cpu().numpy())
    s_ = np.empty(d.height, dtype=np.float32)
    s_[order] = np.concatenate(sc) if sc else np.array([], dtype=np.float32)
    d.select("s1_idx", "src", "t_idx").with_columns(pl.Series("ce3", s_)).write_parquet(f"/kaggle/working/ce3_{part}.parquet")
    log("ce3 scored", part, d.height)
log("DONE")
