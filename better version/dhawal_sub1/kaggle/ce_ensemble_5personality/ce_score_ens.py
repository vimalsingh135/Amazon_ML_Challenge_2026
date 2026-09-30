"""Score the uncertain-band pairs (er.cross export-band) with EVERY ensemble member trained by
ce_train_ens.py, and average them. Runs as a Kaggle kernel or as a subprocess of the single-notebook
runner (paths from env vars, Kaggle globs as fallback).

Env: CE_MODELS (dir holding model_*/), CE_BAND (dir holding ce_band_{fit1,val,test}.parquet),
     CE_OUT (output dir).
Output: <CE_OUT>/ce_scores_{name}.parquet with s1_idx, src, t_idx, ce (= mean over members) and one
column per member (ce_<member>). ``er.cross.stack`` reads the ``ce`` column unchanged, so the ensemble
is a drop-in. A split whose output already exists is skipped (resumable).
"""
import glob
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAXLEN, BS = 128, 512
MODELS = os.environ.get("CE_MODELS")
BAND = os.environ.get("CE_BAND")
OUT = os.environ.get("CE_OUT", "/kaggle/working")
os.makedirs(OUT, exist_ok=True)

pat = f"{MODELS}/model_*/config.json" if MODELS else "/kaggle/input/**/model_*/config.json"
members = sorted(os.path.dirname(p) for p in glob.glob(pat, recursive=True))
assert members, f"no model_*/ dirs found ({pat})"
print("members:", [os.path.basename(m) for m in members], flush=True)

for name in ("fit1", "val", "test"):
    out_path = f"{OUT}/ce_scores_{name}.parquet"
    if os.path.exists(out_path):
        print("exists, skip", name, flush=True)
        continue
    paths = [f"{BAND}/ce_band_{name}.parquet"] if BAND else glob.glob(f"/kaggle/input/**/ce_band_{name}.parquet", recursive=True)
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print("no band for", name, "- skipped", flush=True)
        continue
    d = pd.read_parquet(paths[0])
    order = (d.a.str.len() + d.b.str.len()).argsort().values      # length-sort -> less padding
    a, b = d.a.values[order], d.b.values[order]
    cols = {}
    for mdir in members:
        tok = AutoTokenizer.from_pretrained(mdir)
        model = AutoModelForSequenceClassification.from_pretrained(mdir).half().cuda().eval()
        net = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
        out, t0 = [], time.time()
        with torch.no_grad():
            for i in range(0, len(d), BS):
                enc = tok(list(a[i:i + BS]), list(b[i:i + BS]), truncation=True, max_length=MAXLEN,
                          padding=True, return_tensors="pt")
                out.append(torch.sigmoid(net(**{k: v.cuda() for k, v in enc.items()}).logits.float().squeeze(-1)).cpu())
        res = np.zeros(len(d), np.float32)
        res[order] = torch.cat(out).numpy()
        cols[f"ce_{os.path.basename(mdir).replace('model_', '')}"] = res
        del model, net; torch.cuda.empty_cache()
        print(name, os.path.basename(mdir), len(d), f"{time.time()-t0:.0f}s", flush=True)
    df = d[["s1_idx", "src", "t_idx"]].copy()
    for c, v in cols.items():
        df[c] = v
    df["ce"] = df[list(cols)].mean(axis=1).astype("float32")      # ensemble mean = drop-in `ce`
    df.to_parquet(out_path)
    print("done", name, len(d), "members", len(cols), flush=True)
