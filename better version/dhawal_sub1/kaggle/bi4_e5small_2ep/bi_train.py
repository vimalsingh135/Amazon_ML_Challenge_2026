"""Kaggle GPU kernel: contrastive bi-encoder for dense candidate retrieval (blocking recall).
multilingual-e5-small (MIT, 118M), MultipleNegativesRankingLoss on FIT positive pairs (er.cross export-biencoder),
then retrieval evaluation on India val: S1 -> top-K targets (same country), recall of true pairs overall and of
the pairs our key-based blocking missed (bi_eval_*.parquet from er.cross export-bi-eval)."""
import glob
import json
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

MODEL, BS, EPOCHS, K = "intfloat/multilingual-e5-small", 128, 2, 50   # variant: 2 epochs
t0 = time.time()
d = pd.read_parquet(glob.glob("/kaggle/input/**/bi_train_hn.parquet", recursive=True)[0])   # v2: hard-negative triplets
ex = [InputExample(texts=["query: " + a, "query: " + b, "query: " + c]) for a, b, c in zip(d.a, d.b, d.c)]
model = SentenceTransformer(MODEL, device="cuda")
model.max_seq_length = 96
dl = DataLoader(ex, shuffle=True, batch_size=BS, drop_last=True)
model.fit(train_objectives=[(dl, losses.MultipleNegativesRankingLoss(model))], epochs=EPOCHS,
          warmup_steps=int(0.05 * len(dl)), use_amp=True, show_progress_bar=False)
model.save("/kaggle/working/bi_model")
print("trained", len(ex), f"{time.time() - t0:.0f}s", flush=True)

res = {"train_pairs": len(ex), "train_seconds": time.time() - t0}
ev = glob.glob("/kaggle/input/**/bi_eval_s1.parquet", recursive=True)
if ev:
    s1 = pd.read_parquet(ev[0])                     # idx, text  (India val S1)
    tg = pd.read_parquet(glob.glob("/kaggle/input/**/bi_eval_tg.parquet", recursive=True)[0])   # src, idx, text
    truth = pd.read_parquet(glob.glob("/kaggle/input/**/bi_eval_truth.parquet", recursive=True)[0])  # s1_idx, src, t_idx, inA
    enc = lambda x: model.encode(["query: " + t for t in x], batch_size=1024, convert_to_tensor=True,
                                 normalize_embeddings=True, show_progress_bar=False).half()
    E1 = enc(s1.text.tolist())
    hits = []
    for s in (2, 3):
        t = tg[tg.src == s].reset_index(drop=True)
        Et = enc(t.text.tolist())
        top = []
        for i in range(0, E1.shape[0], 256):   # 256 x ~2M fp16 scores ~ 1 GB of GPU memory per step
            top.append(torch.topk(E1[i:i + 256] @ Et.T, K, dim=1).indices.cpu().numpy())
        top = np.concatenate(top)
        hits.append(pd.DataFrame({"s1_idx": np.repeat(s1.idx.values, K), "src": s,
                                  "t_idx": t.idx.values[top.ravel()], "rank": np.tile(np.arange(1, K + 1), len(s1))}))
        del Et
    hits = pd.concat(hits)
    hits.to_parquet("/kaggle/working/bi_eval_hits.parquet")
    m = truth.merge(hits, on=["s1_idx", "src", "t_idx"], how="left")
    for k in (10, 20, 50):
        ok = m["rank"].le(k)
        res[f"recall@{k}_all"] = float(ok.mean())
        res[f"recall@{k}_blocking_missed"] = float(ok[~m.inA].mean())
    res["n_true"], res["n_blocking_missed"] = len(m), int((~m.inA).sum())
print(json.dumps(res), flush=True)
json.dump(res, open("/kaggle/working/bi_log.json", "w"), indent=1)
