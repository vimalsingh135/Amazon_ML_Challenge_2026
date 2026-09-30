"""Kaggle GPU kernel: dense-retrieval candidates for FRANCE (same bi-encoder, same cosine cut as US/India: 0.5908 kept
95% of new true pairs on validation). France S1 -> top-10 France targets per source, minus existing candidates, cos >= cut,
scored by the cross-encoder. Output dense_france_test.parquet (s1_idx, src, t_idx, cos, rank, gap1, ce)."""
import glob
import os
import time

import numpy as np
import polars as pl
import torch
from sentence_transformers import SentenceTransformer, models
from transformers import AutoModelForSequenceClassification, AutoTokenizer

K, TAU, T0 = 20, 0.5908203125, time.time()   # variant: top-20
g = lambda p: glob.glob(f"/kaggle/input/**/{p}", recursive=True)[0]
bdir = os.path.dirname(g("bi_model/model.safetensors"))
tr = models.Transformer(bdir, max_seq_length=96)
bi = SentenceTransformer(modules=[tr, models.Pooling(tr.get_word_embedding_dimension(), "mean")], device="cuda").half()
enc = lambda x: bi.encode(["query: " + t for t in x], batch_size=1024, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
D = os.path.dirname(g("test_s1.parquet"))
KEYS = ["s1_idx", "src", "t_idx"]
cast = lambda d: d.with_columns(pl.col("s1_idx").cast(pl.UInt32), pl.col("src").cast(pl.UInt8), pl.col("t_idx").cast(pl.UInt32))
s1 = pl.read_parquet(f"{D}/test_s1.parquet")
cand = cast(pl.read_parquet(g("cand_france.parquet")))
Ea, out = enc(s1["text"].to_list()), []
for src in (2, 3):
    t = pl.read_parquet(f"{D}/test_s{src}.parquet")
    Et = enc(t["text"].to_list())
    I, S = [], []
    for i in range(0, Ea.shape[0], 256):
        v, ix = torch.topk(Ea[i:i + 256] @ Et.T, K, dim=1)
        I.append(ix.cpu().numpy()); S.append(v.float().cpu().numpy())
    I, S = np.concatenate(I), np.concatenate(S)
    d = pl.DataFrame({"s1_idx": np.repeat(s1["idx"].to_numpy(), K), "t_idx": t["idx"].to_numpy()[I.ravel()], "cos": S.ravel(),
                      "rank": np.tile(np.arange(1, K + 1), s1.height).astype(np.int16), "gap1": (S[:, :1] - S).ravel()}).with_columns(pl.lit(src).cast(pl.UInt8).alias("src"))
    d = cast(d).join(cand, on=KEYS, how="anti").filter(pl.col("cos") >= TAU)
    out.append(d.join(t.select(pl.col("idx").cast(pl.UInt32).alias("t_idx"), pl.col("text").alias("b")), on="t_idx"))
    print("src", src, "new kept", d.height, f"{time.time() - T0:.0f}s", flush=True)
    del Et; torch.cuda.empty_cache()
d = pl.concat(out).join(s1.select(pl.col("idx").cast(pl.UInt32).alias("s1_idx"), pl.col("text").alias("a")), on="s1_idx")
cdir = os.path.dirname(g("model/config.json"))
tok = AutoTokenizer.from_pretrained(cdir)
ce = AutoModelForSequenceClassification.from_pretrained(cdir).half().cuda().eval()
order = np.argsort((d["a"].str.len_chars() + d["b"].str.len_chars()).to_numpy())
A, B, sc = d["a"].to_numpy()[order], d["b"].to_numpy()[order], []
with torch.no_grad():
    for i in range(0, d.height, 512):
        e = tok(list(A[i:i + 512]), list(B[i:i + 512]), truncation=True, max_length=128, padding=True, return_tensors="pt")
        sc.append(torch.sigmoid(ce(**{k: x.cuda() for k, x in e.items()}).logits.float().squeeze(-1)).cpu().numpy())
s_ = np.empty(d.height, dtype=np.float32); s_[order] = np.concatenate(sc) if sc else np.array([], dtype=np.float32)
d.drop("a", "b").with_columns(pl.Series("ce", s_)).write_parquet("/kaggle/working/dense_france_test.parquet")
print("DONE", d.height, f"{time.time() - T0:.0f}s", flush=True)
