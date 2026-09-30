"""Kaggle GPU kernel: dense-retrieval candidate augmentation (bi-encoder er-bi-model) + cross-encoder scoring.

For US/India and each source: S1 (train FIT fold 1 + VAL, and test) -> top-K targets by cosine; keep only pairs NOT
in the run's existing candidate set; keep those above a cosine threshold chosen on VAL (retain 95% of new true
pairs); score kept pairs with the fine-tuned cross-encoder (er-ce-train model). Outputs dense_{fit1,val,test}.parquet
with s1_idx, src, t_idx, cos, rank, gap1, ce (+ y for fit1/val)."""
import glob
import json
import os
import time

import numpy as np
import polars as pl
import torch
from sentence_transformers import SentenceTransformer, models
from transformers import AutoModelForSequenceClassification, AutoTokenizer

K, T0 = 10, time.time()
inp = lambda pat: glob.glob(f"/kaggle/input/**/{pat}", recursive=True)[0]
D = os.path.dirname(inp("train_roles.parquet"))
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
bdir = os.path.dirname(inp("bi_model/model.safetensors"))   # both attached models have model.safetensors
tr = models.Transformer(bdir, max_seq_length=96)
bi = SentenceTransformer(modules=[tr, models.Pooling(tr.get_word_embedding_dimension(), "mean")], device="cuda").half()
enc = lambda texts: bi.encode(["query: " + t for t in texts], batch_size=1024, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)

roles = pl.read_parquet(f"{D}/train_roles.parquet").select(pl.col("idx").cast(pl.UInt32), "role", "fold")
q_train = roles.filter(((pl.col("role") == "fit") & (pl.col("fold") == 1)) | (pl.col("role") == "val"))
gt = pl.read_parquet(f"{D}/train_gt.parquet").with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
KEYS = ["s1_idx", "src", "t_idx"]
cast = lambda d: d.with_columns(pl.col("s1_idx").cast(pl.UInt32), pl.col("src").cast(pl.UInt8), pl.col("t_idx").cast(pl.UInt32))
out = []
for split in ("train", "test"):
    s1 = pl.read_parquet(f"{D}/{split}_s1.parquet")
    if split == "train":
        s1 = s1.join(q_train.select("idx"), on="idx", how="semi")
    cand = pl.concat([cast(pl.read_parquet(f"{D}/cand_{n}.parquet")) for n in (("fit", "val") if split == "train" else ("test",))]).unique()
    for src in (2, 3):
        tg = pl.read_parquet(f"{D}/{split}_s{src}.parquet")
        for ctry in ("US", "India"):
            a, t = s1.filter(pl.col("country") == ctry), tg.filter(pl.col("country") == ctry)
            Et, Ea = enc(t["text"].to_list()), enc(a["text"].to_list())
            ti = t["idx"].to_numpy()
            I, S = [], []
            for i in range(0, Ea.shape[0], 256):
                v, ix = torch.topk(Ea[i:i + 256] @ Et.T, K, dim=1)
                I.append(ix.cpu().numpy()); S.append(v.float().cpu().numpy())
            I, S = np.concatenate(I), np.concatenate(S)
            d = pl.DataFrame({"s1_idx": np.repeat(a["idx"].to_numpy(), K), "t_idx": ti[I.ravel()], "cos": S.ravel(),
                              "rank": np.tile(np.arange(1, K + 1), a.height).astype(np.int16),
                              "gap1": (S[:, :1] - S).ravel()}).with_columns(pl.lit(src).cast(pl.UInt8).alias("src"))
            d = cast(d).join(cand, on=KEYS, how="anti").with_columns(pl.lit(split).alias("split"))
            out.append(d)
            log(split, src, ctry, "S1", a.height, "targets", t.height, "new pairs", d.height)
            del Et, Ea
            torch.cuda.empty_cache()
dn = pl.concat(out)
# labels for train-split pairs (FIT fold 1 + VAL) from the ground truth, via entity ids
ids1 = pl.read_parquet(f"{D}/train_s1.parquet", columns=["idx", "entity_id"]).rename({"idx": "s1_idx", "entity_id": "source1_entity_id"})
idt = pl.concat([pl.read_parquet(f"{D}/train_s{s}.parquet", columns=["idx", "entity_id"]).with_columns(pl.lit(s).cast(pl.UInt8).alias("src")) for s in (2, 3)]).rename({"idx": "t_idx", "entity_id": "matched_entity_ids"})
lab = cast(gt.join(ids1.with_columns(pl.col("s1_idx").cast(pl.UInt32)), on="source1_entity_id").join(idt.with_columns(pl.col("t_idx").cast(pl.UInt32)), on="matched_entity_ids").select(KEYS)).with_columns(pl.lit(1, pl.Int8).alias("y"))
dn = dn.join(lab, on=KEYS, how="left").with_columns(pl.col("y").fill_null(0))
role = q_train.select(pl.col("idx").alias("s1_idx"), "role")
dn = dn.join(role, on="s1_idx", how="left").with_columns(pl.when(pl.col("split") == "test").then(pl.lit("test")).when(pl.col("role") == "val").then(pl.lit("val")).otherwise(pl.lit("fit1")).alias("part"))
v = dn.filter(pl.col("part") == "val").sort("cos", descending=True)
tot = int(v["y"].sum())
tau = float(v.filter(pl.col("y") == 1)["cos"].quantile(0.05)) if tot else 1.0
stats = {"val_new_pairs": v.height, "val_new_true": tot, "tau_keep95": tau,
         "val_kept": int((v["cos"] >= tau).sum()), "by_part_new": dict(dn.group_by("part").len().rows())}
dn = dn.filter(pl.col("cos") >= tau)
stats["by_part_kept"] = dict(dn.group_by("part").len().rows())
log(json.dumps(stats))
# cross-encoder scores for kept pairs
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
    d = dn.filter(pl.col("part") == part).join(txt[(split, 1)], on="s1_idx")
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
    d.drop("a", "b").with_columns(pl.Series("ce", s_)).write_parquet(f"/kaggle/working/dense_{part}.parquet")
    log("ce scored", part, d.height)
json.dump(stats, open("/kaggle/working/dense_stats.json", "w"), indent=1)
log("DONE")
