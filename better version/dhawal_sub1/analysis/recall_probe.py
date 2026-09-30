"""Recall probe: could a char-trigram TF-IDF nearest-neighbour channel recover v6ce's retrieval misses?

For a sample of val retrieval misses (true link never scored), rank the true target among ALL targets
of the same country and source by (a) name cosine, (b) 0.6*name + 0.4*address cosine. If the true
target is usually in the top few, a fuzzy retrieval channel is worth building; if not, recall is
exhausted. Null-address and Indic-script misses are reported separately (known to be hard).
Usage (ER_WORK/ER_DATA set, src on PYTHONPATH): python recall_probe.py [n_sample]
"""
import json
import re
import sys
import unicodedata

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

from er import config as C
from er.pipeline import id_maps, train_roles
from er.prep import load_gt

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
K = ["s1_idx", "src", "t_idx"]
INDIC = re.compile("[ऀ-ൿ]")


def fold(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()


val = pl.read_parquet(C.WORK / "runs" / "v6ce" / "val_scores.parquet").select(K)
roles = train_roles().filter(pl.col("role") == "val").select(pl.col("idx").alias("s1_idx"), "country")
s1m, tg = id_maps("train")
truth = (load_gt().join(s1m.select("s1_idx", "s1"), on="s1").join(tg, on="tid").select(K)
         .join(roles, on="s1_idx"))
miss = truth.join(val, on=K, how="anti")


def rd(s):
    return (pl.read_parquet(C.WORK / f"train_s{s}.parquet", columns=["idx", "business_name", "business_address", "country"])
              .with_columns(pl.col("idx").cast(pl.UInt32)))


S1 = rd(1).rename({"idx": "s1_idx", "business_name": "n1", "business_address": "a1"}).drop("country")
miss = miss.with_columns(pl.col("s1_idx").cast(pl.UInt32), pl.col("t_idx").cast(pl.UInt32))
hv = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=2 ** 20, alternate_sign=False, norm=None)
res = []
rng = np.random.default_rng(0)
for src in (2, 3):
    T = rd(src)
    for country in ("US", "India"):
        m = miss.filter((pl.col("src") == src) & (pl.col("country") == country)).join(S1, on="s1_idx")
        m = m.join(T.rename({"idx": "t_idx", "business_name": "nm", "business_address": "am"}).drop("country"), on="t_idx")
        m = m.with_columns(pl.col("am").fill_null("").str.strip_chars().eq("").alias("null_addr"),
                           pl.col("nm").fill_null("").map_elements(lambda x: bool(INDIC.search(x)), return_dtype=pl.Boolean).alias("indic"))
        if m.height == 0:
            continue
        m = m.sample(min(m.height, N // 4), seed=0)
        pool = T.filter(pl.col("country") == country)
        tidx = pool["idx"].to_numpy()
        names = [fold(x) for x in pool["business_name"].to_list()]
        addrs = [fold(x) for x in pool["business_address"].to_list()]
        tfn = TfidfTransformer(sublinear_tf=True).fit(hv.transform(names))
        tfa = TfidfTransformer(sublinear_tf=True).fit(hv.transform(addrs))
        Xn = tfn.transform(hv.transform(names)).T.tocsr()      # F x N
        Xa = tfa.transform(hv.transform(addrs)).T.tocsr()
        pos = {int(t): i for i, t in enumerate(tidx)}
        qn = tfn.transform(hv.transform([fold(x) for x in m["n1"].to_list()]))
        qa = tfa.transform(hv.transform([fold(x) for x in m["a1"].to_list()]))
        tcol = np.array([pos[int(t)] for t in m["t_idx"].to_list()])
        for i0 in range(0, m.height, 10):   # 10 x ~3M dense float32 = 120 MB per matrix
            sn = (qn[i0:i0 + 10] @ Xn).toarray().astype(np.float32)
            sa = (qa[i0:i0 + 10] @ Xa).toarray().astype(np.float32)
            sc = 0.6 * sn + 0.4 * sa
            for j in range(sn.shape[0]):
                t = tcol[i0 + j]
                res.append({"country": country, "src": src,
                            "null_addr": m["null_addr"][i0 + j], "indic": m["indic"][i0 + j],
                            "rank_name": int((sn[j] > sn[j, t]).sum()) + 1,
                            "rank_comb": int((sc[j] > sc[j, t]).sum()) + 1,
                            "cos_name": float(sn[j, t])})
        print(country, src, "sampled", m.height, "pool", len(tidx), flush=True)
        del Xn, Xa
r = pl.DataFrame(res)
r.write_parquet(C.WORK.parent / "out" / "recall_probe_raw.parquet")   # save before any printing
grp = pl.when(pl.col("indic")).then(pl.lit("indic")).when(pl.col("null_addr")).then(pl.lit("null_addr")).otherwise(pl.lit("normal"))
out = (r.with_columns(grp.alias("kind")).group_by(["country", "kind"]).agg(
    pl.len().alias("n"),
    (pl.col("rank_comb") <= 1).mean().round(3).alias("comb_top1"),
    (pl.col("rank_comb") <= 5).mean().round(3).alias("comb_top5"),
    (pl.col("rank_comb") <= 20).mean().round(3).alias("comb_top20"),
    (pl.col("rank_name") <= 5).mean().round(3).alias("name_top5"),
    pl.col("cos_name").median().round(3).alias("med_cos_name")).sort(["country", "kind"]))
pl.Config.set_tbl_cols(12)
print(out)
