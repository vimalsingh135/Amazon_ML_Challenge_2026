# Run from src/: ER_WORK=../../../work_v4 python ../scripts/analysis/coh_tune.py <run>  (decision grid incl. one-to-one coherence)
import json, sys, time
import polars as pl
from er import config as C
from er.decide import DecisionParams, select, select_exact
from er.metric import macro_f05
from er.pipeline import train_roles, id_maps
from er.prep import load_gt
run = sys.argv[1]
rd = C.WORK / "runs" / run
val = pl.read_parquet(rd / "val_scores.parquet")
roles = train_roles().select(pl.col("idx").alias("s1_idx"), "role", "country")
s1_ids, tg_ids = id_maps("train")
val_s1 = roles.filter(pl.col("role") == "val").join(s1_ids.select("s1_idx", "s1"), on="s1_idx")
truth = load_gt().join(val_s1.select("s1"), on="s1")
def score(sel, s1s=val_s1, tr=truth):
    pred = sel.join(s1s.select("s1_idx", "s1"), on="s1_idx").join(tg_ids, on=["t_idx", "src"])
    return macro_f05(pred.select("s1", "tid"), tr, s1s["s1"])
rows = []
for kind, fn in (("ratio", select), ("exact", select_exact)):
    for m0 in (0.5, 0.75):
        for eb in (2.0, 2.5, 3.0):
            for margin in (0.05, 0.1):
                for coh in (0.0, 0.5, 1.0):
                    t = time.time(); r = score(fn(val, DecisionParams(margin=margin, m0=m0, empty_bias=eb, coh=coh)))
                    rows.append({"kind": kind, "m0": m0, "eb": eb, "margin": margin, "coh": coh,
                                 **{k: round(v, 5) for k, v in r.items() if k in ("f05", "prec", "rec", "singleton_f")}})
res = pl.DataFrame(rows).sort("f05", descending=True)
res.write_csv(rd / "tune_coh.csv")
pl.Config.set_tbl_rows(12); pl.Config.set_tbl_width_chars(200)
print(res.head(10))
print(res.group_by("coh").agg(pl.col("f05").max()).sort("coh"))
