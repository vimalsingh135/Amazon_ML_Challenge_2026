"""Held-out gate between two runs (anti-overfitting): decision params are tuned on one half of val S1 and the
result is scored on the OTHER half, for both runs, then per-S1 F0.5 differences are bootstrapped.
Usage: python honest_gate.py <run_a> <run_b>     (runs under ER_WORK/runs; uses val_scores_ce.parquet)
Accept b over a only if: overall CI low > 0, and US and India mean deltas >= 0 on the held-out halves."""
import json
import os
import sys

R = "E:/projects/Amazon ML challenge"
os.environ.setdefault("ER_WORK", R + "/.worktrees/v6val/work_v6")
os.environ.setdefault("ER_DATA", R + "/student_resource/dataset")
os.environ.setdefault("POLARS_MAX_THREADS", "6")
sys.path.insert(0, R + "/.worktrees/final2/code/business_entity_resolution/src")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from er import config as C  # noqa: E402
from er.decide import DecisionParams, select  # noqa: E402
from er.pipeline import id_maps, train_roles  # noqa: E402
from er.prep import load_gt  # noqa: E402

A, B = sys.argv[1], sys.argv[2]
K = ["s1_idx", "src", "t_idx"]
roles = train_roles().filter(pl.col("role") == "val").select(pl.col("idx").alias("s1_idx"), "country")
roles = roles.with_columns(((pl.col("s1_idx") * 2654435761) % 1000 < 500).cast(pl.Int8).alias("half"))
s1m, tg = id_maps("train")
truth = load_gt().join(s1m.select("s1_idx", "s1"), on="s1").join(tg, on="tid").select(K).join(roles.select("s1_idx"), on="s1_idx")
nt = truth.group_by("s1_idx").len("nt")
GRID = [DecisionParams(margin=mg, m0=m0, empty_bias=eb, coh=1.0)
        for mg in (0.0, 0.05, 0.1) for m0 in (0.0, 0.25, 0.5, 0.75) for eb in (1.5, 2.0, 2.5, 3.0)]


def per_s1(sel, s1s):
    tp = sel.join(truth, on=K).group_by("s1_idx").len("tp")
    x = (s1s.join(nt, on="s1_idx", how="left").join(sel.group_by("s1_idx").len("np"), on="s1_idx", how="left")
            .join(tp, on="s1_idx", how="left").fill_null(0))
    f = pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0).otherwise(
        1.25 * pl.col("tp") / (0.25 * pl.col("nt") + pl.col("np")).clip(lower_bound=1e-9))
    return x.with_columns(f.alias("f")).select("s1_idx", "country", "f")


def heldout(run):
    v = pl.read_parquet(C.WORK / "runs" / run / "val_scores_ce.parquet").select(K + ["p"]).join(roles.select("s1_idx", "half"), on="s1_idx")
    parts, chosen = [], []
    for h in (0, 1):
        tr, te = v.filter(pl.col("half") != h).select(K + ["p"]), v.filter(pl.col("half") == h).select(K + ["p"])
        s_tr, s_te = roles.filter(pl.col("half") != h), roles.filter(pl.col("half") == h)
        best = max(GRID, key=lambda g: float(per_s1(select(tr, g), s_tr)["f"].mean()))
        chosen.append(vars(best))
        parts.append(per_s1(select(te, best), s_te))
    return pl.concat(parts).sort("s1_idx"), chosen


fa, pa = heldout(A)
fb, pb = heldout(B)
d = fa.join(fb.select("s1_idx", pl.col("f").alias("fb")), on="s1_idx").with_columns((pl.col("fb") - pl.col("f")).alias("d"))
x = d["d"].to_numpy()
rng = np.random.default_rng(0)
bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
rep = {"a": A, "b": B, "a_heldout_f05": float(d["f"].mean()), "b_heldout_f05": float(d["fb"].mean()),
       "delta": float(x.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
       "by_country_delta": {c: float(g["d"].mean()) for (c,), g in d.group_by("country")}, "params_a": pa, "params_b": pb}
rep["accept"] = rep["ci95"][0] > 0 and all(v >= 0 for v in rep["by_country_delta"].values())
print(json.dumps(rep, indent=1))
out = R + "/.worktrees/v6val/out/honest_gates.json"
allr = json.load(open(out)) if os.path.exists(out) else {}
allr[f"{A}__vs__{B}"] = rep
json.dump(allr, open(out, "w"), indent=1)
