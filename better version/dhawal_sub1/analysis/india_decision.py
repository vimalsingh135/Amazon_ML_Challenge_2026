"""India lever 1: India-specific decision parameters on top of Zayaan's decision layer (er.decide.select).

His tune picks ONE (kind, m0, empty_bias, margin, coh) for US+India together. India has lower recall and more
same-name ambiguity, so its optimum may differ. Country stays out of every MODEL; only the decision knobs
are chosen per labelled country (France would keep the global ones). Two-fold cross-fitted over val S1:
tune on one half's India S1, score the other half, compare with the global parameters.
Usage: python india_decision.py [run]   (default v6ce; v6cd once its scores are available)
"""
import itertools
import json
import os
import sys

ROOT = "E:/projects/Amazon ML challenge"
os.environ.setdefault("ER_WORK", ROOT + "/.worktrees/v6val/work_v6")
os.environ.setdefault("ER_DATA", ROOT + "/student_resource/dataset")
os.environ.setdefault("POLARS_MAX_THREADS", "6")
sys.path.insert(0, ROOT + "/.worktrees/ce/code/business_entity_resolution/src")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from er import config as C  # noqa: E402
from er.decide import DecisionParams, select, select_exact  # noqa: E402
from er.pipeline import id_maps, train_roles  # noqa: E402
from er.prep import load_gt  # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else "v6ce"
K = ["s1_idx", "src", "t_idx"]
rd = C.WORK / "runs" / RUN
best = json.loads((rd / "decision_ce.json").read_text())["best"]
G = DecisionParams(margin=best["margin"], m0=best["m0"], empty_bias=best["empty_bias"], coh=best["coh"])
GKIND = best.get("kind", "ratio")
val = pl.read_parquet(rd / "val_scores_ce.parquet").select(K + ["p"])
roles = train_roles().filter(pl.col("role") == "val").select(pl.col("idx").alias("s1_idx"), "country")
roles = roles.with_columns(((pl.col("s1_idx") // 7) % 2).alias("fold"))
s1m, tg = id_maps("train")
truth = load_gt().join(s1m.select("s1_idx", "s1"), on="s1").join(tg, on="tid").select(K).join(roles.select("s1_idx"), on="s1_idx")
nt = truth.group_by("s1_idx").len("nt")


def per_s1(sel, s1s):
    tp = sel.join(truth, on=K).group_by("s1_idx").len("tp")
    x = (s1s.join(nt, on="s1_idx", how="left").join(sel.group_by("s1_idx").len("np"), on="s1_idx", how="left")
            .join(tp, on="s1_idx", how="left").fill_null(0))
    f = pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0).otherwise(
        1.25 * pl.col("tp") / (0.25 * pl.col("nt") + pl.col("np")).clip(lower_bound=1e-9))
    return x.with_columns(f.alias("f")).sort("s1_idx")


def run(kind, prm, d):
    return (select_exact if kind == "exact" else select)(d, prm)


# exclusivity couples S1 only through shared targets; targets never cross countries, so India can be
# decided on its own rows without changing anything for the US.
ind = roles.filter(pl.col("country") == "India")
vi = val.join(ind.select("s1_idx", "fold"), on="s1_idx")
GRID = [(k, DecisionParams(margin=mg, m0=m0, empty_bias=eb, coh=coh))
        for k in ("ratio",) for mg in (0.0, 0.05, 0.1) for m0 in (0.0, 0.25, 0.5, 0.75, 1.0)
        for eb in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0) for coh in (0.5, 1.0)]
print("grid", len(GRID), "global", GKIND, vars(G), flush=True)
fs_g, fs_i, chosen = [], [], []
for f in (0, 1):
    tr, te = vi.filter(pl.col("fold") != f).select(K + ["p"]), vi.filter(pl.col("fold") == f).select(K + ["p"])
    s_tr, s_te = ind.filter(pl.col("fold") != f), ind.filter(pl.col("fold") == f)
    sc = [(float(per_s1(run(k, p, tr), s_tr)["f"].mean()), k, p) for k, p in GRID]
    sc.sort(key=lambda x: -x[0])
    _, k, p = sc[0]
    chosen.append({"kind": k, **vars(p)})
    fs_g.append(per_s1(run(GKIND, G, te), s_te))
    fs_i.append(per_s1(run(k, p, te), s_te))
    print(f, "india-tuned", chosen[-1], "train", round(sc[0][0], 5), flush=True)
g = pl.concat(fs_g).sort("s1_idx")["f"].to_numpy()
i = pl.concat(fs_i).sort("s1_idx")["f"].to_numpy()
diff = i - g
rng = np.random.default_rng(0)
bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
rep = {"run": RUN, "india_global": float(g.mean()), "india_tuned": float(i.mean()), "delta": float(diff.mean()),
       "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "chosen": chosen,
       "lb_effect": round(0.468 * float(diff.mean()), 6)}
print(json.dumps(rep, indent=1))
open(ROOT + f"/.worktrees/v6val/out/india_decision_{RUN}.json", "w").write(json.dumps(rep, indent=1))
