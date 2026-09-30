# Run from src/: ER_WORK=../../../work_v4 python ../scripts/analysis/coh_test_profile.py <run> <kind> <m0> <eb> <margin> <coh,...>  (label-free test profile vs the country-independent true match-count distribution)
import sys, json
import polars as pl
from er import config as C
from er.decide import DecisionParams, select, select_exact
run, kind, m0, eb, margin = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
TRUE = {0: 0.0558, 1: 0.054, 2: 0.17, 3: 0.2405, 4: 0.2194, 5: 0.1459, 6: 0.0747, 7: 0.029, 8: 0.0106}
c = pl.read_parquet(C.WORK / "test_s1.parquet", columns=["idx", "country"]).rename({"idx": "s1_idx"})
t = pl.read_parquet(C.WORK / "runs" / run / "test_scores.parquet").select("s1_idx", "src", "t_idx", "p")
fn = select_exact if kind == "exact" else select
for coh in [float(x) for x in sys.argv[6].split(",")]:
    sel = fn(t, DecisionParams(margin=margin, m0=m0, empty_bias=eb, coh=coh))
    n = c.join(sel.group_by("s1_idx").len("n"), on="s1_idx", how="left").with_columns(pl.col("n").fill_null(0).clip(0, 8))
    for (ct,), g in sorted(n.group_by("country"), key=lambda x: x[0]):
        h = g.group_by("n").len().with_columns(pl.col("len") / g.height)
        h = dict(zip(h["n"].to_list(), h["len"].to_list()))
        tv = 0.5 * sum(abs(h.get(k, 0) - TRUE[k]) for k in TRUE)
        print(f"coh {coh:4.2f} {ct:6s} mean {g['n'].mean():.3f} empty {h.get(0,0):.4f} TV {tv:.4f}")
