"""Validation analysis for a trained run: layer recall, calibration, error taxonomy, segments.

python -m er.analysis --run v1            -> work/runs/<run>/analysis.md (+ printed summary)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl

from . import config as C
from .decide import DecisionParams, select
from .metric import macro_f05
from .pipeline import _files, _parts, id_maps, labels, load, train_roles
from .prep import load_gt


def _val_ids():
    roles = train_roles().filter(pl.col("role") == "val").select(pl.col("idx").alias("s1_idx"), "country")
    return roles


def layer_recall(val: pl.DataFrame) -> pl.DataFrame:
    """Pair recall of true matches for val S1 after blocking (A), stage-1 pruning (P) and final selection."""
    vs = _val_ids()
    lab = labels().join(vs, on="s1_idx")
    a = pl.concat([pl.scan_parquet(_files(d)).select("s1_idx", "src", "t_idx")
                   .join(vs.lazy().select("s1_idx"), on="s1_idx", how="semi").collect(engine="streaming")
                   for d in _parts("train", "A")]).with_columns(pl.lit(1).alias("inA"))
    p = pl.concat([pl.read_parquet(f, columns=["s1_idx", "src", "t_idx"]) for f in _parts("train", "P")]
                  ).join(vs.select("s1_idx"), on="s1_idx", how="semi").with_columns(pl.lit(1).alias("inP"))
    s = val.select("s1_idx", "src", "t_idx").with_columns(pl.lit(1).alias("inS"))
    m = lab.join(a, on=["s1_idx", "src", "t_idx"], how="left").join(p, on=["s1_idx", "src", "t_idx"], how="left") \
           .join(s, on=["s1_idx", "src", "t_idx"], how="left")
    out = m.group_by("country").agg(pl.len().alias("true_pairs"),
                                    pl.col("inA").is_not_null().mean().alias("recall_blocking"),
                                    pl.col("inP").is_not_null().mean().alias("recall_stage1"),
                                    pl.col("inS").is_not_null().mean().alias("recall_scored"))
    vol = pl.DataFrame({"country": ["ALL"], "true_pairs": [m.height],
                        "recall_blocking": [m["inA"].is_not_null().mean()],
                        "recall_stage1": [m["inP"].is_not_null().mean()],
                        "recall_scored": [m["inS"].is_not_null().mean()]})
    cands = {"A_pairs_per_s1": a.height / vs.height, "P_pairs_per_s1": p.height / vs.height}
    return pl.concat([out.sort("country"), vol], how="vertical_relaxed"), cands


def calibration(val: pl.DataFrame, bins: int = 10) -> pl.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(val["p"].to_numpy(), edges) - 1, 0, bins - 1)
    return (val.with_columns(pl.Series("bin", b)).group_by("bin").agg(
                pl.len().alias("n"), pl.col("p").mean().alias("mean_p"), pl.col("y").mean().alias("frac_pos"))
              .sort("bin"))


def analyse(run: str) -> dict:
    rd = C.WORK / "runs" / run
    summ = json.loads((rd / "summary.json").read_text())
    best = summ["best"]
    val = pl.read_parquet(rd / "val_scores.parquet")
    sel = select(val, DecisionParams(margin=best["margin"], m0=best["m0"], empty_bias=best["empty_bias"]))
    s1_ids, tg_ids = id_maps("train")
    vs = _val_ids().join(s1_ids.select("s1_idx", "s1"), on="s1_idx")
    truth = load_gt().join(vs.select("s1"), on="s1")
    to_ids = lambda x: x.join(s1_ids.select("s1_idx", "s1"), on="s1_idx").join(tg_ids, on=["t_idx", "src"])
    pred = to_ids(sel).select("s1", "tid")
    rep = {"overall": macro_f05(pred, truth, vs["s1"])}
    # segments by number of true matches
    ntrue = truth.group_by("s1").len("nt")
    seg = vs.select("s1").join(ntrue, on="s1", how="left").with_columns(pl.col("nt").fill_null(0))
    rep["by_ntrue"] = {}
    for k, lo, hi in (("0", 0, 0), ("1", 1, 1), ("2-3", 2, 3), ("4-6", 4, 6), ("7+", 7, 99)):
        ids = seg.filter(pl.col("nt").is_between(lo, hi))["s1"]
        if ids.len():
            rep["by_ntrue"][k] = {**macro_f05(pred, truth, ids), "share": ids.len() / vs.height}
    lr, cands = layer_recall(val)
    rep["layer_recall"] = lr.to_dicts()
    rep["candidates"] = cands
    cal = calibration(val)
    rep["calibration"] = cal.to_dicts()
    rep["brier"] = float(((val["p"] - val["y"]) ** 2).mean())
    # error taxonomy
    fp = sel.join(val.select("s1_idx", "src", "t_idx", "y", "p"), on=["s1_idx", "src", "t_idx"]).filter(pl.col("y") == 0)
    lab = labels().join(vs.select("s1_idx"), on="s1_idx")
    fn = lab.join(sel, on=["s1_idx", "src", "t_idx"], how="anti")
    fn_scored = fn.join(val.select("s1_idx", "src", "t_idx", "p"), on=["s1_idx", "src", "t_idx"], how="left")
    rep["errors"] = {"false_pos": fp.height, "false_neg": fn.height,
                     "fn_not_in_candidates": int(fn_scored["p"].is_null().sum()),
                     "fn_scored_but_rejected": int(fn_scored["p"].is_not_null().sum()),
                     "fp_on_singletons": int(fp.join(seg.filter(pl.col("nt") == 0).join(s1_ids.select("s1", "s1_idx"), on="s1"),
                                                     on="s1_idx", how="semi").height)}
    # samples for manual inspection
    rec = {s: load("train", s, cols=["idx", "business_name", "business_address"]) for s in ("1", "2", "3")}
    def show(df: pl.DataFrame, n: int = 12) -> list[str]:
        d = df.sample(min(n, df.height), seed=0)
        d = d.join(rec["1"].rename({"idx": "s1_idx", "business_name": "n1", "business_address": "a1"}), on="s1_idx")
        tg = pl.concat([rec[s].with_columns(pl.lit(int(s), pl.UInt8).alias("src")) for s in ("2", "3")])
        d = d.join(tg.rename({"idx": "t_idx", "business_name": "n2", "business_address": "a2"}), on=["t_idx", "src"])
        return [f"{r['n1']} | {r['a1']}  <>  {r['n2']} | {r['a2']}  (p={r.get('p')})" for r in d.iter_rows(named=True)]
    rep["fp_examples"] = show(fp)
    rep["fn_rejected_examples"] = show(fn_scored.filter(pl.col("p").is_not_null()))
    rep["fn_blocking_examples"] = show(fn_scored.filter(pl.col("p").is_null()))
    (rd / "analysis.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v1")
    a = ap.parse_args()
    pl.Config.set_tbl_width_chars(200)
    r = analyse(a.run)
    print(json.dumps({k: v for k, v in r.items() if not k.endswith("examples")}, indent=1, default=str))
    for k in ("fp_examples", "fn_rejected_examples", "fn_blocking_examples"):
        print(f"\n== {k}")
        for e in r[k]:
            print("  ", e)
