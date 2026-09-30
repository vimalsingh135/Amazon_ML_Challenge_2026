"""Acceptance gate: paired comparison of two runs on the SAME validation S1 entities.

python -m er.compare --a work/runs/v0 --b work_v3/runs/v3
For each run: apply its tuned decision to its validation scores, compute per-S1 F0.5 (exact
challenge rules), then a paired bootstrap over S1 (B resamples) of the difference b - a,
overall and per country, plus precision / recall / singleton-F deltas. The candidate (b) is
accepted only if the 95% CI of dF0.5 is entirely > 0 overall AND in every country.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from .decide import DecisionParams, select, select_exact
from .prep import load_gt

BETA2 = 0.25


def _decision(rd: Path) -> dict:
    for name in ("decision_refined.json", "decision.json"):
        if (rd / name).exists():
            return json.loads((rd / name).read_text())["best"]
    return json.loads((rd / "summary.json").read_text())["best"]


def per_s1(rd: Path, work: Path) -> pl.DataFrame:
    """Per-S1 (s1, country, f, tp, np, nt) for one run on its validation S1 set."""
    val = pl.read_parquet(rd / "val_scores.parquet")
    b = _decision(rd)
    fn = select_exact if b.get("kind") == "exact" else select
    sel = fn(val, DecisionParams(margin=b["margin"], m0=b["m0"], empty_bias=b["empty_bias"], coh=b.get("coh", 0.0)))
    roles = pl.read_parquet(work / "train_roles.parquet").filter(pl.col("role") == "val")
    s1 = pl.read_parquet(work / "train_s1.parquet", columns=["idx", "entity_id"]).rename(
        {"idx": "s1_idx", "entity_id": "s1"})
    tg = pl.concat([pl.read_parquet(work / f"train_s{s}.parquet", columns=["idx", "entity_id"])
                      .rename({"idx": "t_idx", "entity_id": "tid"}).with_columns(pl.lit(int(s), pl.UInt8).alias("src"))
                    for s in ("2", "3")])
    base = roles.select(pl.col("idx").alias("s1_idx"), "country").join(s1, on="s1_idx")
    pred = sel.join(tg, on=["t_idx", "src"]).join(s1, on="s1_idx").select("s1", "tid").unique()
    truth = load_gt().join(base.select("s1"), on="s1").unique()
    tp = truth.join(pred, on=["s1", "tid"]).group_by("s1").len("tp")
    df = (base.join(truth.group_by("s1").len("nt"), on="s1", how="left")
              .join(pred.group_by("s1").len("np"), on="s1", how="left")
              .join(tp, on="s1", how="left").fill_null(0))
    p = pl.col("tp") / pl.col("np")
    r = pl.col("tp") / pl.col("nt")
    return df.with_columns(
        pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0)
          .when((pl.col("nt") == 0) | (pl.col("np") == 0) | (pl.col("tp") == 0)).then(0.0)
          .otherwise((1 + BETA2) * p * r / (BETA2 * p + r)).alias("f"))


def compare(a: Path, b: Path, work_a: Path, work_b: Path, n_boot: int = 2000, seed: int = 0) -> dict:
    A, B = per_s1(a, work_a), per_s1(b, work_b)
    m = A.select("s1", "country", pl.col("f").alias("fa"), pl.col("tp").alias("tpa"), pl.col("np").alias("npa"),
                 "nt").join(B.select("s1", pl.col("f").alias("fb"), pl.col("tp").alias("tpb"),
                                     pl.col("np").alias("npb")), on="s1")
    rng = np.random.default_rng(seed)
    out = {"n_common_s1": m.height}
    for name, g in [("ALL", m), *[(c, d) for (c,), d in m.group_by("country")]]:
        d = (g["fb"] - g["fa"]).to_numpy()
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        boots = d[idx].mean(axis=1)
        sing = g.filter(pl.col("nt") == 0)
        out[name] = {
            "f05_a": round(float(g["fa"].mean()), 5), "f05_b": round(float(g["fb"].mean()), 5),
            "delta": round(float(d.mean()), 5),
            "ci95": [round(float(np.percentile(boots, 2.5)), 5), round(float(np.percentile(boots, 97.5)), 5)],
            "p_b_better": round(float((boots > 0).mean()), 4),
            "precision_a": round(float((g["tpa"] / g["npa"]).drop_nans().mean()), 5),
            "precision_b": round(float((g["tpb"] / g["npb"]).drop_nans().mean()), 5),
            "recall_a": round(float((g.filter(pl.col("nt") > 0)["tpa"] / g.filter(pl.col("nt") > 0)["nt"]).mean()), 5),
            "recall_b": round(float((g.filter(pl.col("nt") > 0)["tpb"] / g.filter(pl.col("nt") > 0)["nt"]).mean()), 5),
            "singleton_f_a": round(float(sing["fa"].mean()), 5) if sing.height else None,
            "singleton_f_b": round(float(sing["fb"].mean()), 5) if sing.height else None,
        }
    out["ACCEPT"] = all(out[k]["ci95"][0] > 0 for k in out if isinstance(out[k], dict))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--work-a", required=True)
    ap.add_argument("--work-b", required=True)
    x = ap.parse_args()
    print(json.dumps(compare(Path(x.a), Path(x.b), Path(x.work_a), Path(x.work_b)), indent=1))
