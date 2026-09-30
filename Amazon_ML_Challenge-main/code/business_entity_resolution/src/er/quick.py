"""Stage-1-only submission: decision layer on the stage-1 OOF matcher score p1.

python -m er.quick --run v0s1
Uses the stage-1 candidate files (work/{split}_P_*.parquet), which already hold p1 for every
candidate. The decision layer is tuned on untouched validation S1 (p1 = mean of both fold
models there), then applied to test. Lightweight: no stage-B features needed.
"""
from __future__ import annotations

import argparse
import json
import logging

import polars as pl

from . import config as C
from .decide import DecisionParams, select, select_exact
from .output import write_submission
from .pipeline import _parts, id_maps, labels, run_tune, train_roles

log = logging.getLogger(__name__)


def run_quick(run: str = "v0s1") -> dict:
    rd = C.WORK / "runs" / run
    rd.mkdir(parents=True, exist_ok=True)
    val_ids = train_roles().filter(pl.col("role") == "val").select(pl.col("idx").alias("s1_idx"))
    lab = labels().with_columns(pl.lit(1, pl.UInt8).alias("y"))
    val = (pl.concat([pl.read_parquet(f, columns=["s1_idx", "src", "t_idx", "p1"]) for f in _parts("train", "P")])
             .join(val_ids, on="s1_idx", how="semi").rename({"p1": "p"})
             .join(lab, on=["s1_idx", "src", "t_idx"], how="left").with_columns(pl.col("y").fill_null(0)))
    val.write_parquet(rd / "val_scores.parquet")
    out = run_tune(run)                      # exact/ratio expected-F0.5 grid on validation
    best = out["best"]
    test = pl.concat([pl.read_parquet(f, columns=["s1_idx", "src", "t_idx", "p1"]) for f in _parts("test", "P")]
                     ).rename({"p1": "p"})
    fn = select_exact if best["kind"] == "exact" else select
    sel = fn(test, DecisionParams(margin=best["margin"], m0=best["m0"], empty_bias=best["empty_bias"]))
    s1_ids, tg_ids = id_maps("test")
    to_ids = lambda x: (x.join(s1_ids.select("s1_idx", "s1"), on="s1_idx")
                         .join(tg_ids, on=["t_idx", "src"]).select("s1", "tid"))
    write_submission(to_ids(sel), to_ids(test.select("s1_idx", "src", "t_idx")), s1_ids["s1"], C.OUT / run,
                     validator=C.ROOT / "student_resource" / "utils" / "validate_submission.py",
                     test_dir=C.DATA / "test")
    info = {"best": best, "by_country": out["by_country"], "test_pairs": test.height,
            "matches": sel.height, "s1_with_match": sel["s1_idx"].n_unique(), "test_s1": s1_ids.height}
    (rd / "quick.json").write_text(json.dumps(info, indent=1, default=str))
    log.info("quick submission: %s", info)
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v0s1")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run_quick(a.run)
