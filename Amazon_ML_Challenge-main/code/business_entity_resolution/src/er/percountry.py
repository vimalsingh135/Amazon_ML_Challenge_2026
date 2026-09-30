"""Per-country decision experiment (evaluated: 0.000 gain vs global; not used).

v2's stage-2 scores re-decided with per-country tuned decision parameters.
python -m er.percountry
Reads work/runs/v0/test_scores.parquet (v2's calibrated stage-2 probabilities) and
work/runs/v0/decision_per_country.json (US / India parameters tuned on validation), applies the
expected-F0.5 selector per country (France: v2's validated global parameters) and writes
output/percountry/. Independent of v3 (separate work dir and outputs).
"""
from __future__ import annotations

import json
import logging

import polars as pl

from . import config as C
from .decide import DecisionParams, select
from .output import write_submission
from .pipeline import id_maps, load

log = logging.getLogger(__name__)


def run_percountry() -> None:
    rd = C.WORK / "runs" / "v0"
    glob = json.loads((rd / "decision_refined.json").read_text())["best"]
    per = json.loads((rd / "decision_per_country.json").read_text())
    test = pl.read_parquet(rd / "test_scores.parquet")
    ctry = load("test", "1", cols=["idx", "country"]).rename({"idx": "s1_idx"})
    test = test.join(ctry, on="s1_idx")
    sels = []
    for (c,), d in test.group_by("country"):
        prm = per.get(c, {}).get("params") if per.get(c, {}).get("gain", 0) > 0 else None
        prm = prm or {"m0": glob["m0"], "empty_bias": glob["empty_bias"], "margin": glob["margin"]}
        log.info("percountry %s: %s", c, prm)
        sels.append(select(d.drop("country"), DecisionParams(**prm)))
    sel = pl.concat(sels)
    s1_ids, tg_ids = id_maps("test")
    to_ids = lambda x: (x.join(s1_ids.select("s1_idx", "s1"), on="s1_idx")
                         .join(tg_ids, on=["t_idx", "src"]).select("s1", "tid"))
    write_submission(to_ids(sel), to_ids(test.select("s1_idx", "src", "t_idx")), s1_ids["s1"], C.OUT / "percountry",
                     validator=C.ROOT / "student_resource" / "utils" / "validate_submission.py",
                     test_dir=C.DATA / "test")
    log.info("percountry: %d matches, %d S1 with >=1 match", sel.height, sel["s1_idx"].n_unique())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run_percountry()
