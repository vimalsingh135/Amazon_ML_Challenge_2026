"""Leave-one-country-out (LOCO) transfer test: proxy for the unseen test country (France).

python -m er.run loco --run v1
Trains the stage-2 matcher on fit S1 of one country and evaluates macro F0.5 on val S1 of the
other country (and in-domain for reference), for the full feature set and a shift-robust subset.
The decision layer uses the run's tuned parameters, so only the model's transfer is measured.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl

from . import config as C
from .decide import DecisionParams, select, select_exact
from .metric import macro_f05
from .pipeline import S2_PARAMS, _scan_b, id_maps, labels, train_roles
from .prep import load_gt

log = logging.getLogger(__name__)

# signals whose meaning depends on country-specific lexicons (absent/different for France)
COUNTRY_SPECIFIC = ("state_eq",)


def _data(feats, role):
    roles = train_roles().select(pl.col("idx").alias("s1_idx"), "role", "fold", "country")
    lab = labels().with_columns(pl.lit(1, pl.UInt8).alias("y"))
    return (_scan_b("train").join(roles.lazy().filter(pl.col("role") == role), on="s1_idx")
            .join(lab.lazy(), on=["s1_idx", "src", "t_idx"], how="left").with_columns(pl.col("y").fill_null(0))
            .select("s1_idx", "t_idx", "y", "country", pl.col("src").alias("src_key"),
                    *[pl.col(c).cast(pl.Float32) for c in feats])
            .collect(engine="streaming"))


def run_loco(run_name: str = "v1", max_fit: int = 3_000_000) -> dict:
    import lightgbm as lgb

    rd = C.WORK / "runs" / run_name
    feats = json.loads((rd / "summary.json").read_text())["features"]
    dec = json.loads((rd / "decision.json").read_text())["best"] if (rd / "decision.json").exists() else \
        json.loads((rd / "summary.json").read_text())["best"]
    fn = select_exact if dec.get("kind") == "exact" else select
    prm = DecisionParams(margin=dec["margin"], m0=dec["m0"], empty_bias=dec["empty_bias"])
    fit, val = _data(feats, "fit"), _data(feats, "val")
    s1_ids, tg_ids = id_maps("train")
    roles = train_roles().filter(pl.col("role") == "val").select(pl.col("idx").alias("s1_idx"), "country")
    gt = load_gt()
    res = {}
    for name, fs in (("all", feats), ("robust", [f for f in feats if f not in COUNTRY_SPECIFIC])):
        for tr_c in ("US", "India"):
            tr = fit.filter(pl.col("country") == tr_c)
            if tr.height > max_fit:
                tr = tr.sample(max_fit, seed=C.SEED)
            m = lgb.train({**S2_PARAMS, "num_threads": C.WORKERS}, lgb.Dataset(tr.select(fs).to_numpy(), tr["y"].to_numpy()),
                          800)
            for ev_c in ("US", "India"):
                v = val.filter(pl.col("country") == ev_c)
                v = v.select("s1_idx", pl.col("src_key").alias("src"), "t_idx").with_columns(
                    pl.Series("p", m.predict(v.select(fs).to_numpy(), num_threads=C.WORKERS).astype(np.float32)))
                ids = roles.filter(pl.col("country") == ev_c).join(s1_ids.select("s1_idx", "s1"), on="s1_idx")
                pred = fn(v, prm).join(ids, on="s1_idx").join(tg_ids, on=["t_idx", "src"]).select("s1", "tid")
                r = macro_f05(pred, gt.join(ids.select("s1"), on="s1"), ids["s1"])
                res[f"{name}:{tr_c}->{ev_c}"] = round(r["f05"], 5)
                log.info("loco %s %s->%s f05=%.5f", name, tr_c, ev_c, r["f05"])
    (rd / "loco.json").write_text(json.dumps(res, indent=1))
    return res
