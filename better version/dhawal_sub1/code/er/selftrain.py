"""Self-training (pseudo-label) domain adaptation, validated by simulation before use on France.

python -m er.selftrain simulate --run v3            (ER_WORK must point at the run's work dir)
Protocol (India plays the unseen country):
  1. base  : LightGBM trained on US fit pairs only -> India val macro F0.5 (zero-shot proxy)
  2. pseudo: score India FIT pairs with the base model as if unlabelled; confident positives
             (p >= hi and best candidate for its target) and confident negatives (p <= lo)
  3. adapt : retrain on US fit + India pseudo-labelled pairs -> India val macro F0.5
Accept the recipe only if adapt > base with a paired-bootstrap CI above zero.
"""
from __future__ import annotations

import argparse
import json
import logging

import lightgbm as lgb
import numpy as np
import polars as pl

from . import config as C
from .decide import DecisionParams, select
from .metric import macro_f05
from .pipeline import _scan_b, id_maps, labels, train_roles
from .prep import load_gt

log = logging.getLogger(__name__)
P = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.7,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=3.0, verbose=-1, seed=C.SEED,
         num_threads=int(__import__("os").environ.get("ER_WORKERS", 8)))


def _load(feats):
    roles = train_roles().select(pl.col("idx").alias("s1_idx"), "role", "country")
    lab = labels().with_columns(pl.lit(1, pl.UInt8).alias("y"))
    return (_scan_b("train").join(roles.lazy(), on="s1_idx")
            .join(lab.lazy(), on=["s1_idx", "src", "t_idx"], how="left").with_columns(pl.col("y").fill_null(0))
            .select("s1_idx", "t_idx", "y", "role", "country", pl.col("src").alias("src_key"),
                    *[pl.col(c).cast(pl.Float32) for c in feats])
            .collect(engine="streaming"))


def _per_s1_f(v: pl.DataFrame, prm: DecisionParams, ids: pl.DataFrame, tg_ids, gt) -> np.ndarray:
    sel = select(v, prm).join(ids, on="s1_idx").join(tg_ids, on=["t_idx", "src"]).select("s1", "tid").unique()
    truth = gt.join(ids.select("s1"), on="s1").unique()
    tp = truth.join(sel, on=["s1", "tid"]).group_by("s1").len("tp")
    df = (ids.select("s1").join(truth.group_by("s1").len("nt"), on="s1", how="left")
              .join(sel.group_by("s1").len("np"), on="s1", how="left").join(tp, on="s1", how="left").fill_null(0))
    p, r = pl.col("tp") / pl.col("np"), pl.col("tp") / pl.col("nt")
    return df.select(pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0)
                     .when((pl.col("nt") == 0) | (pl.col("np") == 0) | (pl.col("tp") == 0)).then(0.0)
                     .otherwise(1.25 * p * r / (0.25 * p + r))).to_series().to_numpy()


def simulate(run: str, hi: float = 0.97, lo: float = 0.02, w_pseudo: float = 1.0, n_boot: int = 1000) -> dict:
    rd = C.WORK / "runs" / run
    feats = json.loads((rd / "summary.json").read_text())["features"]
    dec = json.loads((rd / "decision.json").read_text())["best"]
    prm = DecisionParams(margin=dec["margin"], m0=dec["m0"], empty_bias=dec["empty_bias"])
    d = _load(feats)
    us_fit = d.filter((pl.col("country") == "US") & (pl.col("role") == "fit"))
    in_fit = d.filter((pl.col("country") == "India") & (pl.col("role") == "fit"))
    in_val = d.filter((pl.col("country") == "India") & (pl.col("role") == "val"))
    del d
    X = lambda df: df.select(feats).to_numpy()
    base = lgb.train(P, lgb.Dataset(X(us_fit), us_fit["y"].to_numpy()), 600)
    # pseudo-labels on India fit (labels NOT used), target-exclusive confident positives
    pf = in_fit.select("s1_idx", "t_idx", pl.col("src_key").alias("src")).with_columns(
        pl.Series("p", base.predict(X(in_fit))))
    pf = pf.with_columns((pl.col("p") >= pl.col("p").max().over("src", "t_idx")).alias("best"))
    pos = (pf["p"] >= hi) & pf["best"]
    neg = pf["p"] <= lo
    keep = (pos | neg).to_numpy()
    y_pseudo = pos.to_numpy()[keep].astype(int)
    true_y = in_fit["y"].to_numpy()[keep]
    pl_acc = {"n_pseudo": int(keep.sum()), "pos": int(y_pseudo.sum()),
              "pseudo_label_accuracy": float((y_pseudo == true_y).mean())}
    Xa = np.vstack([X(us_fit), X(in_fit)[keep]])
    ya = np.concatenate([us_fit["y"].to_numpy(), y_pseudo])
    wa = np.concatenate([np.ones(us_fit.height), np.full(int(keep.sum()), w_pseudo)])
    adapt = lgb.train(P, lgb.Dataset(Xa, ya, weight=wa), 600)
    s1_ids, tg_ids = id_maps("train")
    ids = (train_roles().filter((pl.col("role") == "val") & (pl.col("country") == "India"))
           .select(pl.col("idx").alias("s1_idx")).join(s1_ids.select("s1_idx", "s1"), on="s1_idx"))
    gt = load_gt()
    res = {"pseudo": pl_acc}
    fs = {}
    for name, m in (("base_us_only", base), ("adapted", adapt)):
        v = in_val.select("s1_idx", pl.col("src_key").alias("src"), "t_idx").with_columns(
            pl.Series("p", m.predict(X(in_val)).astype(np.float32)))
        fs[name] = _per_s1_f(v, prm, ids, tg_ids, gt)
        res[name] = round(float(fs[name].mean()), 5)
    dlt = fs["adapted"] - fs["base_us_only"]
    rng = np.random.default_rng(0)
    boots = dlt[rng.integers(0, len(dlt), size=(n_boot, len(dlt)))].mean(1)
    res["delta"] = round(float(dlt.mean()), 5)
    res["ci95"] = [round(float(np.percentile(boots, 2.5)), 5), round(float(np.percentile(boots, 97.5)), 5)]
    res["ACCEPT"] = res["ci95"][0] > 0
    (rd / "selftrain_sim.json").write_text(json.dumps(res, indent=1))
    log.info("self-training simulation: %s", res)
    return res



def make_pseudo(run: str, country: str, hi: float = 0.97, lo: float = 0.02, out: str | None = None) -> dict:
    """Confident pseudo-labels for one test country from a run's calibrated test scores:
    positives p >= hi that are the best-scoring S1 for their target; negatives p <= lo."""
    from .pipeline import load
    rd = C.WORK / "runs" / run
    t = pl.read_parquet(rd / "test_scores.parquet")
    c = load("test", "1", cols=["idx", "country"]).rename({"idx": "s1_idx"}).filter(pl.col("country") == country)
    t = t.join(c.select("s1_idx"), on="s1_idx", how="semi").with_columns(
        (pl.col("p") >= pl.col("p").max().over("src", "t_idx")).alias("best"))
    ps = t.filter(((pl.col("p") >= hi) & pl.col("best")) | (pl.col("p") <= lo)).select(
        "s1_idx", "src", "t_idx", (pl.col("p") >= hi).cast(pl.Int8).alias("y_pseudo"))
    path = out or str(rd / f"pseudo_{country}.parquet")
    ps.write_parquet(path)
    info = {"country": country, "rows": ps.height, "pos": int(ps["y_pseudo"].sum()), "path": path}
    log.info("pseudo-labels: %s", info)
    return info

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["simulate", "pseudo"])
    ap.add_argument("--run", default="v3")
    ap.add_argument("--country", default="France")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    print(json.dumps(simulate(a.run) if a.what == "simulate" else make_pseudo(a.run, a.country), indent=1))
