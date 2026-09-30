"""Stage-2 ensemble on top of a trained run: LightGBM lambdarank + CatBoost + logistic stacker.

python -m er.run ensemble --run v1
Reuses run v1's stage-B features and fold split. Every base model is trained 2-fold out-of-fold
on fit S1. The stacker is fit on OOF predictions only (no leakage), calibrated per source, and
scored on the untouched val S1. Writes val_scores_ens.parquet / test_scores_ens.parquet; `tune`
and `predict` can use them with --scores ens.
"""
from __future__ import annotations

import gc
import json
import logging
import pickle

import numpy as np
import polars as pl

from . import config as C
from .pipeline import META, _files, _parts, _scan_b, feature_cols, labels, train_roles

log = logging.getLogger(__name__)

RANK_PARAMS = dict(objective="lambdarank", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
                   feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=3.0,
                   lambdarank_truncation_level=10, eval_at=[5], num_threads=C.WORKERS, verbose=-1, seed=C.SEED)
CAT_PARAMS = dict(loss_function="Logloss", iterations=3000, learning_rate=0.08, depth=8, l2_leaf_reg=5,
                  border_count=128, od_type="Iter", od_wait=100, thread_count=C.WORKERS, random_seed=C.SEED,
                  verbose=250, allow_writing_files=False)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _group_softmax(keys: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Softmax of ranking scores within each (S1, source) group -> comparable across groups."""
    df = pl.DataFrame({"g": keys, "s": s}).with_columns(
        (pl.col("s") - pl.col("s").max().over("g")).exp().alias("e"))
    return (df["e"] / df.select(pl.col("e").sum().over("g"))["e"]).to_numpy()


def _load_fit(feats):
    roles = train_roles().select(pl.col("idx").alias("s1_idx"), "role", "fold")
    lab = labels().with_columns(pl.lit(1, pl.UInt8).alias("y"))
    fit = (_scan_b("train").join(roles.lazy().filter(pl.col("role") == "fit"), on="s1_idx")
           .join(lab.lazy(), on=["s1_idx", "src", "t_idx"], how="left").with_columns(pl.col("y").fill_null(0))
           .select("s1_idx", "t_idx", "y", "fold", *[pl.col(c).cast(pl.Float32) for c in feats])
           .sort("s1_idx", "src").collect(engine="streaming"))
    return fit


def _base_predict(X, keys, models) -> dict[str, np.ndarray]:
    import lightgbm as lgb  # noqa: F401
    out = {}
    out["lgb"] = np.mean([m.predict(X, num_threads=C.WORKERS) for m in models["lgb"]], axis=0)
    raw = np.mean([m.predict(X, num_threads=C.WORKERS) for m in models["rank"]], axis=0)
    out["rank"], out["rank_raw"] = _group_softmax(keys, raw), raw
    out["cat"] = np.mean([m.predict_proba(X)[:, 1] for m in models["cat"]], axis=0)
    return out


def _stack_X(base: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([_logit(base["lgb"]), _logit(base["rank"]), base["rank_raw"], _logit(base["cat"])])


def run_ensemble(run_name: str = "v1") -> dict:
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    rd = C.WORK / "runs" / run_name
    summ = json.loads((rd / "summary.json").read_text())
    feats = summ["features"]
    fit = _load_fit(feats)
    y, fold = fit["y"].to_numpy(), fit["fold"].to_numpy()
    keys = (fit["s1_idx"].cast(pl.Int64) * 4 + fit["src"].cast(pl.Int64)).to_numpy()
    X = fit.select(feats).to_numpy()
    src = fit["src"].to_numpy()
    del fit
    gc.collect()
    models = {"lgb": [lgb.Booster(model_file=str(rd / f"stage2_f{f}.lgb")) for f in (0, 1)], "rank": [], "cat": []}
    oof = {k: np.zeros(len(y), dtype=np.float64) for k in ("lgb", "rank", "rank_raw", "cat")}
    for f in (0, 1):
        tr, va = fold != f, fold == f
        oof["lgb"][va] = models["lgb"][f].predict(X[va], num_threads=C.WORKERS)
        # lambdarank: rows are sorted by (s1_idx, src) so groups are contiguous
        _, gtr = np.unique(keys[tr], return_counts=True)
        _, gva = np.unique(keys[va], return_counts=True)
        dtr = lgb.Dataset(X[tr], y[tr], group=gtr, feature_name=feats)
        dva = lgb.Dataset(X[va], y[va], group=gva, reference=dtr)
        mr = lgb.train(RANK_PARAMS, dtr, 2000, valid_sets=[dva],
                       callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(250)])
        mr.save_model(str(rd / f"rank_f{f}.lgb"))
        models["rank"].append(mr)
        oof["rank_raw"][va] = mr.predict(X[va], num_threads=C.WORKERS)
        oof["rank"][va] = _group_softmax(keys[va], oof["rank_raw"][va])
        del dtr, dva
        mc = CatBoostClassifier(**CAT_PARAMS)
        mc.fit(X[tr], y[tr], eval_set=(X[va], y[va]))
        mc.save_model(str(rd / f"cat_f{f}.cbm"))
        models["cat"].append(mc)
        oof["cat"][va] = mc.predict_proba(X[va])[:, 1]
        gc.collect()
    S = _stack_X(oof)
    stacker = LogisticRegression(C=1.0, max_iter=1000).fit(S, y)
    ps = stacker.predict_proba(S)[:, 1]
    cal = {s: IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(ps[src == s], y[src == s])
           for s in (2, 3)}
    from sklearn.metrics import log_loss, roc_auc_score
    diag = {k: {"auc": float(roc_auc_score(y, v))} for k, v in {**oof, "stack": ps}.items()}
    diag["stacker_coef"] = stacker.coef_.tolist()
    log.info("ensemble OOF diagnostics %s", diag)
    del X
    gc.collect()
    with open(rd / "ensemble.pkl", "wb") as fh:
        pickle.dump({"stacker": stacker, "cal": cal, "feats": feats}, fh)
    for split, role, name in (("train", "val", "val_scores_ens.parquet"), ("test", None, "test_scores_ens.parquet")):
        _score_split(split, role, models, stacker, cal, feats).write_parquet(rd / name)
    (rd / "ensemble.json").write_text(json.dumps(diag, indent=1))
    return diag


def _score_split(split, role, models, stacker, cal, feats) -> pl.DataFrame:
    roles = train_roles().select(pl.col("idx").alias("s1_idx"), "role") if role else None
    lab = labels().with_columns(pl.lit(1, pl.UInt8).alias("y")) if split == "train" else None
    out = []
    for d_ in _parts(split, "B"):
        for f in _files(d_):
            d = pl.read_parquet(f)
            if roles is not None:
                d = d.join(roles.filter(pl.col("role") == role).select("s1_idx"), on="s1_idx", how="semi")
            if d.height == 0:
                continue
            d = d.sort("s1_idx", "src")
            X = d.select([pl.col(c).cast(pl.Float32) if c in d.columns else pl.lit(None, pl.Float32).alias(c)
                          for c in feats]).to_numpy()
            keys = (d["s1_idx"].cast(pl.Int64) * 4 + d["src"].cast(pl.Int64)).to_numpy()
            ps = stacker.predict_proba(_stack_X(_base_predict(X, keys, models)))[:, 1]
            src = d["src"].to_numpy()
            p = ps.copy()
            for s, c in cal.items():
                if (src == s).any():
                    p[src == s] = c.predict(ps[src == s])
            r = d.select("s1_idx", "src", "t_idx").with_columns(pl.Series("p", p.astype(np.float32)))
            out.append(r)
    res = pl.concat(out)
    if lab is not None:
        res = res.join(lab, on=["s1_idx", "src", "t_idx"], how="left").with_columns(pl.col("y").fill_null(0))
    return res
