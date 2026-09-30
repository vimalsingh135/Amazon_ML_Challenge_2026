"""Cross-encoder data exchange (GPU route): export raw-text pairs, import transformer scores.

A multilingual transformer (mdeberta-v3-base, MIT, 278M) is fine-tuned on raw "name | address"
pairs on a Kaggle GPU (scripts/kaggle/ce_kernel.py) and scores the uncertain band of a run.
Leakage-safe protocol:
  * transformer training : FIT S1 of fold 0 only (labels from train ground truth)
  * stacker training     : FIT S1 of fold 1 (stage-2 OOF p + transformer score), er.cross stack
  * gate                 : validation S1, never seen by either model

python -m er.cross export-train --run v4fr --out <dir>          (ER_WORK=<work dir>)
python -m er.cross export-band  --run v6 --out <dir> [--lo 0.005 --hi 0.995]
"""
from __future__ import annotations

import argparse
import logging

import polars as pl

from . import config as C

log = logging.getLogger(__name__)


def _texts(split: str) -> dict[str, pl.DataFrame]:
    out = {}
    for s in ("1", "2", "3"):
        d = pl.read_parquet(C.WORK / f"{split}_s{s}.parquet", columns=["idx", "business_name", "business_address"])
        out[s] = d.select(pl.col("idx").cast(pl.UInt32),
                          (pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null("")).alias("t"))
    return out


def _attach(pairs: pl.DataFrame, split: str) -> pl.DataFrame:
    tx = _texts(split)
    p = pairs.with_columns(pl.col("s1_idx").cast(pl.UInt32), pl.col("src").cast(pl.UInt8), pl.col("t_idx").cast(pl.UInt32))
    a = p.join(tx["1"].rename({"idx": "s1_idx", "t": "a"}), on="s1_idx")
    parts = [a.filter(pl.col("src") == int(s)).join(tx[s].rename({"idx": "t_idx", "t": "b"}), on="t_idx") for s in ("2", "3")]
    return pl.concat(parts)


def export_train(run: str, out: str, neg_easy_frac: float = 0.05, seed: int = 0) -> dict:
    from .pipeline import train_roles
    fold0 = train_roles().filter((pl.col("role") == "fit") & (pl.col("fold") == 0)).select(pl.col("idx").cast(pl.UInt32).alias("s1_idx"))
    f = pl.read_parquet(C.WORK / "runs" / run / "fit_oof.parquet").with_columns(pl.col("s1_idx").cast(pl.UInt32))
    f = f.join(fold0, on="s1_idx", how="semi")
    h = pl.col("t_idx").hash(seed) % 1000
    # every uncertain / hard pair, a quarter of the easy positives (p >= 0.99), 5% of the easy negatives
    keep = (((pl.col("y") == 1) & ((pl.col("oof") < 0.99) | (h < 250)))
            | ((pl.col("y") == 0) & ((pl.col("oof") >= 0.01) | (h < int(neg_easy_frac * 1000)))))
    ctry = pl.read_parquet(C.WORK / "train_s1.parquet", columns=["idx", "country"]).select(pl.col("idx").cast(pl.UInt32).alias("s1_idx"), "country")
    d = _attach(f.filter(keep).select("s1_idx", "src", "t_idx", pl.col("y").cast(pl.Int8).alias("label"), "oof"), "train")
    d = d.join(ctry, on="s1_idx")      # lets a kernel train on one country only (transfer / LOCO test)
    d = d.sample(fraction=1.0, shuffle=True, seed=seed)
    path = f"{out}/ce_train.parquet"
    d.write_parquet(path)
    info = {"rows": d.height, "pos": int(d["label"].sum()), "path": path}
    log.info("export-train %s", info)
    return info


def _fit1_calibrated(rd) -> pl.DataFrame:
    """FIT fold-1 pairs with the run's per-source isotonic calibration applied to the raw OOF score,
    so they are on the same scale as val/test scores (which are calibrated at prediction time)."""
    import pickle

    import numpy as np
    from .pipeline import train_roles
    fold1 = train_roles().filter((pl.col("role") == "fit") & (pl.col("fold") == 1)).select(pl.col("idx").cast(pl.UInt32).alias("s1_idx"))
    f = (pl.read_parquet(rd / "fit_oof.parquet").with_columns(pl.col("s1_idx").cast(pl.UInt32))
           .join(fold1, on="s1_idx", how="semi"))
    cal = pickle.load(open(rd / "calibrators.pkl", "rb"))
    raw, src = f["oof"].to_numpy(), f["src"].to_numpy()
    p = raw.copy()
    for k, c in cal.items():
        m = src == k
        if m.any():
            p[m] = c.predict(raw[m])
    return f.with_columns(pl.Series("p", p.astype(np.float32))).select("s1_idx", "src", "t_idx", "y", "p")


def export_band(run: str, out: str, lo: float = 0.005, hi: float = 0.995) -> dict:
    """Pairs whose stage-2 probability is uncertain: fit fold-1 (stacker), val (gate), test (submission)."""
    rd = C.WORK / "runs" / run
    band = (pl.col("p") >= lo) & (pl.col("p") <= hi)
    fit1 = _fit1_calibrated(rd)
    val = pl.read_parquet(rd / "val_scores.parquet").select("s1_idx", "src", "t_idx", "p")
    test = pl.read_parquet(rd / "test_scores.parquet").select("s1_idx", "src", "t_idx", "p")
    info = {}
    for name, d, split in (("fit1", fit1, "train"), ("val", val, "train"), ("test", test, "test")):
        d = _attach(d.filter(band).select("s1_idx", "src", "t_idx", "p"), split)
        d.write_parquet(f"{out}/ce_band_{name}.parquet")
        info[name] = d.height
    log.info("export-band %s", info)
    return info


def export_biencoder(out: str, max_pairs: int = 1_200_000, seed: int = 0) -> dict:
    """Positive (S1 text, target text) pairs of FIT S1 (never val) for a contrastive bi-encoder
    (in-batch negatives). Used for dense candidate retrieval (blocking recall)."""
    from .pipeline import labels, train_roles
    fit = train_roles().filter(pl.col("role") == "fit").select(pl.col("idx").cast(pl.UInt32).alias("s1_idx"), "country")
    lab = labels().select(pl.col("s1_idx").cast(pl.UInt32), pl.col("src").cast(pl.UInt8), pl.col("t_idx").cast(pl.UInt32)).join(fit, on="s1_idx")
    if lab.height > max_pairs:
        lab = lab.sample(max_pairs, seed=seed)
    d = _attach(lab, "train").sample(fraction=1.0, shuffle=True, seed=seed)
    d.select("s1_idx", "src", "t_idx", "country", "a", "b").write_parquet(f"{out}/bi_train.parquet")
    info = {"pairs": d.height, "by_country": dict(d.group_by("country").len().rows())}
    log.info("export-biencoder %s", info)
    return info


def export_bi_eval(out: str, country: str = "India", decomp_run: str = "v5") -> dict:
    """Retrieval evaluation set: val S1 texts of one country, all its train target texts, and its true val
    pairs flagged by whether key-based blocking found them (runs/<decomp_run>/val_recall_decomp.parquet)."""
    from .pipeline import train_roles
    val = train_roles().filter((pl.col("role") == "val") & (pl.col("country") == country)).select(pl.col("idx").cast(pl.UInt32))
    tx = _texts("train")
    tx["1"].join(val, on="idx").rename({"t": "text"}).write_parquet(f"{out}/bi_eval_s1.parquet")
    parts = []
    for s_ in ("2", "3"):
        c = pl.read_parquet(C.WORK / f"train_s{s_}.parquet", columns=["idx", "country"]).filter(pl.col("country") == country)
        parts.append(tx[s_].join(c.select(pl.col("idx").cast(pl.UInt32)), on="idx")
                     .select(pl.lit(int(s_)).cast(pl.UInt8).alias("src"), "idx", pl.col("t").alias("text")))
    tg = pl.concat(parts)
    tg.write_parquet(f"{out}/bi_eval_tg.parquet")
    tr = (pl.read_parquet(C.WORK / "runs" / decomp_run / "val_recall_decomp.parquet")
            .filter(pl.col("country") == country).select("s1_idx", "src", "t_idx", "inA"))
    tr.write_parquet(f"{out}/bi_eval_truth.parquet")
    info = {"s1": val.height, "targets": tg.height, "true_pairs": tr.height, "blocking_missed": int((~tr["inA"]).sum())}
    log.info("export-bi-eval %s", info)
    return info


def export_pseudo(scores: str, split: str, out: str, hi: float = 0.97, lo: float = 0.02, neg_per_pos: float = 1.0,
                  country: str | None = None, seed: int = 0) -> dict:
    """Label-free transformer adaptation data for a target country: confident positives (p >= hi and the best
    S1 for their target) and confident negatives (p <= lo, sampled neg_per_pos per positive), with raw text."""
    d = pl.read_parquet(scores).select("s1_idx", "src", "t_idx", "p").with_columns(pl.col("s1_idx").cast(pl.UInt32), pl.col("src").cast(pl.UInt8), pl.col("t_idx").cast(pl.UInt32))
    if country:
        c = pl.read_parquet(C.WORK / f"{split}_s1.parquet", columns=["idx", "country"]).filter(pl.col("country") == country)
        d = d.join(c.select(pl.col("idx").cast(pl.UInt32).alias("s1_idx")), on="s1_idx", how="semi")
    d = d.with_columns((pl.col("p") >= pl.col("p").max().over("src", "t_idx")).alias("best"))
    pos = d.filter((pl.col("p") >= hi) & pl.col("best")).with_columns(pl.lit(1, pl.Int8).alias("label"))
    neg = d.filter(pl.col("p") <= lo)
    neg = neg.sample(min(neg.height, int(pos.height * neg_per_pos)), seed=seed).with_columns(pl.lit(0, pl.Int8).alias("label"))
    x = _attach(pl.concat([pos, neg]).select("s1_idx", "src", "t_idx", "label"), split).sample(fraction=1.0, shuffle=True, seed=seed)
    x.write_parquet(f"{out}/ce_pseudo.parquet")
    info = {"pos": pos.height, "neg": neg.height}
    log.info("export-pseudo %s", info)
    return info


def export_country_adapt(run: str, country: str, out: str, n_pseudo: int = 150_000, hi: float = 0.97, lo: float = 0.02,
                         blo: float = 0.005, bhi: float = 0.995, seed: int = 0) -> dict:
    """Everything the GPU needs to adapt the cross-encoder to an unseen test country (validated on India-as-unseen:
    +0.0094 [0.0087, 0.0101] F0.5): label-free pseudo pairs of the country (confident run decisions), the country's
    uncertain test band, and the run's labelled FIT fold-1 band (stacker training). Also keeps the country's full
    test scores and the fold-1 scores (p, y) locally for stacking and the final decision."""
    rd = C.WORK / "runs" / run
    cid = (pl.read_parquet(C.WORK / "test_s1.parquet", columns=["idx", "country"]).filter(pl.col("country") == country)
             .select(pl.col("idx").cast(pl.UInt32).alias("s1_idx")))
    t = (pl.scan_parquet(rd / "test_scores.parquet").select(*KEYS, "p")
           .with_columns(pl.col("s1_idx").cast(pl.UInt32), pl.col("src").cast(pl.UInt8), pl.col("t_idx").cast(pl.UInt32))
           .join(cid.lazy(), on="s1_idx", how="semi").collect())
    t.write_parquet(f"{out}/country_test_scores.parquet")
    t2 = t.with_columns((pl.col("p") >= pl.col("p").max().over("src", "t_idx")).alias("best"))
    pos = t2.filter((pl.col("p") >= hi) & pl.col("best"))
    pos = pos.sample(min(pos.height, n_pseudo), seed=seed).with_columns(pl.lit(1, pl.Int8).alias("label"))
    neg = t2.filter(pl.col("p") <= lo)
    neg = neg.sample(min(neg.height, n_pseudo), seed=seed).with_columns(pl.lit(0, pl.Int8).alias("label"))
    _attach(pl.concat([pos, neg]).select(*KEYS, "label"), "test").sample(fraction=1.0, shuffle=True, seed=seed).write_parquet(f"{out}/ce_pseudo.parquet")
    band = t.filter((pl.col("p") >= blo) & (pl.col("p") <= bhi))
    _attach(band.select(*KEYS, "p"), "test").write_parquet(f"{out}/ce_band_test.parquet")
    f1 = _fit1_calibrated(rd).with_columns(pl.col("s1_idx").cast(pl.UInt32), pl.col("src").cast(pl.UInt8), pl.col("t_idx").cast(pl.UInt32))
    f1.write_parquet(f"{out}/fit1_scores.parquet")
    _attach(f1.filter((pl.col("p") >= blo) & (pl.col("p") <= bhi)).select(*KEYS, "p"), "train").write_parquet(f"{out}/ce_band_fit1.parquet")
    info = {"country_pairs": t.height, "pseudo_pos": pos.height, "pseudo_neg": neg.height, "test_band": band.height,
            "fit1_pairs": f1.height, "fit1_band": int(((f1["p"] >= blo) & (f1["p"] <= bhi)).sum())}
    log.info("export-country-adapt %s", info)
    return info


KEYS = ["s1_idx", "src", "t_idx"]
STACK_FEATS = ["lp", "lce", "src", "ce_rank_s1", "ce_gap_t", "p_rank_s1", "dce"]


def _stack_feats(d: pl.DataFrame) -> pl.DataFrame:
    lg = lambda c: (pl.col(c).clip(1e-6, 1 - 1e-6) / (1 - pl.col(c).clip(1e-6, 1 - 1e-6))).log()
    return d.with_columns(lg("p").alias("lp"), lg("ce").alias("lce"),
                          pl.col("ce").rank("ordinal", descending=True).over("s1_idx", "src").cast(pl.Float32).alias("ce_rank_s1"),
                          pl.col("p").rank("ordinal", descending=True).over("s1_idx", "src").cast(pl.Float32).alias("p_rank_s1"),
                          # competition on the transformer score: lead over the best other S1 for the same target
                          # winner: lead over the runner-up; loser: (negative) deficit to the winner; sole candidate: ce
                          pl.when(pl.len().over("src", "t_idx") == 1).then(pl.col("ce"))
                            .when(pl.col("ce") >= pl.col("ce").max().over("src", "t_idx"))
                            .then(pl.col("ce") - pl.col("ce").top_k(2).min().over("src", "t_idx"))
                            .otherwise(pl.col("ce") - pl.col("ce").max().over("src", "t_idx")).alias("ce_gap_t"),
                          (pl.col("ce") - pl.col("p")).alias("dce"))


def stack(run: str, ce_dir: str, out_run: str, tag: str = "ce") -> dict:
    """Train a small LightGBM stacker on FIT fold-1 band pairs (stage-2 calibrated OOF p + transformer
    score), apply it to the val/test band; pairs outside the band keep their stage-2 probability.
    Writes runs/<out_run>/{val_scores.parquet, val_scores_<tag>.parquet, test_scores_<tag>.parquet} and
    links the run's models/calibrators/summary so tune / predict / compare work unchanged."""
    import os

    import lightgbm as lgb
    from .pipeline import labels
    rd, od = C.WORK / "runs" / run, C.WORK / "runs" / out_run
    od.mkdir(parents=True, exist_ok=True)
    for f in ("summary.json", "calibrators.pkl", "stage2_f0.lgb", "stage2_f1.lgb"):
        if not (od / f).exists():
            os.link(rd / f, od / f)
    cast = lambda d: d.with_columns(pl.col("s1_idx").cast(pl.UInt32), pl.col("src").cast(pl.UInt8), pl.col("t_idx").cast(pl.UInt32))
    ce = {n: cast(pl.read_parquet(f"{ce_dir}/ce_scores_{n}.parquet")) for n in ("fit1", "val", "test")}
    lab = cast(labels()).with_columns(pl.lit(1, pl.Int8).alias("yy"))
    tr = _stack_feats(cast(_fit1_calibrated(rd)).join(ce["fit1"], on=KEYS)).join(lab, on=KEYS, how="left")
    y = tr["yy"].fill_null(0).to_numpy()
    prm = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.9,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=C.SEED, num_threads=C.WORKERS)
    m = lgb.train(prm, lgb.Dataset(tr.select(STACK_FEATS).to_numpy(), y), 400)
    m.save_model(str(od / "stacker.lgb"))
    info = {"fit1_band_rows": tr.height, "fit1_pos": int(y.sum())}
    for name, fname in (("val", "val_scores.parquet"), ("test", "test_scores.parquet")):
        base = cast(pl.read_parquet(rd / fname).select(*KEYS, "p"))
        bd = _stack_feats(base.join(ce[name], on=KEYS))
        bd = bd.with_columns(pl.Series("p2", m.predict(bd.select(STACK_FEATS).to_numpy()).astype("float32")))
        out = (base.join(bd.select(*KEYS, "p2"), on=KEYS, how="left")
                   .with_columns(pl.coalesce("p2", "p").alias("p")).drop("p2"))
        out.write_parquet(od / f"{name}_scores_{tag}.parquet")
        if name == "val":
            out.write_parquet(od / "val_scores.parquet")      # er.compare reads val_scores.parquet
        info[f"{name}_band"] = bd.height
    log.info("stack %s", info)
    return info


def loco_prepare(run: str, out: str, max_rows: int = 3_000_000, lo: float = 0.005, hi: float = 0.995) -> dict:
    """Transfer test (India plays the unseen country): stage-2 LightGBM trained on US fit fold-0 only ->
    scores US fit fold-1 (stacker training) and India val (evaluation). Writes the uncertain band of both
    with raw text for the US-only transformer (kernel er-ce-loco): ce_band_fit1 / ce_band_val."""
    import json

    import lightgbm as lgb
    import numpy as np
    from .pipeline import S2_PARAMS, _scan_b, labels, train_roles
    feats = json.loads((C.WORK / "runs" / run / "summary.json").read_text())["features"]
    roles = train_roles().select(pl.col("idx").cast(pl.UInt32).alias("s1_idx"), "role", "fold", "country")
    lab = labels().select(pl.col("s1_idx").cast(pl.UInt32), pl.col("src").cast(pl.UInt8), pl.col("t_idx").cast(pl.UInt32)).with_columns(pl.lit(1, pl.Int8).alias("y"))

    def load(flt):
        return (_scan_b("train").with_columns(pl.col("s1_idx").cast(pl.UInt32), pl.col("src").cast(pl.UInt8), pl.col("t_idx").cast(pl.UInt32))
                .join(roles.lazy().filter(flt), on="s1_idx").join(lab.lazy(), on=KEYS, how="left")
                .with_columns(pl.col("y").fill_null(0))
                .select(*KEYS, "y", *[pl.col(c).cast(pl.Float32) for c in feats if c not in KEYS]).collect(engine="streaming"))
    tr = load((pl.col("country") == "US") & (pl.col("role") == "fit") & (pl.col("fold") == 0))
    if tr.height > max_rows:
        tr = tr.sample(max_rows, seed=C.SEED)
    prm = {**S2_PARAMS, "num_threads": C.WORKERS}
    m = lgb.train(prm, lgb.Dataset(tr.select(feats).to_numpy(), tr["y"].to_numpy()), 800)
    del tr
    info = {}
    for name, flt, split in (("fit1", (pl.col("country") == "US") & (pl.col("role") == "fit") & (pl.col("fold") == 1), "train"),
                             ("val", (pl.col("country") == "India") & (pl.col("role") == "val"), "train")):
        d = load(flt)
        d = d.select(*KEYS, "y").with_columns(pl.Series("p", m.predict(d.select(feats).to_numpy()).astype(np.float32)))
        d.write_parquet(f"{out}/loco_scores_{name}.parquet")
        b = _attach(d.filter((pl.col("p") >= lo) & (pl.col("p") <= hi)).select(*KEYS, "p"), split)
        b.write_parquet(f"{out}/ce_band_{name}.parquet")
        info[name] = {"rows": d.height, "band": b.height}
    log.info("loco-prepare %s", info)
    return info


def loco_eval(dir_: str, run: str, n_boot: int = 2000) -> dict:
    """India val macro F0.5: US-only stage 2 vs US-only stage 2 + US-only transformer (stacker trained on
    US fit fold 1). Decision: the run's tuned parameters. Paired bootstrap over India val S1."""
    import json

    import lightgbm as lgb
    import numpy as np
    from .decide import DecisionParams, select
    from .pipeline import id_maps, train_roles
    from .prep import load_gt
    cast = lambda d: d.with_columns(pl.col("s1_idx").cast(pl.UInt32), pl.col("src").cast(pl.UInt8), pl.col("t_idx").cast(pl.UInt32))
    sc = {n: cast(pl.read_parquet(f"{dir_}/loco_scores_{n}.parquet")) for n in ("fit1", "val")}
    ce = {n: cast(pl.read_parquet(f"{dir_}/ce_scores_{n}.parquet")) for n in ("fit1", "val")}
    tr = _stack_feats(sc["fit1"].join(ce["fit1"], on=KEYS))
    prm = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.9,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=C.SEED, num_threads=C.WORKERS)
    m = lgb.train(prm, lgb.Dataset(tr.select(STACK_FEATS).to_numpy(), tr["y"].to_numpy()), 400)
    v = sc["val"]
    bd = _stack_feats(v.join(ce["val"], on=KEYS))
    bd = bd.with_columns(pl.Series("p2", m.predict(bd.select(STACK_FEATS).to_numpy()).astype("float32")))
    v2 = v.join(bd.select(*KEYS, "p2"), on=KEYS, how="left").with_columns(pl.coalesce("p2", "p").alias("p")).drop("p2")
    b = json.loads((C.WORK / "runs" / run / "decision.json").read_text())["best"]
    prm_d = DecisionParams(margin=b["margin"], m0=b["m0"], empty_bias=b["empty_bias"], coh=b.get("coh", 0.0))
    s1_ids, tg_ids = id_maps("train")
    ids = (train_roles().filter((pl.col("role") == "val") & (pl.col("country") == "India"))
           .select(pl.col("idx").alias("s1_idx")).join(s1_ids.select("s1_idx", "s1"), on="s1_idx"))
    gt = load_gt()
    from .selftrain import _per_s1_f
    tg = tg_ids.with_columns(pl.col("t_idx").cast(pl.UInt32), pl.col("src").cast(pl.UInt8))
    idc = ids.with_columns(pl.col("s1_idx").cast(pl.UInt32))
    fa = _per_s1_f(v.select(*KEYS, "p"), prm_d, idc, tg, gt)
    fb = _per_s1_f(v2.select(*KEYS, "p"), prm_d, idc, tg, gt)
    d = fb - fa
    rng = np.random.default_rng(0)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    res = {"india_unseen_f05_stage2_only": round(float(fa.mean()), 5), "with_transformer": round(float(fb.mean()), 5),
           "delta": round(float(d.mean()), 5), "ci95": [round(float(np.percentile(boots, 2.5)), 5), round(float(np.percentile(boots, 97.5)), 5)]}
    res["TRANSFERS"] = res["ci95"][0] > 0
    log.info("loco-eval %s", res)
    return res


def country_adapt_predict(run: str, adapt_dir: str, country: str, base_output: str, out_name: str) -> dict:
    """Finish the unseen-country adaptation: stacker (calibrated stage-2 p + adapted cross-encoder features) trained on
    the run's labelled FIT fold-1 band, applied to the country's uncertain test band; the run's own tuned decision on
    the country's full candidate set. Writes output/<out_name>: the country's rows re-decided (same candidate set),
    every other country's rows copied unchanged from output/<base_output>. Validator run at the end."""
    import json
    import subprocess
    import sys

    import lightgbm as lgb
    from .decide import DecisionParams, select, select_exact
    cast = lambda d: d.with_columns(pl.col("s1_idx").cast(pl.UInt32), pl.col("src").cast(pl.UInt8), pl.col("t_idx").cast(pl.UInt32))
    f1 = cast(pl.read_parquet(f"{adapt_dir}/fit1_scores.parquet"))
    ce_f1 = cast(pl.read_parquet(f"{adapt_dir}/ce_scores_fit1.parquet"))
    ce_te = cast(pl.read_parquet(f"{adapt_dir}/ce_scores_test.parquet"))
    tr = _stack_feats(f1.join(ce_f1, on=KEYS))
    m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.9,
                       bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=C.SEED, num_threads=C.WORKERS),
                  lgb.Dataset(tr.select(STACK_FEATS).to_numpy(), tr["y"].to_numpy()), 400)
    t = cast(pl.read_parquet(f"{adapt_dir}/country_test_scores.parquet"))
    bd = _stack_feats(t.join(ce_te, on=KEYS))
    bd = bd.with_columns(pl.Series("p2", m.predict(bd.select(STACK_FEATS).to_numpy()).astype("float32")))
    t2 = t.join(bd.select(*KEYS, "p2"), on=KEYS, how="left").with_columns(pl.coalesce("p2", "p").alias("p")).drop("p2")
    b = json.loads((C.WORK / "runs" / run / "decision.json").read_text())["best"]
    fn = select_exact if b.get("kind") == "exact" else select
    sel = fn(t2, DecisionParams(margin=b["margin"], m0=b["m0"], empty_bias=b["empty_bias"], coh=b.get("coh", 0.0)))
    s1 = pl.read_parquet(C.WORK / "test_s1.parquet", columns=["idx", "entity_id", "country"]).select(
        pl.col("idx").cast(pl.UInt32).alias("s1_idx"), pl.col("entity_id").alias("source1_entity_id"), "country")
    tg = pl.concat([pl.read_parquet(C.WORK / f"test_s{k}.parquet", columns=["idx", "entity_id"]).select(
        pl.col("idx").cast(pl.UInt32).alias("t_idx"), pl.lit(k).cast(pl.UInt8).alias("src"), pl.col("entity_id").alias("tid")) for k in (2, 3)])
    ids = sel.join(s1, on="s1_idx").join(tg, on=["t_idx", "src"])
    ctry = s1.filter(pl.col("country") == country).select("source1_entity_id")
    mr = (ctry.join(ids.group_by("source1_entity_id").agg(pl.col("tid").sort().str.join(",").alias("matched_entity_ids")), on="source1_entity_id", how="left")
              .with_columns(pl.col("matched_entity_ids").fill_null("")))
    od = C.OUT / out_name
    od.mkdir(parents=True, exist_ok=True)
    for fname, col, new in (("matching_results.tsv", "matched_entity_ids", mr), ("candidate_pairs.tsv", "candidate_entity_ids", None)):
        base = pl.read_csv(C.OUT / base_output / fname, separator="\t", infer_schema=False)
        if new is not None:
            keep = base.join(ctry, on="source1_entity_id", how="anti")
            outp = base.select("source1_entity_id").join(pl.concat([keep.with_columns(pl.col(col).fill_null("")), new.rename({"matched_entity_ids": col})]), on="source1_entity_id", maintain_order="left")
        else:
            outp = base      # same candidate set: the run scored exactly these pairs
        outp.write_csv(od / fname, separator="\t", quote_style="never")
    r = subprocess.run([sys.executable, str(C.ROOT / "student_resource" / "utils" / "validate_submission.py"), "--matching", str(od / "matching_results.tsv"),
                        "--candidate", str(od / "candidate_pairs.tsv"), "--test-dir", str(C.DATA / "test")], capture_output=True, text=True, encoding="utf-8")
    out = (r.stdout + r.stderr).strip().splitlines()
    return {"band": bd.height, "selected_pairs": sel.height, "country_rows": ctry.height,
            "mean_matches": float(mr["matched_entity_ids"].str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len().mean()),
            "validator": out[-1] if out else r.returncode}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["export-train", "export-band", "stack", "loco-prepare", "loco-eval", "export-biencoder", "export-bi-eval", "export-pseudo", "export-country-adapt", "country-predict"])
    ap.add_argument("--ce", default=None)
    ap.add_argument("--scores", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--country", default=None)
    ap.add_argument("--base-output", default=None)
    ap.add_argument("--out-run", default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--lo", type=float, default=0.005)
    ap.add_argument("--hi", type=float, default=0.995)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if a.what == "export-train":
        print(export_train(a.run, a.out))
    elif a.what == "export-band":
        print(export_band(a.run, a.out, a.lo, a.hi))
    elif a.what == "stack":
        print(stack(a.run, a.ce, a.out_run))
    elif a.what == "country-predict":
        print(country_adapt_predict(a.run, a.out, a.country, a.base_output, a.out_run))
    elif a.what == "export-country-adapt":
        print(export_country_adapt(a.run, a.country, a.out))
    elif a.what == "export-pseudo":
        print(export_pseudo(a.scores, a.split, a.out, country=a.country))
    elif a.what == "export-bi-eval":
        print(export_bi_eval(a.out))
    elif a.what == "export-biencoder":
        print(export_biencoder(a.out))
    elif a.what == "loco-prepare":
        print(loco_prepare(a.run, a.out))
    else:
        print(loco_eval(a.out, a.run))
