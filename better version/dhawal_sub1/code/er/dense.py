"""Dense-retrieval candidate augmentation (GPU route, lever 3): export for the Kaggle kernel er-dense.

Streams the raw source TSVs (row order = our idx, exactly as er.prep assigns it) to compact parquet for US/India,
plus the run's existing candidate keys (so the GPU only scores NEW pairs), roles and ground truth.
python -m er.dense export --run v4fr --out <dir>      (ER_WORK=<work dir of the run>)
"""
from __future__ import annotations

import argparse

import polars as pl

from . import config as C


def export(run: str, out: str, countries=("US", "India")) -> dict:
    info = {}
    for split in ("train", "test"):
        for s in ("1", "2", "3"):
            path = C.DATA / split / f"{split}_source{s}.tsv"
            (pl.scan_csv(path, separator="\t", quote_char=None, infer_schema=False, encoding="utf8-lossy")
               .with_row_index("idx").with_columns(pl.col("idx").cast(pl.UInt32), pl.col("country").fill_null(""))
               .filter(pl.col("country").is_in(list(countries)))
               .select("idx", "entity_id", "country", (pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null("")).alias("text"))
               .sink_parquet(f"{out}/{split}_s{s}.parquet"))
            info[f"{split}_s{s}"] = pl.scan_parquet(f"{out}/{split}_s{s}.parquet").select(pl.len()).collect().item()
    rd = C.WORK / "runs" / run
    k = lambda f: (pl.scan_parquet(rd / f).select(pl.col("s1_idx").cast(pl.UInt32), pl.col("src").cast(pl.UInt8), pl.col("t_idx").cast(pl.UInt32)))
    k("fit_oof.parquet").sink_parquet(f"{out}/cand_fit.parquet")
    k("val_scores.parquet").sink_parquet(f"{out}/cand_val.parquet")
    k("test_scores.parquet").sink_parquet(f"{out}/cand_test.parquet")
    pl.read_parquet(C.WORK / "train_roles.parquet").write_parquet(f"{out}/train_roles.parquet")
    pl.scan_csv(C.DATA / "train" / "train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False).sink_parquet(f"{out}/train_gt.parquet")
    return info


DENSE_FEATS = ["cos", "rank", "gap1", "ce", "lce", "src"]


def merge(base_run: str, dense_dir: str, out_run: str, tag: str = "ce") -> dict:
    """Score the new dense pairs with a LightGBM trained on FIT fold-1 dense pairs (labels from the ground truth)
    and add them to the base run's stacked scores -> runs/<out_run>/{val_scores, val_scores_<tag>, test_scores_<tag>}.
    Model files are hard-linked so tune / predict / compare work unchanged."""
    import os

    import lightgbm as lgb
    KEYS = ["s1_idx", "src", "t_idx"]
    rd, od = C.WORK / "runs" / base_run, C.WORK / "runs" / out_run
    od.mkdir(parents=True, exist_ok=True)
    for f in ("summary.json", "calibrators.pkl", "stage2_f0.lgb", "stage2_f1.lgb"):
        if not (od / f).exists():
            os.link(rd / f, od / f)
    lce = (pl.col("ce").clip(1e-6, 1 - 1e-6) / (1 - pl.col("ce").clip(1e-6, 1 - 1e-6))).log().alias("lce")
    ld = lambda n: pl.read_parquet(f"{dense_dir}/dense_{n}.parquet").with_columns(lce, pl.col("src").cast(pl.Float32).alias("srcf"))
    fe = [c if c != "src" else "srcf" for c in DENSE_FEATS]
    tr = ld("fit1")
    m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.9,
                       bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=C.SEED, num_threads=C.WORKERS),
                  lgb.Dataset(tr.select(fe).to_numpy(), tr["y"].to_numpy()), 300)
    m.save_model(str(od / "dense_stacker.lgb"))
    info = {"fit1_pairs": tr.height, "fit1_pos": int(tr["y"].sum())}
    for name in ("val", "test"):
        d = ld(name)
        d = d.select(pl.col("s1_idx").cast(pl.UInt32), pl.col("src").cast(pl.UInt8), pl.col("t_idx").cast(pl.UInt32),
                     pl.Series("p", m.predict(d.select(fe).to_numpy()).astype("float32")))
        base = pl.read_parquet(rd / f"{name}_scores_{tag}.parquet").select(pl.col("s1_idx").cast(pl.UInt32), pl.col("src").cast(pl.UInt8), pl.col("t_idx").cast(pl.UInt32), "p")
        u = pl.concat([base, d.join(base, on=KEYS, how="anti")])
        u.write_parquet(od / f"{name}_scores_{tag}.parquet")
        if name == "val":
            u.write_parquet(od / "val_scores.parquet")
        info[f"{name}_added"] = d.height
        info[f"{name}_added_p>0.5"] = int((d["p"] > 0.5).sum())
    return info


def add_test_pairs(run: str, dense_file: str, out_run: str, tag: str = "ce") -> dict:
    """Add extra test-only dense pairs (e.g. France, er-fr-dense) to a merged run: scored with the run's own dense
    stacker (trained on labelled FIT fold-1 US/India dense pairs), same decision. Validation files are shared unchanged."""
    import os
    import shutil

    import lightgbm as lgb
    KEYS = ["s1_idx", "src", "t_idx"]
    rd, od = C.WORK / "runs" / run, C.WORK / "runs" / out_run
    od.mkdir(parents=True, exist_ok=True)
    for f in ("summary.json", "calibrators.pkl", "stage2_f0.lgb", "stage2_f1.lgb", "dense_stacker.lgb",
              "val_scores.parquet", f"val_scores_{tag}.parquet"):
        if not (od / f).exists():
            os.link(rd / f, od / f)
    for f in ("decision.json", f"decision_{tag}.json"):
        shutil.copyfile(rd / f, od / f)
    m = lgb.Booster(model_file=str(rd / "dense_stacker.lgb"))
    lce = (pl.col("ce").clip(1e-6, 1 - 1e-6) / (1 - pl.col("ce").clip(1e-6, 1 - 1e-6))).log().alias("lce")
    d = pl.read_parquet(dense_file).with_columns(lce, pl.col("src").cast(pl.Float32).alias("srcf"))
    fe = [c if c != "src" else "srcf" for c in DENSE_FEATS]
    d = d.select(pl.col("s1_idx").cast(pl.UInt32), pl.col("src").cast(pl.UInt8), pl.col("t_idx").cast(pl.UInt32),
                 pl.Series("p", m.predict(d.select(fe).to_numpy()).astype("float32")))
    base = pl.read_parquet(rd / f"test_scores_{tag}.parquet").select(pl.col("s1_idx").cast(pl.UInt32), pl.col("src").cast(pl.UInt8), pl.col("t_idx").cast(pl.UInt32), "p")
    new = d.join(base, on=KEYS, how="anti")
    pl.concat([base, new]).write_parquet(od / f"test_scores_{tag}.parquet")
    return {"added": new.height, "added_p>0.5": int((new["p"] > 0.5).sum())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["export", "merge", "add-test"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dense", default=None)
    ap.add_argument("--out-run", default=None)
    a = ap.parse_args()
    if a.what == "export":
        print(export(a.run, a.out))
    elif a.what == "merge":
        print(merge(a.run, a.dense, a.out_run))
    else:
        print(add_test_pairs(a.run, a.dense, a.out_run))
