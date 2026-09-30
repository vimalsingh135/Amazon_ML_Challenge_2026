"""Stage orchestration.

Stage A  (per country x target-source partition, ALL S1): blocking -> cheap string features
          -> context (within-S1) and competition (within-target) features. Competition
          features see every S1, exactly as at test time.
Stage 1  : 2-fold out-of-fold LightGBM on stage-A features -> prune to final candidate set.
Stage B  : rich pair features on the pruned set -> stage-2 model -> decision layer.
"""
from __future__ import annotations

import functools
import gc
import logging
import os

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from . import config as C
from .blocking import block
from .prep import load_gt

log = logging.getLogger(__name__)

N_VAL = int(os.environ.get("ER_N_VAL", 150_000))   # held-out train S1 entities (validation, mirrors test)
N_FIT = int(os.environ.get("ER_N_FIT", 350_000))   # train S1 entities used to fit models


def load(split: str, src: str, country: str | None = None, cols: list[str] | None = None) -> pl.DataFrame:
    """Lazy, predicate-pushed load of a normalised table (optionally one country / some columns)."""
    q = pl.scan_parquet(C.WORK / f"{split}_s{src}.parquet")
    if country is not None:
        q = q.filter(pl.col("country") == country)
    if cols is not None:
        q = q.select(cols)
    return q.collect()


def train_roles() -> pl.DataFrame:
    """Deterministic S1 role assignment on train: fit / val (others unused for fitting)."""
    path = C.WORK / "train_roles.parquet"
    if path.exists():
        return pl.read_parquet(path)
    s1 = load("train", "1", cols=["entity_id", "idx", "country"]).sample(fraction=1.0, shuffle=True, seed=C.SEED)
    roles = pl.concat([
        s1.head(N_VAL).with_columns(pl.lit("val").alias("role")),
        s1.slice(N_VAL, N_FIT).with_columns(pl.lit("fit").alias("role")),
    ]).with_columns((pl.col("idx") % 2).cast(pl.UInt8).alias("fold"))
    roles.write_parquet(path)
    return roles


def _cheap_strings(pairs: pl.DataFrame, s1: pl.DataFrame, tg: pl.DataFrame, chunk: int = 4_000_000):
    """Core-name and address token-set ratios, computed in chunks to bound memory."""
    out_c, out_a = [], []
    for i in range(0, pairs.height, chunk):
        p = pairs.slice(i, chunk).select("s1_idx", "t_idx")
        a = p.join(s1, left_on="s1_idx", right_on="idx", how="left", maintain_order="left")
        b = p.join(tg, left_on="t_idx", right_on="idx", how="left", maintain_order="left")
        out_c.append(process.cpdist(a["core_s"].to_list(), b["core_s"].to_list(),
                                    scorer=fuzz.token_set_ratio, workers=-1, dtype=np.uint8))
        out_a.append(process.cpdist(a["addr_norm"].to_list(), b["addr_norm"].to_list(),
                                    scorer=fuzz.token_set_ratio, workers=-1, dtype=np.uint8))
    return np.concatenate(out_c), np.concatenate(out_a)


def cx_features(g: pl.DataFrame) -> pl.DataFrame:
    """Context features within each S1 (exact inside a blocking chunk)."""
    return g.with_columns(
        pl.len().over("s1_idx").cast(pl.UInt16).alias("cx_n"),
        (pl.col("bs") / pl.col("bs").max().over("s1_idx")).alias("cx_rel"),
        (pl.col("c_tset").cast(pl.Int16) - pl.col("c_tset").max().over("s1_idx")).alias("cx_gap_c"),
        (pl.col("a_tset").cast(pl.Int16) - pl.col("a_tset").max().over("s1_idx")).alias("cx_gap_a"),
        pl.col("c_tset").rank("ordinal", descending=True).over("s1_idx").cast(pl.UInt16).alias("cx_rank_c"),
    )


def target_aggs(files: list, score: str, extra: tuple[str, ...] = ()) -> pl.DataFrame:
    """Streaming per-target aggregates over all chunk files: count, max and runner-up of
    ``score`` and max of ``extra`` columns. Memory is proportional to #targets, not #pairs."""
    q = pl.scan_parquet(files).group_by("t_idx").agg(
        pl.len().cast(pl.UInt16).alias("tx_n"),
        pl.col(score).max().alias("_m1"),
        pl.col(score).top_k(2).min().alias("_m2"),
        *[pl.col(c).max().alias(f"_mx_{c}") for c in extra])
    return q.collect(engine="streaming")


def tx_features(g: pl.DataFrame, aggs: pl.DataFrame, score: str, extra: tuple[str, ...] = (),
                tag: str = "") -> pl.DataFrame:
    """Competition features of a pair against every S1 competing for the same target."""
    g = g.join(aggs, on="t_idx", how="left")
    best = pl.col(score) >= pl.col("_m1")
    other = pl.when(best).then(pl.when(pl.col("tx_n") <= 1).then(0.0).otherwise(pl.col("_m2"))
                               ).otherwise(pl.col("_m1"))
    g = g.with_columns(
        pl.when(best).then(1).when(pl.col(score) >= pl.col("_m2")).then(2).otherwise(3)
          .cast(pl.UInt8).alias(f"tx_rank{tag}"),
        (pl.col(score) - pl.col("_m1")).alias(f"tx_gap{tag}"),
        (pl.col(score) / pl.col("_m1")).alias(f"tx_rel{tag}"),
        (pl.col(score) - other).alias(f"tx_margin{tag}"),   # lead over the best *other* S1
        *[(pl.col(c).cast(pl.Int16) - pl.col(f"_mx_{c}").cast(pl.Int16)).alias(f"tx_gap_{c[0]}{tag}")
          for c in extra],
    )
    g = g.drop(["_m1", "_m2", *[f"_mx_{c}" for c in extra]])
    return g.rename({"tx_n": f"tx_n{tag}"}) if tag else g


def _str_view(df: pl.DataFrame) -> pl.DataFrame:
    return df.select("idx", pl.col("core").list.join(" ").alias("core_s"),
                     pl.col("addr_norm").fill_null(""), pl.col("hno").fill_null(""))


def _exact_flags(g: pl.DataFrame, a: pl.DataFrame, t: pl.DataFrame) -> pl.DataFrame:
    """House-number equality and missing-target-address flags (cheap, exact)."""
    return (g.join(a.select(pl.col("idx").alias("s1_idx"), pl.col("hno").alias("_h1")), on="s1_idx", how="left")
             .join(t.select(pl.col("idx").alias("t_idx"), pl.col("hno").alias("_h2"),
                            (pl.col("addr_norm").fill_null("") == "").cast(pl.UInt8).alias("a_missing2")),
                   on="t_idx", how="left")
             .with_columns(pl.when((pl.col("_h1") == "") | (pl.col("_h2") == "")).then(-1)
                             .otherwise((pl.col("_h1") == pl.col("_h2")).cast(pl.Int8))
                             .cast(pl.Int8).alias("h_eq"))
             .drop("_h1", "_h2"))


def _done(d) -> bool:
    return (d / "_SUCCESS").exists()


def _finish(d) -> None:
    (d / "_SUCCESS").write_text("ok")


def _files(d) -> list:
    return sorted(d.glob("part_*.parquet"))


def run_stage_a(split: str) -> None:
    """Memory-bounded: blocking chunks are featurised and spilled to disk immediately; the
    competition features come from streaming per-target aggregates joined back per chunk."""
    import shutil
    cols = ["idx", "country", "core", "skel", "loc", "street", "nums", "addr_norm", "hno", "business_address"]
    countries = load(split, "1", cols=["country"])["country"].unique().sort().to_list()
    for country in countries:
        a = None
        for src in ("2", "3"):
            out = C.WORK / f"{split}_A_{country}_{src}"
            if _done(out):
                continue
            shutil.rmtree(out, ignore_errors=True)
            tmp = out / "tmp"
            tmp.mkdir(parents=True)
            if a is None:
                a = load(split, "1", country, cols)
                if split == "train" and C.TRAIN_ROLES_ONLY:   # fast path: only fit/val S1
                    a = a.join(train_roles().select("idx"), on="idx", how="semi")
            t = load(split, src, country, cols)
            if a.height == 0 or t.height == 0:
                _finish(out)
                continue
            av, tv = _str_view(a), _str_view(t)

            def on_chunk(g: pl.DataFrame, ch: int, a=a, t=t, av=av, tv=tv, tmp=tmp, src=src) -> None:
                g = g.with_columns(pl.col("bs", "bs_name", "bs_addr", "bs_pure", "arcs").cast(pl.Float32),
                                   pl.col("nkeys", "nfam", "nk_name", "nk_addr", "brank", "brank_n",
                                          "brank_a", "brank_p").cast(pl.UInt16))
                c_tset, a_tset = _cheap_strings(g, av, tv)
                g = g.with_columns(pl.Series("c_tset", c_tset), pl.Series("a_tset", a_tset))
                g = cx_features(_exact_flags(g, a, t)).with_columns(pl.lit(int(src), pl.UInt8).alias("src"))
                g.write_parquet(tmp / f"part_{ch:05d}.parquet")

            block(a, t, cap_single=C.CAP_SINGLE, cap_pair=C.CAP_PAIR, join_budget=C.JOIN_BUDGET, on_chunk=on_chunk,
                  **C.BUDGETS)
            del t, tv
            gc.collect()
            files = _files(tmp)
            aggs = target_aggs(files, "bs", ("c_tset", "a_tset"))
            for f in files:
                tx_features(pl.read_parquet(f), aggs, "bs", ("c_tset", "a_tset")).write_parquet(out / f.name)
            shutil.rmtree(tmp)
            _finish(out)
            log.info("stage A %s %s S%s done (%d chunks)", split, country, src, len(files))
            gc.collect()


def id_maps(split: str):
    s1 = load(split, "1", cols=["idx", "entity_id", "country"]).rename({"idx": "s1_idx", "entity_id": "s1"})
    tg = pl.concat([load(split, s, cols=["idx", "entity_id"]).rename({"idx": "t_idx", "entity_id": "tid"})
                    .with_columns(pl.lit(int(s), pl.UInt8).alias("src")) for s in ("2", "3")])
    return s1, tg


def labels(split: str = "train") -> pl.DataFrame:
    """(s1_idx, src, t_idx) of true matches."""
    s1, tg = id_maps(split)
    return (load_gt().join(s1.select("s1", "s1_idx"), on="s1").join(tg, on="tid")
            .select("s1_idx", "src", "t_idx"))


# ---------------------------------------------------------------------------
# Stage 1: out-of-fold pruning ranker
# ---------------------------------------------------------------------------
S1_FEATS = ["bs", "bs_name", "bs_addr", "bs_pure", "arcs", "nk_name", "nk_addr", "fam_mask",
            "brank_p", "h_eq", "a_missing2",
            "nkeys", "nfam", "brank", "brank_n", "brank_a", "c_tset",
            "a_tset", "cx_n", "cx_rel", "cx_gap_c", "cx_gap_a", "cx_rank_c", "tx_n", "tx_rank",
            "tx_gap", "tx_rel", "tx_margin", "tx_gap_c", "tx_gap_a", "src"]
S1_PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=200,
                 feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                 num_threads=C.WORKERS, verbose=-1, seed=C.SEED)
K1 = 20          # max candidates per S1 per source kept after stage 1
EPS1 = float(os.environ.get("ER_EPS1", 0.002))   # min stage-1 probability to keep


def _parts(split: str, stage: str):
    """Stage outputs named {split}_{stage}_{country}_{src}: directories (A) or .parquet files (P, B)."""
    return sorted(p for p in C.WORK.glob(f"{split}_{stage}_*") if p.is_dir() or p.suffix == ".parquet")


def s1_feats() -> list[str]:
    return [f for f in S1_FEATS if not (C.NO_TX and f.startswith("tx_"))]


def train_stage1() -> None:
    import lightgbm as lgb
    paths = [C.WORK / f"stage1_f{f}.lgb" for f in (0, 1)]
    if all(p.exists() for p in paths):
        return
    roles = train_roles().filter(pl.col("role") == "fit").select(pl.col("idx").alias("s1_idx"), "fold")
    lab = labels().with_columns(pl.lit(1, pl.UInt8).alias("y"))
    neg_mod = int(os.environ.get("ER_S1_NEG_MOD", 4))   # keep 1/neg_mod negatives, weight neg_mod (unbiased)
    data = []
    for d_ in _parts("train", "A"):
        d = (pl.scan_parquet(_files(d_)).join(roles.lazy(), on="s1_idx")
             .join(lab.lazy(), on=["s1_idx", "src", "t_idx"], how="left").with_columns(pl.col("y").fill_null(0))
             .filter((pl.col("y") == 1) | (pl.col("t_idx").hash(C.SEED) % neg_mod == 0))   # 1/neg_mod of negatives
             .select(*s1_feats(), "y", "fold").collect(engine="streaming"))
        data.append(d.with_columns(pl.when(pl.col("y") == 1).then(1.0).otherwise(float(neg_mod)).alias("w")))
    data = pl.concat(data)
    log.info("stage1 train rows=%d pos=%d", data.height, data["y"].sum())
    for f in (0, 1):
        tr, va = data.filter(pl.col("fold") != f), data.filter(pl.col("fold") == f)
        fs = s1_feats()
        dtr = lgb.Dataset(tr.select(fs).to_numpy(), tr["y"].to_numpy(), weight=tr["w"].to_numpy(),
                          feature_name=fs)
        dva = lgb.Dataset(va.select(fs).to_numpy(), va["y"].to_numpy(), weight=va["w"].to_numpy(),
                          reference=dtr)
        m = lgb.train(S1_PARAMS, dtr, 600, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(100)])
        m.save_model(str(paths[f]))
        log.info("stage1 fold %d best_iter=%d", f, m.best_iteration)


def _stage1_predict(d: pl.DataFrame, models, fold: pl.Series | None) -> np.ndarray:
    X = d.select(s1_feats()).to_numpy()
    if fold is None and os.environ.get("ER_S1_SINGLE") == "1":   # test-time speed: one fold model
        return models[0].predict(X, num_threads=C.WORKERS)
    p = [m.predict(X, num_threads=C.WORKERS) for m in models]
    avg = (p[0] + p[1]) / 2
    if fold is None:
        return avg
    f = fold.to_numpy()
    # out-of-fold for fit rows: model trained on the other fold
    return np.where(f == 0, p[0], np.where(f == 1, p[1], avg))


def run_stage1(split: str) -> None:
    """Score every retrieved pair (two streaming passes), then prune to the final candidate set."""
    import shutil

    import lightgbm as lgb
    models = [lgb.Booster(model_file=str(C.WORK / f"stage1_f{f}.lgb")) for f in (0, 1)]
    roles = train_roles() if split == "train" else None
    fm = (roles.filter(pl.col("role") == "fit").select(pl.col("idx").alias("s1_idx"), "fold")
          if roles is not None else None)
    for d_ in _parts(split, "A"):
        out = C.WORK / (d_.name.replace("_A_", "_P_") + ".parquet")
        if out.exists() or not _done(d_):   # skip finished outputs and still-running upstream partitions
            continue
        tmp = C.WORK / d_.name.replace("_A_", "_S_")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        for f in _files(d_):  # pass 1: score + rank within S1 (complete inside a chunk)
            d = pl.read_parquet(f)
            if fm is not None:
                d = d.join(fm, on="s1_idx", how="left")
                p1 = _stage1_predict(d, models, d["fold"].fill_null(9))
                d = d.drop("fold")
            else:
                p1 = _stage1_predict(d, models, None)
            d = d.with_columns(pl.Series("p1", p1.astype(np.float32)))
            d.with_columns(pl.col("p1").rank("ordinal", descending=True).over("s1_idx").cast(pl.UInt16)
                           .alias("r1")).write_parquet(tmp / f.name)
        files = _files(tmp)  # pass 2: p1 competition over ALL S1, then prune
        extra = (pl.scan_parquet(files).group_by("t_idx").agg(
                     pl.col("p1").sum().alias("tx_sum_p1"),
                     (pl.col("p1") >= 0.5).sum().cast(pl.UInt16).alias("tx_n_hi"))
                 .collect(engine="streaming"))
        aggs = target_aggs(files, "p1")
        keep = []
        for f in files:
            d = pl.scan_parquet(f).filter((pl.col("r1") <= K1) & (pl.col("p1") >= EPS1))
            if roles is not None:
                d = d.join(roles.lazy().select(pl.col("idx").alias("s1_idx")), on="s1_idx", how="semi")
            d = d.collect().join(extra, on="t_idx", how="left")
            keep.append(tx_features(d, aggs, "p1", tag="_p1"))
        tmp_out = out.with_suffix(".part")
        pl.concat(keep).write_parquet(tmp_out)
        tmp_out.replace(out)            # atomic: downstream never sees a half-written file
        shutil.rmtree(tmp)
        log.info("stage1 %s -> %d pairs", out.name, sum(k.height for k in keep))


# ---------------------------------------------------------------------------
# Stage B: rich pair features on the pruned candidate set
# ---------------------------------------------------------------------------
from .features import REC_COLS, name_freq, pair_features, token_idf  # noqa: E402

META = {"s1_idx", "t_idx", "country", "fold", "role", "y", "w"}


def _part_meta(path) -> tuple[str, str, str]:
    """{split}_{stage}_{country}_{src}[.parquet] -> (split, country, src)."""
    stem = path.name.removesuffix(".parquet")
    split, _, rest = stem.split("_", 2)
    country, src = rest.rsplit("_", 1)
    return split, country, src


def _anchors(split: str, country: str) -> pl.DataFrame:
    """Top stage-1 candidate per (S1, source): the anchor for cluster-coherence features."""
    ps = [C.WORK / f"{split}_P_{country}_{s}.parquet" for s in ("2", "3")]
    d = pl.concat([pl.read_parquet(p, columns=["s1_idx", "src", "t_idx", "p1"]) for p in ps if p.exists()])
    return (d.sort("p1", descending=True).group_by("s1_idx", "src", maintain_order=True).first()
             .rename({"t_idx": "a_idx", "p1": "a_p1"}))


def _idf_stream(split: str, country: str, col: str) -> pl.DataFrame:
    """Same result as features.token_idf over the three sources, computed out-of-core."""
    scans = [pl.scan_parquet(C.WORK / f"{split}_s{s}.parquet").filter(pl.col("country") == country)
             for s in ("1", "2", "3")]
    n = sum(q.select(pl.len()).collect().item() for q in scans)
    df = (pl.concat([q.select(pl.col(col).list.unique().alias("tok")) for q in scans])
            .explode("tok").drop_nulls("tok").group_by("tok").len("df").collect(engine="streaming"))
    return df.select(pl.lit(country).alias("country"), "tok",
                     ((n + 1) / (pl.col("df") + 1)).log().cast(pl.Float32).alias("idf"))


def _name_freq_stream(split: str, country: str, src: str, suffix: str) -> pl.DataFrame:
    return (pl.scan_parquet(C.WORK / f"{split}_s{src}.parquet").filter(pl.col("country") == country)
              .group_by("name_norm").len(f"nfreq{suffix}").rename({"name_norm": f"name_norm{suffix}"})
              .collect(engine="streaming"))


def _country_tables(split: str, country: str, aux):
    """Record tables restricted to records that occur in this country's candidate pairs, plus
    corpus statistics (IDF, name frequency) computed over the *full* tables, column-pruned."""
    ps = [C.WORK / f"{split}_P_{country}_{s}.parquet" for s in ("2", "3")]
    pairs = pl.concat([pl.read_parquet(p, columns=["s1_idx", "src", "t_idx"]) for p in ps if p.exists()])
    need = {"1": pairs.select(pl.col("s1_idx").alias("idx")).unique()}
    for s in ("2", "3"):
        need[s] = pairs.filter(pl.col("src") == int(s)).select(pl.col("t_idx").alias("idx")).unique()
    # corpus statistics via streaming aggregation (tables are never materialised in memory)
    idf_c, idf_a = _idf_stream(split, country, "core"), _idf_stream(split, country, "atoks")
    nf1 = _name_freq_stream(split, country, "1", "_1")
    nf2 = {s: _name_freq_stream(split, country, s, "_2") for s in ("2", "3")}
    gc.collect()
    tabs = {}
    for s in ("1", "2", "3"):
        t = (pl.scan_parquet(C.WORK / f"{split}_s{s}.parquet").filter(pl.col("country") == country)
               .select("idx", *REC_COLS).join(need[s].lazy(), on="idx", how="semi").collect())
        tabs[s] = aux.enrich(t) if aux is not None else t
    return tabs, idf_c, idf_a, nf1, nf2, _anchors(split, country)


def run_stage_b(split: str, chunk: int = int(os.environ.get("ER_B_CHUNK", 500_000))) -> None:
    import shutil

    from .auxfit import AUX, Aux
    from .features import coherence_features
    aux = Aux() if (AUX / "addr_alias.json").exists() else None
    if aux is None:
        log.warning("stage B without aux models (run `er.run aux` first for alias/noise features)")
    cache: dict[str, tuple] = {}
    for p in _parts(split, "P"):
        out = C.WORK / p.name.removesuffix(".parquet").replace("_P_", "_B_")
        if _done(out):
            continue
        _, country_, _ = _part_meta(p)
        only = os.environ.get("ER_COUNTRIES")
        if only and country_ not in only.split(","):
            continue
        # coherence anchors need BOTH sources' stage-1 output for this country
        if not all((C.WORK / f"{split}_P_{country_}_{s}.parquet").exists() for s in ("2", "3")):
            continue
        lock = C.WORK / f"{out.name}.lock"   # lets several stage-B workers share the partitions safely
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
        except FileExistsError:
            continue
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir()
        _, country, src = _part_meta(p)
        if country not in cache:
            cache.clear()
            gc.collect()
            cache[country] = _country_tables(split, country, aux)
        tabs, idf_c, idf_a, nf1, nf2, anchors = cache[country]
        tg = tabs[src].with_columns(pl.lit(int(src), pl.UInt8).alias("src"))
        d = pl.read_parquet(p)
        minp1 = float(os.environ.get("ER_B_MINP1", 0))   # optional extra stage-1 floor (time budget)
        if minp1 > 0:
            d = d.filter(pl.col("p1") >= minp1)
        for k, i in enumerate(range(0, d.height, chunk)):
            f = pair_features(d.slice(i, chunk), tabs["1"], tg, idf_c, idf_a, nf1, nf2[src])
            coherence_features(f, tabs, anchors).drop("country", strict=False).write_parquet(
                out / f"part_{k:05d}.parquet")
        _finish(out)
        lock.unlink(missing_ok=True)
        log.info("stage B %s: %d pairs", out.name, d.height)
        del d
        gc.collect()


# ---------------------------------------------------------------------------
# Stage 2: matcher, calibration, decision tuning
# ---------------------------------------------------------------------------
S2_PARAMS = dict(objective="binary", learning_rate=float(os.environ.get("ER_S2_LR", 0.05)),
                 num_leaves=int(os.environ.get("ER_S2_LEAVES", 255)), min_data_in_leaf=100,
                 feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=3.0,
                 max_bin=255, num_threads=C.WORKERS, verbose=-1, seed=C.SEED)
S2_ROUNDS = int(os.environ.get("ER_S2_ROUNDS", 3000))
S2_EARLY = int(os.environ.get("ER_S2_EARLY", 100))


SIB = os.environ.get("ER_SIB", "0") == "1"   # sibling-aware target features (France: 21% of S1 have siblings)


@functools.lru_cache(maxsize=2)
def sibling_aggs(split: str) -> pl.DataFrame:
    """Per target, over ALL its stage-A S1 candidates (every S1, the same view as test): how many share its house
    number / have a different one, the best name similarity among same-number candidates, how many look alike by
    name. Siblings (same brand + street, other house number) are 21% of French S1 vs 7% India, 0.3% US."""
    path = C.WORK / f"sib_{split}.parquet"
    if not path.exists():
        files = [f for d in _parts(split, "A") for f in _files(d)]
        agg = (pl.scan_parquet(files).select("src", "t_idx", "h_eq", "c_tset")
                 .group_by("src", "t_idx").agg(
                     (pl.col("h_eq") == 1).sum().cast(pl.Float32).alias("sib_heq_n"),
                     (pl.col("h_eq") == 0).sum().cast(pl.Float32).alias("sib_hne_n"),
                     pl.when(pl.col("h_eq") == 1).then(pl.col("c_tset")).otherwise(None).max().cast(pl.Float32).alias("sib_heq_cmax"),
                     (pl.col("c_tset") >= 80).sum().cast(pl.Float32).alias("sib_alike_n"),
                     pl.col("c_tset").max().cast(pl.Float32).alias("sib_cmax"))
                 .collect(engine="streaming"))
        tmp = path.with_suffix(".part")
        agg.write_parquet(tmp)
        tmp.replace(path)
    return pl.read_parquet(path)


def _add_sib(lf: pl.LazyFrame, split: str) -> pl.LazyFrame:
    s = sibling_aggs(split)
    return (lf.join(s.lazy(), on=["src", "t_idx"], how="left")
              .with_columns((pl.col("sib_heq_n").fill_null(0) - (pl.col("h_eq") == 1).cast(pl.Float32)).alias("sib_heq_oth"),
                            (pl.col("c_tset") - pl.col("sib_heq_cmax")).alias("sib_heq_cgap"),
                            (pl.col("c_tset") - pl.col("sib_cmax")).alias("sib_cgap")))


def _scan_b(split: str) -> pl.LazyFrame:
    files = [f for d in _parts(split, "B") for f in _files(d)]
    lf = pl.scan_parquet(files, missing_columns="insert", extra_columns="ignore")
    return _add_sib(lf, split) if SIB else lf


def feature_cols(schema) -> list[str]:
    return [c for c, t in schema.items() if c not in META and t.is_numeric()
            and not (C.NO_TX and c.startswith("tx_"))]


def _predict_b(split: str, models, feats, cal, roles: pl.DataFrame | None = None, role: str | None = None
               ) -> pl.DataFrame:
    """Chunk-wise scoring of stage-B parts -> (s1_idx, src, t_idx, p[, y])."""
    out = []
    for d_ in _parts(split, "B"):
        for f in _files(d_):
            d = pl.read_parquet(f)
            if SIB:
                d = _add_sib(d.lazy(), split).collect()
            if roles is not None:
                d = d.join(roles.filter(pl.col("role") == role).select("s1_idx"), on="s1_idx", how="semi")
            if d.height == 0:
                continue
            keep = ["s1_idx", "src", "t_idx"] + (["y"] if "y" in d.columns else [])
            out.append(d.select(keep).with_columns(pl.Series("p", predict2(d, models, feats, cal))))
    return pl.concat(out)


def run_train2(run_name: str = "v1") -> dict:
    """2-fold OOF LightGBM on fit S1, isotonic calibration on OOF, decision tuning on val."""
    import json
    import pickle

    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression

    from .decide import DecisionParams, select, threshold_select
    from .metric import macro_f05

    run_dir = C.WORK / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    roles = train_roles().select(pl.col("idx").alias("s1_idx"), "role", "fold")
    lab = labels().with_columns(pl.lit(1, pl.UInt8).alias("y"))
    scan = _scan_b("train")
    feats = feature_cols(scan.collect_schema())
    fit = (scan.join(roles.lazy().filter(pl.col("role") == "fit"), on="s1_idx")
               .join(lab.lazy(), on=["s1_idx", "src", "t_idx"], how="left")
               .with_columns(pl.col("y").fill_null(0))
               .select("s1_idx", "t_idx", "y", "fold", *[pl.col(c).cast(pl.Float32) for c in feats])
               .collect(engine="streaming"))
    log.info("stage2: fit=%d (pos %d) feats=%d", fit.height, fit["y"].sum(), len(feats))
    models, oof = [], np.zeros(fit.height, dtype=np.float32)
    fold = fit["fold"].to_numpy()
    y = fit["y"].to_numpy()
    X = fit.select(feats).to_numpy()
    meta = fit.select("s1_idx", "src", "t_idx", "y")
    del fit
    gc.collect()
    # optional self-training rows (e.g. confident France test pseudo-labels): appended to the TRAINING
    # folds only - never to OOF calibration or validation, so all reported metrics stay label-honest
    Xp = yp = None
    if os.environ.get("ER_PSEUDO"):
        ps = pl.read_parquet(os.environ["ER_PSEUDO"]).select("s1_idx", "src", "t_idx", pl.col("y_pseudo").cast(pl.Int32))
        pb = (_scan_b("test").join(ps.lazy(), on=["s1_idx", "src", "t_idx"])
              .select("y_pseudo", *[pl.col(c).cast(pl.Float32) for c in feats]).collect(engine="streaming"))
        Xp, yp = pb.select(feats).to_numpy(), pb["y_pseudo"].to_numpy()
        wpseudo = float(os.environ.get("ER_PSEUDO_W", 1.0))
        log.info("stage2: +%d pseudo-labelled rows (pos %d, weight %.2f)", len(yp), int(yp.sum()), wpseudo)
        del pb
    for f in (0, 1):
        tr, va = fold != f, fold == f
        if Xp is not None:
            dtr = lgb.Dataset(np.vstack([X[tr], Xp]), np.concatenate([y[tr], yp]), feature_name=feats,
                              weight=np.concatenate([np.ones(int(tr.sum())), np.full(len(yp), wpseudo)]),
                              free_raw_data=True)
        else:
            dtr = lgb.Dataset(X[tr], y[tr], feature_name=feats, free_raw_data=True)
        dva = lgb.Dataset(X[va], y[va], reference=dtr)
        m = lgb.train(S2_PARAMS, dtr, S2_ROUNDS, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(S2_EARLY, verbose=False), lgb.log_evaluation(250)])
        oof[va] = m.predict(X[va], num_threads=C.WORKERS)
        m.save_model(str(run_dir / f"stage2_f{f}.lgb"))
        models.append(m)
        log.info("stage2 fold %d best_iter=%d", f, m.best_iteration)
        del dtr, dva
    del X
    gc.collect()
    cal = {}  # calibration per source on out-of-fold predictions
    for s in (2, 3):
        msk = meta["src"].to_numpy() == s
        cal[s] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof[msk], y[msk])
    with open(run_dir / "calibrators.pkl", "wb") as fh:
        pickle.dump(cal, fh)
    meta.with_columns(pl.Series("oof", oof)).write_parquet(run_dir / "fit_oof.parquet")
    val = (_predict_b("train", models, feats, cal, roles, "val")
           .join(lab, on=["s1_idx", "src", "t_idx"], how="left").with_columns(pl.col("y").fill_null(0)))
    val.write_parquet(run_dir / "val_scores.parquet")
    # decision tuning on the untouched validation S1 set (incl. S1 with no candidates)
    s1_ids, tg_ids = id_maps("train")
    val_s1 = roles.filter(pl.col("role") == "val").join(s1_ids, on="s1_idx")
    truth = load_gt().join(val_s1.select("s1"), on="s1")

    def score(sel: pl.DataFrame, s1s: pl.DataFrame = val_s1, tr: pl.DataFrame = truth) -> dict:
        pred = sel.join(s1s.select("s1_idx", "s1"), on="s1_idx").join(tg_ids, on=["t_idx", "src"])
        return macro_f05(pred.select("s1", "tid"), tr, s1s["s1"])

    results = [{"kind": "thr", "thr": thr, **score(threshold_select(val, thr))} for thr in (0.3, 0.4, 0.5, 0.6, 0.7)]
    best = None
    for m0 in (0.0, 0.2, 0.5):
        for eb in (0.5, 1.0, 1.5):
            for margin in (0.0, 0.1):
                prm = DecisionParams(margin=margin, m0=m0, empty_bias=eb)
                r = {"kind": "ef", "m0": m0, "empty_bias": eb, "margin": margin, **score(select(val, prm))}
                results.append(r)
                if best is None or r["f05"] > best["f05"]:
                    best = r
    res = pl.DataFrame(results, infer_schema_length=None).sort("f05", descending=True)
    print(res.head(12))
    res.write_csv(run_dir / "decision_grid.csv")
    imp = sorted(zip(feats, models[0].feature_importance("gain")), key=lambda x: -x[1])
    summary = {"run": run_name, "features": feats, "best": best, "val_by_country": {},
               "top_features": [(f, float(g)) for f, g in imp[:40]]}
    sel = select(val, DecisionParams(margin=best["margin"], m0=best["m0"], empty_bias=best["empty_bias"]))
    for (country,), g in val_s1.group_by("country"):
        summary["val_by_country"][country] = score(sel, g, truth.join(g.select("s1"), on="s1"))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    log.info("stage2 best %s | by country %s", best, summary["val_by_country"])
    return summary


def predict2(df: pl.DataFrame, models, feats, cal) -> np.ndarray:
    X = df.select([pl.col(f).cast(pl.Float32) if f in df.columns else pl.lit(None, pl.Float32).alias(f)
                   for f in feats]).to_numpy()
    raw = np.mean([m.predict(X, num_threads=C.WORKERS) for m in models], axis=0)
    out = raw.copy()  # sources without a calibrator keep raw scores
    src = df["src"].to_numpy()
    for s, c in cal.items():
        msk = src == s
        if msk.any():
            out[msk] = c.predict(raw[msk])
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Test inference -> submission
# ---------------------------------------------------------------------------
def run_predict(run_name: str = "v1", tag: str = "") -> None:
    import json
    import pickle

    import lightgbm as lgb

    from .decide import DecisionParams, select
    from .output import write_submission

    run_dir = C.WORK / "runs" / run_name
    summ = json.loads((run_dir / "summary.json").read_text())
    feats, best = summ["features"], summ["best"]
    sfx = f"_{tag}" if tag else ""
    dec_file = run_dir / os.environ.get("ER_DECISION", f"decision{sfx}.json")
    if dec_file.exists():   # tuned decision layer (er.run tune / refined grid) takes precedence
        best = json.loads(dec_file.read_text())["best"]
    models = [lgb.Booster(model_file=str(run_dir / f"stage2_f{f}.lgb")) for f in (0, 1)]
    with open(run_dir / "calibrators.pkl", "rb") as fh:
        cal = pickle.load(fh)
    if tag:  # scores precomputed by another scorer (e.g. the ensemble)
        d = pl.read_parquet(run_dir / f"test_scores{sfx}.parquet")
    else:
        d = _predict_b("test", models, feats, cal)
        d.write_parquet(run_dir / "test_scores.parquet")
    from .decide import select_exact
    fn = select_exact if best.get("kind") == "exact" else select
    sel = fn(d, DecisionParams(margin=best["margin"], m0=best["m0"], empty_bias=best["empty_bias"], coh=best.get("coh", 0.0)))
    log.info("predict: decision %s", best)
    s1_ids, tg_ids = id_maps("test")
    to_ids = lambda x: (x.join(s1_ids.select("s1_idx", "s1"), on="s1_idx")
                         .join(tg_ids, on=["t_idx", "src"]).select("s1", "tid"))
    out_dir = C.OUT / os.environ["ER_SUB_NAME"] if os.environ.get("ER_SUB_NAME") else C.OUT
    write_submission(to_ids(sel), to_ids(d.select("s1_idx", "src", "t_idx")), s1_ids["s1"], out_dir,
                     validator=C.ROOT / "student_resource" / "utils" / "validate_submission.py",
                     test_dir=C.DATA / "test")
    log.info("predict: %d matches over %d test S1 (%d with >=1 match)", sel.height, s1_ids.height,
             sel["s1_idx"].n_unique())


# ---------------------------------------------------------------------------
# Decision tuning from saved validation scores (no retraining)
# ---------------------------------------------------------------------------
COH_GRID = tuple(float(x) for x in os.environ.get("ER_COH_GRID", "0,1").split(","))


def run_tune(run_name: str = "v1", tag: str = "") -> dict:
    import json

    from .decide import DecisionParams, select, select_exact
    from .metric import macro_f05

    rd = C.WORK / "runs" / run_name
    sfx = f"_{tag}" if tag else ""
    val = pl.read_parquet(rd / f"val_scores{sfx}.parquet")
    roles = train_roles().select(pl.col("idx").alias("s1_idx"), "role", "country")
    s1_ids, tg_ids = id_maps("train")
    val_s1 = roles.filter(pl.col("role") == "val").join(s1_ids.select("s1_idx", "s1"), on="s1_idx")
    truth = load_gt().join(val_s1.select("s1"), on="s1")

    def score(sel, s1s=val_s1, tr=truth):
        pred = sel.join(s1s.select("s1_idx", "s1"), on="s1_idx").join(tg_ids, on=["t_idx", "src"])
        return macro_f05(pred.select("s1", "tid"), tr, s1s["s1"])

    rows = []
    for kind, fn in (("ratio", select), ("exact", select_exact)):
        for m0 in (0.2, 0.5, 0.75):
            for eb in (1.0, 1.5, 2.0, 2.5):
                for margin in (0.05, 0.1):
                    for coh in COH_GRID:     # one-to-one coherent posterior off / on (decide N3); ER_COH_GRID
                        prm = DecisionParams(margin=margin, m0=m0, empty_bias=eb, coh=coh)
                        rows.append({"kind": kind, "m0": m0, "empty_bias": eb, "margin": margin, "coh": coh,
                                     **score(fn(val, prm))})
    res = pl.DataFrame(rows).sort("f05", descending=True)
    res.write_csv(rd / f"tune_grid{sfx}.csv")
    best = res.row(0, named=True)
    fn = select_exact if best["kind"] == "exact" else select
    sel = fn(val, DecisionParams(margin=best["margin"], m0=best["m0"], empty_bias=best["empty_bias"], coh=best.get("coh", 0.0)))
    by_c = {c: score(sel, g, truth.join(g.select("s1"), on="s1"))
            for (c,), g in val_s1.group_by("country")}
    out = {"best": best, "by_country": by_c,
           "best_ratio": res.filter(pl.col("kind") == "ratio").row(0, named=True),
           "best_exact": res.filter(pl.col("kind") == "exact").row(0, named=True)}
    (rd / f"decision{sfx}.json").write_text(json.dumps(out, indent=1, default=str))
    log.info("tune: best %s | by country %s", best, by_c)
    return out
