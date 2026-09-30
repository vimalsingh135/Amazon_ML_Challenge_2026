"""Pair feature computation (vectorised: rapidfuzz cpdist + Polars list/set ops)."""
from __future__ import annotations

import logging
import math

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

log = logging.getLogger(__name__)

REC_COLS = ["name_norm", "core", "legal", "skel", "addr_norm", "atoks", "nums", "hno", "state",
            "loc", "street", "country"]
EXTRA_COLS = ["core_a", "atoks_a", "nrep"]   # optional, added by er.auxfit.Aux.enrich


def token_idf(tables: list[pl.DataFrame], col: str) -> pl.DataFrame:
    """Per-country smoothed IDF for tokens in list column ``col`` over the given tables."""
    parts = [t.select("country", pl.col(col).list.unique().alias("tok")) for t in tables]
    ex = pl.concat(parts).explode("tok").drop_nulls("tok")
    n = pl.concat([t.select("country") for t in tables]).group_by("country").len("n")
    df = ex.group_by("country", "tok").len("df").join(n, on="country")
    return df.select("country", "tok",
                     ((pl.col("n") + 1) / (pl.col("df") + 1)).log().cast(pl.Float32).alias("idf"))


def _cp(scorer, a, b, **kw) -> np.ndarray:
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw)


def _set_feats(df: pl.DataFrame, col: str, name: str) -> list[pl.Expr]:
    a, b = pl.col(col + "_1"), pl.col(col + "_2")
    inter = a.list.set_intersection(b).list.len().cast(pl.Float32)
    la, lb = a.list.unique().list.len().cast(pl.Float32), b.list.unique().list.len().cast(pl.Float32)
    return [
        inter.alias(f"{name}_inter"),
        (inter / (la + lb - inter)).fill_nan(0).alias(f"{name}_jac"),
        (inter / la).fill_nan(0).alias(f"{name}_c12"),
        (inter / lb).fill_nan(0).alias(f"{name}_c21"),
        la.alias(f"{name}_n1"), lb.alias(f"{name}_n2"),
    ]


def _idf_overlap(df: pl.DataFrame, col: str, idf: pl.DataFrame, name: str) -> pl.DataFrame:
    """IDF-weighted overlap / union mass for list column ``col`` (per pair row ``pid``)."""
    def mass(expr_df: pl.DataFrame, alias: str) -> pl.DataFrame:
        return (expr_df.explode("tok").drop_nulls("tok")
                .join(idf, on=["country", "tok"], how="left")
                .group_by("pid").agg(pl.col("idf").fill_null(12.0).sum().alias(alias)))
    base = df.select("pid", "country", pl.col(col + "_1").alias("a"), pl.col(col + "_2").alias("b"))
    inter = mass(base.select("pid", "country", pl.col("a").list.set_intersection("b").alias("tok")), "i")
    uni = mass(base.select("pid", "country", pl.col("a").list.set_union("b").alias("tok")), "u")
    m1 = mass(base.select("pid", "country", pl.col("a").list.unique().alias("tok")), "m1")
    m2 = mass(base.select("pid", "country", pl.col("b").list.unique().alias("tok")), "m2")
    r = (df.select("pid").join(inter, on="pid", how="left").join(uni, on="pid", how="left")
           .join(m1, on="pid", how="left").join(m2, on="pid", how="left")
           .with_columns(pl.col("i", "u", "m1", "m2").fill_null(0.0)))  # never touch the u32 key
    return r.select(
        "pid",
        pl.col("i").alias(f"{name}_widf"),
        (pl.col("i") / pl.col("u")).fill_nan(0).alias(f"{name}_wjac"),
        (pl.col("i") / pl.col("m1")).fill_nan(0).alias(f"{name}_wc12"),
        (pl.col("i") / pl.col("m2")).fill_nan(0).alias(f"{name}_wc21"),
        pl.col("m1").alias(f"{name}_mass1"),
    )


def pair_features(pairs: pl.DataFrame, s1: pl.DataFrame, tg: pl.DataFrame,
                  idf_core: pl.DataFrame, idf_addr: pl.DataFrame,
                  name_freq1: pl.DataFrame, name_freq2: pl.DataFrame) -> pl.DataFrame:
    """pairs: s1_idx, t_idx, src + blocking cols. s1/tg: normalised record tables keyed by idx
    (tg has a ``src`` column). Returns pairs with feature columns appended."""
    extra = [c for c in EXTRA_COLS if c in s1.columns and c in tg.columns]
    cols = REC_COLS + extra
    df = (pairs.with_row_index("pid")
          .join(s1.select(pl.col("idx").alias("s1_idx"), *cols), on="s1_idx", how="left")
          .join(tg.select(pl.col("idx").alias("t_idx"), "src", *[c for c in cols if c != "country"]),
                on=["t_idx", "src"], how="left", suffix="_2")
          .rename({c: c + "_1" for c in cols if c != "country"}))
    s = lambda c: df[c].fill_null("").to_list()
    j = lambda c: df[c].list.join(" ").fill_null("").to_list()
    n1, n2 = s("name_norm_1"), s("name_norm_2")
    c1, c2 = j("core_1"), j("core_2")
    k1, k2 = j("skel_1"), j("skel_2")
    a1, a2 = s("addr_norm_1"), s("addr_norm_2")
    h1, h2 = s("hno_1"), s("hno_2")
    f = {
        "n_ratio": _cp(fuzz.ratio, n1, n2), "n_pratio": _cp(fuzz.partial_ratio, n1, n2),
        "n_tsort": _cp(fuzz.token_sort_ratio, n1, n2), "n_tset": _cp(fuzz.token_set_ratio, n1, n2),
        "n_wratio": _cp(fuzz.WRatio, n1, n2), "n_jw": _cp(JaroWinkler.normalized_similarity, n1, n2),
        "c_ratio": _cp(fuzz.ratio, c1, c2), "c_pratio": _cp(fuzz.partial_ratio, c1, c2),
        "c_tsort": _cp(fuzz.token_sort_ratio, c1, c2), "c_tset": _cp(fuzz.token_set_ratio, c1, c2),
        "c_jw": _cp(JaroWinkler.normalized_similarity, c1, c2),
        "c_lev": _cp(Levenshtein.normalized_similarity, c1, c2),
        "k_ratio": _cp(fuzz.ratio, k1, k2), "k_tset": _cp(fuzz.token_set_ratio, k1, k2),
        "a_ratio": _cp(fuzz.ratio, a1, a2), "a_pratio": _cp(fuzz.partial_ratio, a1, a2),
        "a_tset": _cp(fuzz.token_set_ratio, a1, a2), "a_tsort": _cp(fuzz.token_sort_ratio, a1, a2),
        "h_lev": _cp(Levenshtein.distance, h1, h2),
    }
    if "core_a" in extra:  # alias-mapped, filler-free name (er.auxfit)
        ca1, ca2 = j("core_a_1"), j("core_a_2")
        f.update({"ca_ratio": _cp(fuzz.ratio, ca1, ca2), "ca_tset": _cp(fuzz.token_set_ratio, ca1, ca2),
                  "ca_tsort": _cp(fuzz.token_sort_ratio, ca1, ca2)})
    if "atoks_a" in extra:
        aa1, aa2 = j("atoks_a_1"), j("atoks_a_2")
        f.update({"aa_tset": _cp(fuzz.token_set_ratio, aa1, aa2)})
    df = df.with_columns([pl.Series(k, v) for k, v in f.items()])
    for c, nm in (("core_a", "ca"), ("atoks_a", "aa")):
        if c in extra:
            df = df.with_columns(*_set_feats(df, c, nm))
    first1 = df["core_1"].list.first().fill_null("").to_list()
    first2 = df["core_2"].list.first().fill_null("").to_list()
    df = df.with_columns(pl.Series("first_jw", _cp(JaroWinkler.normalized_similarity, first1, first2)))

    df = df.with_columns(
        *_set_feats(df, "core", "core"), *_set_feats(df, "skel", "skel"),
        *_set_feats(df, "atoks", "at"), *_set_feats(df, "nums", "num"),
        *_set_feats(df, "loc", "loc"), *_set_feats(df, "street", "str"),
        (pl.col("hno_1") != "").cast(pl.Int8).alias("h_has1"),
        (pl.col("hno_2") != "").cast(pl.Int8).alias("h_has2"),
        ((pl.col("hno_1") == pl.col("hno_2")) & (pl.col("hno_1") != "")).cast(pl.Int8).alias("h_eq"),
        (pl.col("hno_1").cast(pl.Float64, strict=False) - pl.col("hno_2").cast(pl.Float64, strict=False))
            .abs().log1p().cast(pl.Float32).alias("h_logdiff"),
        pl.when((pl.col("state_1") == "") | (pl.col("state_2") == "")).then(-1)
          .otherwise((pl.col("state_1") == pl.col("state_2")).cast(pl.Int8)).alias("state_eq"),
        (pl.col("addr_norm_2") == "").cast(pl.Int8).alias("a_missing2"),
        pl.when((pl.col("legal_1").list.len() == 0) | (pl.col("legal_2").list.len() == 0)).then(-1)
          .otherwise(pl.col("legal_1").list.set_intersection("legal_2").list.len().cast(pl.Int32))
          .alias("legal_inter"),
        pl.col("legal_1").list.len().alias("legal_n1"), pl.col("legal_2").list.len().alias("legal_n2"),
        pl.col("name_norm_1").str.len_chars().alias("n_len1"),
        pl.col("name_norm_2").str.len_chars().alias("n_len2"),
    )
    df = (df.join(_idf_overlap(df, "core", idf_core, "core"), on="pid", how="left")
            .join(_idf_overlap(df, "atoks", idf_addr, "at"), on="pid", how="left")
            .join(name_freq1, on="name_norm_1", how="left")
            .join(name_freq2, on="name_norm_2", how="left"))
    drop = [c for c in df.columns if c.endswith(("_1", "_2")) and c.split("_")[0] in
            {"name", "core", "legal", "skel", "addr", "atoks", "nums", "hno", "state", "loc", "street"}]
    return df.drop(drop + ["pid"])


def context_features(df: pl.DataFrame, score: str = "bs") -> pl.DataFrame:
    """Within-S1 and competition (within-target) features over a candidate set."""
    return df.with_columns(
        pl.len().over("s1_idx").alias("cx_n"),
        pl.len().over("s1_idx", "src").alias("cx_n_src"),
        (pl.col(score) / pl.col(score).max().over("s1_idx", "src")).alias("cx_rel"),
        pl.col(score).rank("ordinal", descending=True).over("s1_idx", "src").alias("cx_rank"),
        pl.col("n_tset").rank("ordinal", descending=True).over("s1_idx", "src").alias("cx_rank_ntset"),
        pl.col("a_tset").rank("ordinal", descending=True).over("s1_idx", "src").alias("cx_rank_atset"),
        (pl.col("n_tset") - pl.col("n_tset").max().over("s1_idx", "src")).alias("cx_gap_ntset"),
        (pl.col("a_tset") - pl.col("a_tset").max().over("s1_idx", "src")).alias("cx_gap_atset"),
        pl.len().over("src", "t_idx").alias("tx_n"),
        pl.col(score).rank("ordinal", descending=True).over("src", "t_idx").alias("tx_rank"),
        (pl.col(score) - pl.col(score).max().over("src", "t_idx")).alias("tx_gap"),
        (pl.col("n_tset") - pl.col("n_tset").max().over("src", "t_idx")).alias("tx_gap_ntset"),
        (pl.col("a_tset") - pl.col("a_tset").max().over("src", "t_idx")).alias("tx_gap_atset"),
    )


def coherence_features(d: pl.DataFrame, tabs: dict[str, pl.DataFrame], anchors: pl.DataFrame
                       ) -> pl.DataFrame:
    """Cluster coherence (collective ER, README N3).

    anchors: per (s1_idx, src) the top stage-1 candidate (a_idx, a_p1). For each pair we compare
    the candidate with the S1's anchor in the *other* source and with its *own-source* anchor
    (S2/S3 duplicates of one business resemble each other). Self-comparisons are masked (-1).
    """
    strs = pl.concat([t.select(pl.col("idx").alias("x_idx"), pl.lit(int(s), pl.UInt8).alias("x_src"),
                               pl.col("core").list.join(" ").alias("xc"), pl.col("addr_norm").fill_null("").alias("xa"))
                      for s, t in tabs.items() if s in ("2", "3")])
    out = d.select("s1_idx", "src", "t_idx").with_row_index("rid")
    me = out.join(strs, left_on=["t_idx", "src"], right_on=["x_idx", "x_src"], how="left")
    feats = {}
    for tag, src_expr in (("same", pl.col("src")), ("oth", (5 - pl.col("src").cast(pl.Int16)).cast(pl.UInt8))):
        a = (out.with_columns(src_expr.alias("a_src"))
                .join(anchors, left_on=["s1_idx", "a_src"], right_on=["s1_idx", "src"], how="left")
                .join(strs, left_on=["a_idx", "a_src"], right_on=["x_idx", "x_src"], how="left").sort("rid"))
        c = _cp(fuzz.token_set_ratio, me.sort("rid")["xc"].fill_null("").to_list(), a["xc"].fill_null("").to_list())
        ad = _cp(fuzz.token_set_ratio, me.sort("rid")["xa"].fill_null("").to_list(), a["xa"].fill_null("").to_list())
        miss = a["a_idx"].is_null().to_numpy()
        if tag == "same":
            miss |= (a["a_idx"] == a["t_idx"]).fill_null(False).to_numpy()
        c[miss], ad[miss] = -1, -1
        feats[f"coh_c_{tag}"], feats[f"coh_a_{tag}"] = c, ad
        feats[f"coh_p_{tag}"] = a["a_p1"].fill_null(-1).to_numpy().astype(np.float32)
    return d.with_columns([pl.Series(k, v) for k, v in feats.items()])


def name_freq(tbl: pl.DataFrame, suffix: str) -> pl.DataFrame:
    return (tbl.group_by("name_norm").len(f"nfreq{suffix}")
               .rename({"name_norm": f"name_norm{suffix}"}))
