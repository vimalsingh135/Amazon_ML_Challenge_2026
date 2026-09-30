"""Candidate generation: IDF-weighted multi-key blocking, run per country and per target source.

Each record emits hashed blocking keys from several families. Name families: core
tokens, phonetic skeletons, token pairs, name x locality, prefix4 x locality. Address
families: house-number x street, house-number x locality, street x locality. A pair's
score is the sum of IDF weights of the keys it shares. Keys that are too frequent in
the target source are dropped (document-frequency cap). The name and address families
give independent retrieval paths, so name swaps are still recovered through the address.
"""
from __future__ import annotations

import logging
import math
import time

import polars as pl

log = logging.getLogger(__name__)

NAME_FAMS = ("n", "s", "np", "sp", "nl", "sl", "nx", "fn", "fs")
ADDR_FAMS = ("a1", "a2", "a4", "aa", "h0", "h1")
PURE_NAME = ("n", "s", "np", "sp", "fn", "fs")  # name-only evidence (usable when the address is missing)
SINGLE = ("n", "s", "h0")                 # single-token keys get a tighter frequency cap
NUM_FAMS = ("a1", "a2", "a4", "h0", "h1")  # house-number evidence (own candidate budget)
FULL_NAME_CAP = 2000
# coarse phonetic classes for the whole-name skeleton key (Soundex-style): sound-alike consonants
# that transliteration swaps (s/z/j, b/v/f/p, t/d, m/n) collapse to one symbol
_SOUNDEX_FROM = ["b", "f", "p", "v", "c", "g", "j", "k", "q", "s", "x", "z", "d", "t", "m", "n"]
_SOUNDEX_TO = ["1", "1", "1", "1", "2", "2", "2", "2", "2", "2", "2", "2", "3", "3", "5", "5"]                      # whole-name key: exact names with a missing address
# alphanumeric house ids ("C-346", "A/98", "19-B", "B0702") that the word tokenizer splits apart
_HID_JOIN1 = r"\b([a-z])\s?[-/]?\s?(\d{1,5})\b"
_HID_JOIN2 = r"\b(\d{1,5})\s?[-/]\s?([a-z])\b"
_HID = r"\b[a-z]\d{1,5}\b|\b\d{1,5}[a-z]\b"
FAM_ID = {f: i for i, f in enumerate(NAME_FAMS + ADDR_FAMS)}
BIT = {i: 1 << i for i in FAM_ID.values()}


def _tok(df: pl.DataFrame, col: str, limit: int, min_len: int = 2, prefix: int = 0) -> pl.DataFrame:
    """(idx, h): u64 hashes of the first ``limit`` distinct tokens of list column ``col``.

    Tokens are hashed immediately so compound keys are built from integers, never from
    concatenated strings (keeps key construction for ~5M-row tables within a few GB)."""
    e = (df.select("idx", pl.col(col).list.unique(maintain_order=True).list.head(limit).alias("t"))
           .explode("t").drop_nulls("t").filter(pl.col("t").str.len_chars() >= max(min_len, prefix)))
    if prefix:
        e = e.with_columns(pl.col("t").str.slice(0, prefix)).unique()
    return e.select("idx", pl.col("t").hash(seed=len(col) * 31 + prefix).alias("h"))


def _single(a: pl.DataFrame, fam: str) -> pl.DataFrame:
    return a.select("idx", pl.col("h").hash(seed=FAM_ID[fam]).alias("key"),
                    pl.lit(FAM_ID[fam], pl.UInt8).alias("fam"))


def _pairs(a: pl.DataFrame, b: pl.DataFrame, fam: str, ordered: bool = False) -> pl.DataFrame:
    j = a.join(b, on="idx", suffix="_b")
    if ordered:
        j = j.filter(pl.col("h") < pl.col("h_b"))
    return j.select("idx", pl.struct("h", "h_b").hash(seed=100 + FAM_ID[fam]).alias("key"),
                    pl.lit(FAM_ID[fam], pl.UInt8).alias("fam"))


def iter_key_families(df: pl.DataFrame):
    """Yield (family, keys) one family at a time: (idx, key u64, fam u8) frames."""
    core = _tok(df, "core", 4)
    skel = _tok(df, "skel", 4, 2)
    skel3 = _tok(df, "skel", 4, 3)
    loc = _tok(df, "loc", 2)
    street = _tok(df, "street", 3, 3)
    nums = _tok(df, "nums", 3, 1)
    pre = _tok(df, "core", 4, prefix=4)
    # street and locality tokens pooled: classification differs when a record has no number
    addr = (df.select("idx", pl.concat_list(pl.col("street").list.head(3), pl.col("loc").list.head(3))
                      .list.unique(maintain_order=True).list.head(5).alias("aa"))
              .pipe(_tok, "aa", 5, 3))
    if "business_address" in df.columns:
        hid = (df.select("idx", pl.col("business_address").fill_null("").str.to_lowercase()
                         .str.replace_all(_HID_JOIN1, "$1$2").str.replace_all(_HID_JOIN2, "$1$2")
                         .str.extract_all(_HID).list.eval(pl.element().str.replace(r"^([a-z]+)0+(\d)", "$1$2"))
                         .list.unique().list.head(3).alias("t"))
                 .explode("t").drop_nulls("t")
                 .select("idx", pl.col("t").hash(seed=977).alias("h")))
        yield "h0", _single(hid, "h0")
        yield "h1", _pairs(hid, loc, "h1")
    full = (df.select("idx", pl.col("core").list.sort().list.join(" ").alias("t"))
              .filter(pl.col("t").str.len_chars() >= 3).select("idx", pl.col("t").hash(seed=991).alias("h")))
    yield "fn", _single(full, "fn")
    # whole-name phonetic skeleton: transliterations / vowel typos ("siti midiya" ~ "city media" -> "md st")
    fskel = (df.select("idx", pl.col("skel").list.eval(
                    pl.element().filter(pl.element().str.len_chars() >= 2)
                    .str.replace_many(_SOUNDEX_FROM, _SOUNDEX_TO))
                              .list.unique().list.sort().list.join(" ").alias("t"))
               .filter(pl.col("t").str.len_chars() >= 4).select("idx", pl.col("t").hash(seed=997).alias("h")))
    yield "fs", _single(fskel, "fs")
    yield "n", _single(core, "n")
    yield "s", _single(skel3, "s")
    yield "np", _pairs(core, core, "np", ordered=True)
    yield "sp", _pairs(skel, skel, "sp", ordered=True)
    yield "nl", _pairs(core, loc, "nl")
    yield "sl", _pairs(skel3, loc, "sl")
    yield "nx", _pairs(pre, loc, "nx")
    yield "a1", _pairs(nums, street, "a1")
    yield "a2", _pairs(nums, loc, "a2")
    yield "a4", _pairs(nums, nums, "a4", ordered=True)
    yield "aa", _pairs(addr, addr, "aa", ordered=True)


def make_keys(df: pl.DataFrame) -> pl.DataFrame:
    """df needs idx, core, skel, loc, street, nums. Returns (idx, key u64, fam u8)."""
    return pl.concat([k for _, k in iter_key_families(df)])


def _weighted_target_keys(tgt: pl.DataFrame, s1_keys: pl.DataFrame, n_t: int, cap_single: int,
                          cap_pair: int) -> pl.DataFrame:
    """Target keys with IDF weights, built one family at a time (bounded memory). Document
    frequencies are computed over *all* targets before pruning; keys absent from S1 are then
    dropped because they can never produce a candidate pair."""
    out = []
    for fam, part in iter_key_families(tgt):
        cap = cap_single if fam in SINGLE else (FULL_NAME_CAP if fam in ("fn", "fs") else cap_pair)
        dfk = (part.group_by("key").len("df").filter(pl.col("df") <= cap)
                   .join(s1_keys.filter(pl.col("fam") == FAM_ID[fam]).select("key"), on="key", how="semi"))
        dfk = dfk.with_columns((math.log(n_t + 1) - pl.col("df").cast(pl.Float32).log()).alias("w"),
                               (1.0 / pl.col("df").cast(pl.Float32)).alias("inv_df"))
        out.append(part.join(dfk.select("key", "w", "inv_df"), on="key"))
        del part, dfk
    return pl.concat(out)


def block(s1: pl.DataFrame, tgt: pl.DataFrame, cap_single: int = 300, cap_pair: int = 1500,
          top_k: int = 50, top_name: int = 20, top_pure: int = 15, top_addr: int = 25, top_num: int = 15,
          chunk: int = 50_000, join_budget: int = 20_000_000, on_chunk=None) -> pl.DataFrame | None:
    """Candidate pairs (s1_idx, t_idx, bs, bs_name, bs_addr, nkeys, nfam, brank, ...).

    Keeps the union of the top_k by total score and the top_name / top_addr by the name-only
    and address-only scores, so name-only matches (null target address) and name-swap
    matches (address only) never compete with pairs supported by both.

    Memory: if ``on_chunk(g, chunk_id)`` is given, each S1 chunk's candidates are handed to it
    (e.g. written to disk) instead of being accumulated, and None is returned. A chunk always
    holds *all* candidates of its S1 records, so per-S1 window features are exact inside it."""
    s1k = make_keys(s1)
    tk = _weighted_target_keys(tgt, s1k.select("key", "fam").unique(), max(tgt.height, 1),
                               cap_single, cap_pair)
    # cost-aware chunking: an S1's join cost is the sum of the target frequencies of its keys, so
    # each chunk is capped at ``join_budget`` joined rows (and ``chunk`` S1) whatever the names are
    kdf = tk.group_by("key", "fam").len("df")
    s1k = s1k.join(kdf.select("key", "fam"), on=["key", "fam"], how="semi")  # keys that can match
    cost = (s1k.join(kdf, on=["key", "fam"]).group_by("idx").agg(pl.col("df").sum().alias("c"))
               .sort("idx").with_row_index("pos"))
    cost = cost.with_columns(pl.max_horizontal(pl.col("c").cum_sum() // join_budget,
                                               pl.col("pos") // chunk).alias("ch"))
    # make chunk ids monotone and dense
    cost = cost.with_columns(pl.col("ch").cum_max().rank("dense").cast(pl.UInt32).alias("ch"))
    s1k = s1k.join(cost.select("idx", "ch"), on="idx")
    del kdf, cost
    a_ids, p_ids = [FAM_ID[f] for f in ADDR_FAMS], [FAM_ID[f] for f in PURE_NAME]
    n_ids = [FAM_ID[f] for f in NUM_FAMS]
    out, n_pairs = [], 0
    for (ch,), sub in s1k.group_by("ch"):
        t0 = time.time()
        j = (sub.drop("ch").rename({"idx": "s1_idx"}).join(tk.rename({"idx": "t_idx"}), on=["key", "fam"])
                .with_columns(pl.col("fam").is_in(a_ids).cast(pl.Float32).alias("a"),
                              pl.col("fam").is_in(p_ids).cast(pl.Float32).alias("p"),
                              pl.col("fam").is_in(n_ids).cast(pl.Float32).alias("q"),
                              pl.col("fam").replace_strict(BIT, return_dtype=pl.UInt16).alias("bit")))
        g = j.group_by("s1_idx", "t_idx").agg(
            pl.col("w").sum().alias("bs"),
            (pl.col("w") * (1 - pl.col("a"))).sum().alias("bs_name"),
            (pl.col("w") * pl.col("a")).sum().alias("bs_addr"),
            (pl.col("w") * pl.col("p")).sum().alias("bs_pure"),
            (pl.col("w") * pl.col("q")).sum().alias("bs_num"),
            pl.len().alias("nkeys"),
            # supervised meta-blocking edge weights (GSM, VLDB'22): ARCS and per-family evidence
            pl.col("inv_df").sum().alias("arcs"),
            pl.col("a").sum().alias("nk_addr"),
            pl.col("bit").bitwise_or().alias("fam_mask"),
        ).with_columns(pl.col("fam_mask").bitwise_count_ones().alias("nfam"),
                       (pl.col("nkeys") - pl.col("nk_addr")).alias("nk_name"))
        j_rows = j.height
        del j
        g = (g.with_columns(
                pl.col("bs").rank("ordinal", descending=True).over("s1_idx").alias("brank"),
                pl.col("bs_name").rank("ordinal", descending=True).over("s1_idx").alias("brank_n"),
                pl.col("bs_addr").rank("ordinal", descending=True).over("s1_idx").alias("brank_a"),
                pl.col("bs_pure").rank("ordinal", descending=True).over("s1_idx").alias("brank_p"),
                pl.col("bs_num").rank("ordinal", descending=True).over("s1_idx").alias("brank_q"))
              .filter((pl.col("brank") <= top_k) | ((pl.col("brank_n") <= top_name) & (pl.col("bs_name") > 0))
                      | ((pl.col("brank_a") <= top_addr) & (pl.col("bs_addr") > 0))
                      | ((pl.col("brank_p") <= top_pure) & (pl.col("bs_pure") > 0))
                      | ((pl.col("brank_q") <= top_num) & (pl.col("bs_num") > 0))))
        n_pairs += g.height
        log.debug("chunk %s: join=%d pairs=%d %.1fs", ch, j_rows, g.height, time.time() - t0)
        if on_chunk is None:
            out.append(g)
        else:
            on_chunk(g, int(ch))
    log.info("block s1=%d tgt=%d pairs=%d", s1.height, tgt.height, n_pairs)
    return (pl.concat(out) if out else pl.DataFrame()) if on_chunk is None else None
