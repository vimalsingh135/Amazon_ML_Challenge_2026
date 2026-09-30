"""Alias mining (README N4) and name-structure repair, learned only from the provided data.

- split_alias_markers: S3 contains about 105k names of the form "<pseudo-word> d/b/a <real name>"
  (also f/k/a, formerly:, DBA:, FKA). The identity-bearing name is the part after the marker.
- mine_token_aliases: target token -> S1 token, e.g. transliteration artifacts and systematic
  misspellings ("kansalting" -> "consulting", "calcutta" -> "kolkata"). Tokens are aligned inside
  each positive pair by phonetic skeleton and Jaro-Winkler, then kept only if the mapping is
  frequent and pure across pairs.
- mine_part_aliases: whole comma-part aliases in addresses (native-script / abbreviated states,
  city variants) by within-pair co-occurrence purity. No similarity is needed, so it also works
  when the transliteration is far from the English spelling ("tamizhnatu" -> "tamil nadu").
- segment: DP word-break for glued names ("meghanagold" -> ["meghana", "gold"]).
"""
from __future__ import annotations

import math
import re

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from .text import skeleton, to_ascii

# ---------------------------------------------------------------------------
# (4) alias markers
# ---------------------------------------------------------------------------
_MARKER = re.compile(
    r"\s*(?:\b[df]\s*/\s*[bk]\s*/\s*a\b|\bformerly(?:\s+known\s+as)?\b|\bdoing\s+business\s+as\b"
    r"|\balso\s+known\s+as\b|\btrading\s+as\b|\bdba\b|\bfka\b)\s*:?\s*",
    re.IGNORECASE)
_EDGE = re.compile(r"^[\s\-:;,.(\[]+|[\s\-:;,.)\]]+$")


def split_alias_markers(name_ascii: str | None) -> tuple[str, str | None]:
    """Split "<main> d/b/a <alias>" into (main, alias). The alias is the real business name.

    A split happens only when both sides are non-empty, so real names that *start* with the
    letters ("DBA Organisers Pvt Ltd", "Aka Technologies") are left intact. "aka" is never a
    marker: in this data it only occurs as a genuine name token.
    """
    s = name_ascii or ""
    m = _MARKER.search(s)
    if not m:
        return s, None
    main, alias = _EDGE.sub("", s[:m.start()]), _EDGE.sub("", s[m.end():])
    if not main or not alias:
        return s, None
    return main, alias


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def norm_part(part: str | None) -> str:
    return _NON_ALNUM.sub(" ", to_ascii(part or "")).strip()


def _skel_map(tokens) -> dict[str, str]:
    return {t: skeleton(t) for t in tokens}


def _purity_filter(pairs: pl.DataFrame, key: str, val: str, min_count: int, min_purity: float,
                   total: pl.DataFrame | None = None) -> dict[str, str]:
    """pairs: one row per aligned (key, val) occurrence. Keep the best val per key if it is
    frequent and pure. Purity = best count / all occurrences of key (``total`` if given)."""
    c = pairs.group_by(key, val).len("n")
    best = c.sort([key, "n"], descending=[False, True]).group_by(key, maintain_order=True).first()
    tot = total if total is not None else c.group_by(key).agg(pl.col("n").sum().alias("tot"))
    best = best.join(tot, on=key).filter((pl.col("n") >= min_count) &
                                         (pl.col("n") / pl.col("tot") >= min_purity))
    return dict(zip(best[key].to_list(), best[val].to_list()))


# ---------------------------------------------------------------------------
# (1) token aliases
# ---------------------------------------------------------------------------
def mine_token_aliases(pos_s1_core: pl.Series, pos_tgt_core: pl.Series, min_count: int = 5,
                       min_purity: float = 0.6, s1_vocab_freq: dict[str, int] | None = None,
                       max_vocab_freq: int = 2, min_sim: float = 0.80) -> dict[str, str]:
    """Map target tokens that are rare or absent in S1 to their aligned S1 token.

    Alignment within each positive pair: a target token t (not in the pair's S1 token set, and
    with S1-vocabulary frequency <= max_vocab_freq) is paired with the S1 token s of that pair
    (not present in the target) that maximises sim = max(JW(t, s), ratio(skel t, skel s)/100).
    It is kept if sim >= min_sim, or if the skeletons are equal. Across pairs, t -> s must
    occur at least min_count times and with purity >= min_purity, measured over *all*
    occurrences of t as an unmatched token.
    """
    if pos_s1_core.len() != pos_tgt_core.len():
        raise ValueError("inputs must be aligned")
    L = pl.List(pl.Utf8)
    df = pl.DataFrame({"a": pos_s1_core.cast(L), "b": pos_tgt_core.cast(L)}).with_row_index("pid")
    df = df.with_columns(pl.col("a").fill_null(pl.lit([], dtype=L)), pl.col("b").fill_null(pl.lit([], dtype=L)))
    if s1_vocab_freq is None:
        vc = df.select(pl.col("a").explode().alias("t")).drop_nulls().group_by("t").len("f")
    else:
        vc = pl.DataFrame({"t": list(s1_vocab_freq), "f": list(s1_vocab_freq.values())})
    ut = (df.select("pid", pl.col("b").list.set_difference("a").alias("t")).explode("t")
            .drop_nulls().filter(pl.col("t").str.len_chars() >= 3)
            .join(vc, on="t", how="left").filter(pl.col("f").fill_null(0) <= max_vocab_freq).drop("f"))
    us = (df.select("pid", pl.col("a").list.set_difference("b").alias("s")).explode("s")
            .drop_nulls().filter(pl.col("s").str.len_chars() >= 2))
    if ut.height == 0 or us.height == 0:
        return {}
    cand = ut.join(us, on="pid")
    sk = _skel_map(set(cand["t"].to_list()) | set(cand["s"].to_list()))
    t_l, s_l = cand["t"].to_list(), cand["s"].to_list()
    jw = process.cpdist(t_l, s_l, scorer=JaroWinkler.normalized_similarity, workers=-1, dtype=np.float32)
    ks = process.cpdist([sk[t] for t in t_l], [sk[s] for s in s_l], scorer=fuzz.ratio, workers=-1,
                        dtype=np.float32) / 100.0
    eq = np.array([sk[t] == sk[s] and len(sk[t]) >= 2 for t, s in zip(t_l, s_l)])
    cand = cand.with_columns(pl.Series("sim", np.maximum(jw, ks)), pl.Series("eq", eq))
    cand = cand.filter((pl.col("sim") >= min_sim) | pl.col("eq"))
    aligned = (cand.sort("sim", descending=True).group_by("pid", "t", maintain_order=True).first()
                   .select("t", "s"))
    total = ut.group_by("t").len("tot")
    return _purity_filter(aligned, "t", "s", min_count, min_purity, total)


# ---------------------------------------------------------------------------
# (2) comma-part aliases
# ---------------------------------------------------------------------------
def mine_part_aliases(pos_s1_addr_raw: pl.Series, pos_tgt_addr_raw: pl.Series, min_count: int = 5,
                      min_purity: float = 0.6, max_part_tokens: int = 4) -> dict[str, str]:
    """Map a normalised target comma-part to the S1 part it stands for.

    Only digit-free parts with at most max_part_tokens tokens are considered (states, cities,
    districts). For each target part p absent from the pair's S1 parts, every S1 part absent
    from the target is a co-occurrence candidate. p -> q is kept when q co-occurs with p in at
    least min_count pairs and in >= min_purity of p's occurrences. Street-like and random parts
    never reach that purity, while a consistent alias (a state in native script) does.
    """
    if pos_s1_addr_raw.len() != pos_tgt_addr_raw.len():
        raise ValueError("inputs must be aligned")
    def parts(s: pl.Series) -> pl.Series:
        return pl.Series([[q for q in dict.fromkeys(norm_part(x) for x in (v or "").split(","))
                           if q and not any(c.isdigit() for c in q) and len(q.split()) <= max_part_tokens]
                          for v in s.to_list()], dtype=pl.List(pl.Utf8))
    df = pl.DataFrame({"a": parts(pos_s1_addr_raw), "b": parts(pos_tgt_addr_raw)}).with_row_index("pid")
    up = df.select("pid", pl.col("b").list.set_difference("a").alias("p")).explode("p").drop_nulls()
    uq = df.select("pid", pl.col("a").list.set_difference("b").alias("q")).explode("q").drop_nulls()
    if up.height == 0 or uq.height == 0:
        return {}
    co = up.join(uq, on="pid").select("p", "q")
    total = up.group_by("p").len("tot")
    return _purity_filter(co, "p", "q", min_count, min_purity, total)


# ---------------------------------------------------------------------------
# (3) word segmentation
# ---------------------------------------------------------------------------
ALWAYS_OK = {"com", "net", "org", "in", "co", "fr", "llc", "inc", "ltd", "pvt", "llp", "eurl",
             "sarl", "sas", "sasu", "sci", "corp", "pc", "sa"}


def segment(token: str, vocab_freq: dict[str, int], min_freq: int = 3, max_word: int = 20,
            max_unknown: int = 3, min_known_cover: float = 0.75, self_freq: int = 20) -> list[str]:
    """Minimum-cost word break. A known word costs log(N / freq). An unknown piece of length
    <= max_unknown (initials such as "bs", "d2o") costs 12 + 4*len. The split is returned only
    if it has >= 2 pieces, known words cover >= min_known_cover of the characters, and the token
    is not itself a common word (freq >= self_freq). Otherwise the result is [token]."""
    n = len(token)
    if n < 6 or vocab_freq.get(token, 0) >= self_freq:
        return [token]
    total = float(sum(vocab_freq.values()) or 1)
    def cost(w: str) -> tuple[float, bool] | None:
        if w in ALWAYS_OK:
            return 6.0, True
        f = vocab_freq.get(w, 0)
        if f >= min_freq and len(w) >= 2:
            return math.log(total / f), True
        if len(w) <= max_unknown:
            return 12.0 + 4.0 * len(w), False
        return None
    best: list[tuple[float, int, int] | None] = [None] * (n + 1)   # (cost, prev, known_chars)
    best[0] = (0.0, -1, 0)
    for j in range(1, n + 1):
        for i in range(max(0, j - max_word), j):
            if best[i] is None:
                continue
            c = cost(token[i:j])
            if c is None:
                continue
            cand = (best[i][0] + c[0], i, best[i][2] + (j - i if c[1] else 0))
            if best[j] is None or cand[0] < best[j][0]:
                best[j] = cand
    if best[n] is None:
        return [token]
    out, j = [], n
    while j > 0:
        i = best[j][1]
        out.append(token[i:j])
        j = i
    out.reverse()
    if len(out) < 2 or best[n][2] / n < min_known_cover:
        return [token]
    return out
