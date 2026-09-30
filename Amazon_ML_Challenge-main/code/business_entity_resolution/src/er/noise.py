"""Noise-generator inversion (research note R1a + R1b).

The corruptions in this dataset are synthetic and come from a finite set of operators. Two
of them are learned here from training positive pairs, using only the provided data:

R1a  filler tokens: words the generator *adds* to a target name ("services", "enterprises",
     transliterated legal forms such as "praivet"/"limitet"/"pra li", ...). For each
     token, ``rate = P(token absent from the S1 name | token present in the target name)``
     over positive pairs. Callers down-weight high-rate tokens in IDF overlap and blocking.

R1b  name replacement: the generator sometimes replaces the whole name with a pronounceable
     pseudo-word assembled from a syllable inventory ("Ciraecto", "Zephevozeph",
     "Orbisyndelta"). A hashed char-n-gram logistic classifier separates these from ordinary
     target names (clean, typo'd, transliterated, domains). Its score tells the matcher that
     name dissimilarity is uninformative, so it should trust the address.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# R1a — filler tokens
# ---------------------------------------------------------------------------
def mine_filler_tokens(s1_core: pl.Series, tgt_core: pl.Series, min_count: int = 30,
                       prior: float = 1.0) -> dict[str, float]:
    """Per-token "added by noise" rate from aligned positive pairs.

    rate(t) = (#pairs where t is in the target but not in S1 + prior*0.5)
              / (#pairs where t is in the target + prior)
    Only tokens seen in at least ``min_count`` target names are returned. Rare tokens are
    mostly typos, and IDF already handles those.
    """
    if s1_core.len() != tgt_core.len():
        raise ValueError("s1_core and tgt_core must be aligned (same length)")
    df = pl.DataFrame({"a": s1_core, "b": tgt_core}).with_columns(
        pl.col("a").fill_null(pl.lit([], dtype=pl.List(pl.Utf8))),
        pl.col("b").fill_null(pl.lit([], dtype=pl.List(pl.Utf8))))
    present = df.select(pl.col("b").list.unique().alias("t")).explode("t").drop_nulls()
    added = df.select(pl.col("b").list.set_difference("a").alias("t")).explode("t").drop_nulls()
    cnt = present.group_by("t").len("n").join(added.group_by("t").len("k"), on="t", how="left")
    cnt = cnt.fill_null(0).filter(pl.col("n") >= min_count)
    cnt = cnt.with_columns(((pl.col("k") + prior * 0.5) / (pl.col("n") + prior)).alias("rate"))
    return dict(zip(cnt["t"].to_list(), cnt["rate"].to_list()))


def split_filler(rates: dict[str, float], s1_vocab: set[str], threshold: float = 0.8
                 ) -> tuple[set[str], set[str]]:
    """Split high-rate tokens into (filler, foreign).

    filler : words that also occur in clean S1 names ("services", "center", "formerly"). They
             are true noise insertions, so down-weight them.
    foreign: tokens never seen in S1 names, mostly transliteration artifacts ("kanstrakshan",
             "praivet"). They carry identity, so map them via skeleton/alias matching rather
             than dropping them.
    """
    hi = {t for t, r in rates.items() if r >= threshold}
    return {t for t in hi if t in s1_vocab}, {t for t in hi if t not in s1_vocab}


# ---------------------------------------------------------------------------
# R1b — whole-name replacement detector
# ---------------------------------------------------------------------------
_EMPTY = pl.lit([], dtype=pl.List(pl.Utf8))
_HASH = dict(analyzer="char_wb", ngram_range=(2, 4), n_features=2 ** 20, alternate_sign=False,
             norm="l2", lowercase=False, dtype=np.float32)


@dataclass
class NameReplacementModel:
    vocab: set[str]
    clf: LogisticRegression | None = None
    meta: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "NameReplacementModel":
        with open(path, "rb") as f:
            return pickle.load(f)


def _dense(core: pl.Series, legal: pl.Series, vocab: pl.Series) -> tuple[list[str], np.ndarray]:
    """Text for the char model plus dense features [ntok, oov_frac, has_legal, has_digit, len/20]."""
    df = pl.DataFrame({"c": core, "l": legal}).with_columns(
        pl.col("c").fill_null(_EMPTY), pl.col("l").fill_null(_EMPTY)).with_row_index("i")
    oov = (df.select("i", "c").explode("c").drop_nulls("c")
             .with_columns((~pl.col("c").is_in(vocab.implode())).alias("o"))
             .group_by("i").agg(pl.col("o").mean().alias("oov")))
    df = df.join(oov, on="i", how="left").sort("i")
    text = df["c"].list.join(" ")
    X = np.column_stack([
        df["c"].list.len().cast(pl.Float32).to_numpy(),
        df["oov"].fill_null(1.0).cast(pl.Float32).to_numpy(),
        (df["l"].list.len() > 0).cast(pl.Float32).to_numpy(),
        text.str.contains(r"\d").cast(pl.Float32).to_numpy(),
        (text.str.len_chars().cast(pl.Float32) / 20.0).to_numpy(),
    ]).astype(np.float32)
    return text.to_list(), X


def _features(model: NameReplacementModel, core: pl.Series, legal: pl.Series):
    vocab = model.meta["vocab_series"]
    text, dense = _dense(core, legal, vocab)
    H = HashingVectorizer(**_HASH).transform([f" {t} " for t in text])
    return sp.hstack([H, sp.csr_matrix(dense)], format="csr"), dense


def label_replacements(pos_s1_core: pl.Series, pos_tgt_core: pl.Series, pos_tgt_raw: pl.Series,
                       low: float = 45.0, high: float = 80.0) -> pl.Series:
    """Weak labels for target names in positive pairs.

    replaced (1): name similarity to its S1 < ``low``, raw name is ASCII-only (so not a
        transliteration) and contains no domain/hashtag marks ('.', '#', '@'), and there is
        at least one core token.
    ordinary (0): similarity >= ``high``, or the raw name is in a non-Latin script
        (transliterations are dissimilar but not replacements).
    Everything in between is null (ambiguous). Returns a label Series aligned with the inputs.
    """
    sim = process.cpdist(pos_s1_core.list.join(" ").fill_null("").to_list(),
                         pos_tgt_core.list.join(" ").fill_null("").to_list(),
                         scorer=fuzz.token_set_ratio, workers=-1)
    raw = pos_tgt_raw.fill_null("")
    ascii_ = ~raw.str.contains(r"[^\x00-\x7F]")
    marks = raw.str.contains(r"[.#@]")
    ntok = pos_tgt_core.list.len().fill_null(0)
    sim_s = pl.Series(sim)
    y = (pl.when((sim_s < low) & ascii_ & ~marks & (ntok > 0)).then(1)
           .when((sim_s >= high) | ~ascii_).then(0).otherwise(None))
    return pl.select(y.alias("y")).to_series()


def fit_name_lm(s1_core_tokens: pl.Series, pos_s1_core: pl.Series | None = None,
                pos_tgt_core: pl.Series | None = None, pos_tgt_raw: pl.Series | None = None,
                pos_tgt_legal: pl.Series | None = None, seed: int = 42,
                max_neg: int = 200_000, clean_core: pl.Series | None = None,
                max_clean: int = 200_000) -> NameReplacementModel:
    """Fit the name-replacement detector.

    s1_core_tokens: S1 core token lists (or flat tokens), which define the clean vocabulary.
    pos_*: aligned positive pairs (S1 core, target core, target raw name, target legal),
    used to mine weak labels. Without them the model falls back to the OOV rate
    (vocabulary-only).
    clean_core: extra *known-clean* names added as negatives, e.g. S1 names of any split.
    S1 is the deduplicated reference source and is never replaced, so this label-free data
    teaches the detector what unseen-country names (France) look like. It also joins the
    vocabulary.
    """
    s = s1_core_tokens
    if clean_core is not None:
        s = pl.concat([s if s.dtype == pl.List(pl.Utf8) else s.map_elements(lambda t: [t], return_dtype=pl.List(pl.Utf8)),
                       clean_core.cast(pl.List(pl.Utf8))])
    toks = s.explode() if s.dtype == pl.List(pl.Utf8) else s
    vc = toks.drop_nulls().value_counts()
    vocab = vc.filter(pl.col("count") >= 2)[vc.columns[0]]
    model = NameReplacementModel(vocab=set(vocab.to_list()), meta={"vocab_series": vocab})
    if pos_s1_core is None:
        return model
    y = label_replacements(pos_s1_core, pos_tgt_core, pos_tgt_raw)
    legal = (pos_tgt_legal if pos_tgt_legal is not None
             else pl.Series([[]] * pos_tgt_core.len(), dtype=pl.List(pl.Utf8)))
    lab = pl.DataFrame({"core": pos_tgt_core, "legal": legal, "y": y}).drop_nulls("y")
    pos, neg = lab.filter(pl.col("y") == 1), lab.filter(pl.col("y") == 0)
    neg = neg.sample(min(max_neg, neg.height), seed=seed)
    if clean_core is not None:
        cc = clean_core.drop_nulls().sample(min(max_clean, clean_core.drop_nulls().len()), seed=seed)
        neg = pl.concat([neg, pl.DataFrame({"core": cc.cast(pl.List(pl.Utf8)),
                                            "legal": pl.Series([[]] * cc.len(), dtype=pl.List(pl.Utf8)),
                                            "y": pl.Series([0] * cc.len(), dtype=neg["y"].dtype)})])
    tr = pl.concat([pos, neg]).sample(fraction=1.0, shuffle=True, seed=seed)
    X, _ = _features(model, tr["core"], tr["legal"])
    clf = LogisticRegression(C=2.0, max_iter=300, class_weight="balanced", solver="liblinear")
    clf.fit(X, tr["y"].to_numpy())
    model.clf = clf
    model.meta.update(n_pos=pos.height, n_neg=neg.height)
    log.info("name-replacement model: pos=%d neg=%d", pos.height, neg.height)
    return model


def name_replaced_score(model: NameReplacementModel, core: pl.Series, legal: pl.Series,
                        batch: int = 500_000) -> np.ndarray:
    """P(name was replaced by a generated pseudo-word), in [0, 1]. Vectorised and batched."""
    out = []
    for i in range(0, core.len(), batch):
        c, l = core.slice(i, batch), legal.slice(i, batch)
        X, dense = _features(model, c, l)
        if model.clf is None:  # vocabulary-only fallback: single fully-OOV token
            out.append(((dense[:, 0] == 1) & (dense[:, 1] == 1)).astype(np.float32))
        else:
            out.append(model.clf.predict_proba(X)[:, 1].astype(np.float32))
    return np.concatenate(out) if out else np.zeros(0, np.float32)


# ---------------------------------------------------------------------------
# Convenience: fit both from the training split (sampled, RAM-bounded)
# ---------------------------------------------------------------------------
def fit_from_training(work_dir: str | Path, n_pairs: int = 300_000, seed: int = 42,
                      clean_splits: tuple[str, ...] = ("train", "test")):
    """Return (filler_rates, replacement_model) fitted on a sample of train positives.

    S1 names of ``clean_splits`` are added as label-free clean negatives and vocabulary.
    This includes test S1 (France), which is legitimate because no labels are used.
    """
    from .prep import load_gt
    work = Path(work_dir)
    gt = load_gt()
    gt = gt.sample(min(n_pairs, gt.height), seed=seed)
    cols = ["entity_id", "core", "legal", "business_name"]
    s1 = (pl.scan_parquet(work / "train_s1.parquet").select(cols)
            .join(gt.lazy().select(pl.col("s1").alias("entity_id")).unique(), on="entity_id").collect())
    tg = pl.concat([pl.scan_parquet(work / f"train_s{k}.parquet").select(cols)
                    .join(gt.lazy().select(pl.col("tid").alias("entity_id")), on="entity_id").collect()
                    for k in "23"])
    d = (gt.join(s1.rename({c: c + "_1" for c in cols}).rename({"entity_id_1": "s1"}), on="s1")
           .join(tg.rename({c: c + "_2" for c in cols}).rename({"entity_id_2": "tid"}), on="tid"))
    vocab_src = pl.scan_parquet(work / "train_s1.parquet").select("core").collect()["core"]
    clean = [pl.scan_parquet(work / f"{sp_}_s1.parquet").select("core").collect()["core"]
             for sp_ in clean_splits if (work / f"{sp_}_s1.parquet").exists()]
    clean_core = pl.concat(clean) if clean else None
    filler = mine_filler_tokens(d["core_1"], d["core_2"])
    vocab_base = vocab_src if clean_core is None else pl.Series([], dtype=pl.List(pl.Utf8))
    model = fit_name_lm(vocab_base, d["core_1"], d["core_2"], d["business_name_2"], d["legal_2"],
                        seed=seed, clean_core=clean_core)
    return filler, model
