"""Fit data-derived auxiliary models once, from train positives only (plus label-free S1 names).

Outputs in work/aux/:
  filler.json       token -> P(inserted by noise)                     (er.noise, R1a)
  name_model.pkl    pseudo-word name-replacement detector             (er.noise, R1b)
  core_alias.json   name token aliases   (kansalting -> consulting)   (er.aliases, N4)
  addr_alias.json   address token aliases (calcutta -> kolkata)       (er.aliases, N4)
and helpers that apply them to normalised record tables.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

from . import config as C
from .aliases import mine_token_aliases
from .noise import NameReplacementModel, fit_from_training, name_replaced_score, split_filler
from .prep import load_gt

log = logging.getLogger(__name__)
AUX = C.WORK / "aux"


def fit_aux(n_pairs: int = 300_000) -> None:
    AUX.mkdir(parents=True, exist_ok=True)
    if (AUX / "addr_alias.json").exists():
        return
    filler, model = fit_from_training(C.WORK, n_pairs=n_pairs, seed=C.SEED)
    model.save(AUX / "name_model.pkl")
    s1_core = pl.scan_parquet(C.WORK / "train_s1.parquet").select("core").collect()["core"]
    vocab = dict(s1_core.explode().drop_nulls().value_counts().iter_rows())
    # filler must be *common* in clean S1 names; rare S1 tokens (rayal, skai, siti) are identity-bearing
    fill_set, foreign = split_filler(filler, {t for t, c in vocab.items() if c >= 100})
    (AUX / "filler.json").write_text(json.dumps({"filler": sorted(fill_set), "foreign": sorted(foreign),
                                                 "rates": filler}))
    gt = load_gt()
    gt = gt.sample(min(n_pairs, gt.height), seed=C.SEED + 1)
    cols = ["entity_id", "core", "atoks"]
    s1 = (pl.scan_parquet(C.WORK / "train_s1.parquet").select(cols)
            .join(gt.lazy().select(pl.col("s1").alias("entity_id")).unique(), on="entity_id").collect())
    tg = pl.concat([pl.scan_parquet(C.WORK / f"train_s{k}.parquet").select(cols)
                    .join(gt.lazy().select(pl.col("tid").alias("entity_id")), on="entity_id").collect()
                    for k in "23"])
    d = (gt.join(s1.rename({"entity_id": "s1"}), on="s1")
           .join(tg.rename({"entity_id": "tid"}), on="tid", suffix="_2"))
    core_alias = mine_token_aliases(d["core"], d["core_2"], min_count=5, min_purity=0.6, s1_vocab_freq=vocab)
    a_vocab = dict(pl.scan_parquet(C.WORK / "train_s1.parquet").select("atoks").collect()["atoks"]
                   .explode().drop_nulls().value_counts().iter_rows())
    addr_alias = mine_token_aliases(d["atoks"], d["atoks_2"], min_count=8, min_purity=0.6, s1_vocab_freq=a_vocab)
    (AUX / "core_alias.json").write_text(json.dumps(core_alias))
    (AUX / "addr_alias.json").write_text(json.dumps(addr_alias))
    log.info("aux: filler=%d foreign=%d core_alias=%d addr_alias=%d name_model pos=%s",
             len(fill_set), len(foreign), len(core_alias), len(addr_alias), model.meta.get("n_pos"))


def ocr_fix(tok: pl.Expr) -> pl.Expr:
    """Undo the generator's OCR-style digit swaps in word tokens (g1obal, c0astal, 5uper, roya1).

    Applied only to tokens with >= 3 letters whose digits are all in {0, 1, 5}, so real
    alphanumerics (b2b, 24x7, 3m, d2o, 41st) are left untouched."""
    cond = (tok.str.count_matches(r"[a-z]") >= 3) & tok.str.contains(r"[015]") & ~tok.str.contains(r"[2-46-9]")
    return pl.when(cond).then(tok.str.replace_many(["0", "1", "5"], ["o", "l", "s"])).otherwise(tok)


class Aux:
    """Loaded auxiliary models + the table-level transforms used by stage B."""

    def __init__(self, path: Path = AUX):
        f = json.loads((path / "filler.json").read_text())
        self.filler = f["filler"]
        self.core_alias = json.loads((path / "core_alias.json").read_text())
        self.addr_alias = json.loads((path / "addr_alias.json").read_text())
        self.model = NameReplacementModel.load(path / "name_model.pkl")

    def enrich(self, tab: pl.DataFrame) -> pl.DataFrame:
        """Add core_a (alias-mapped, filler-free core), atoks_a (alias-mapped address) and nrep."""
        ca, aa = self.core_alias, self.addr_alias
        tab = tab.with_columns(
            pl.col("core").list.eval(pl.element().replace(ca)).list.eval(ocr_fix(pl.element())).list.eval(
                pl.element().filter(~pl.element().is_in(self.filler))).alias("core_a"),
            pl.col("atoks").list.eval(pl.element().replace(aa)).alias("atoks_a"),
        )
        nrep = name_replaced_score(self.model, tab["core"], tab["legal"])
        return tab.with_columns(pl.Series("nrep", nrep.astype(np.float32)))
