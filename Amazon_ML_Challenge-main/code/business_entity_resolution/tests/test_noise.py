import random

import numpy as np
import polars as pl
import pytest

from er.noise import (NameReplacementModel, fit_name_lm, label_replacements, mine_filler_tokens,
                      name_replaced_score)

L = pl.List(pl.Utf8)


def test_filler_rate_separates_noise_words_from_identity_words():
    s1 = pl.Series([["heritage", "interiors"]] * 40 + [["lotus", "exports"]] * 40, dtype=L)
    tg = pl.Series([["heritage", "interiors", "services"]] * 40 + [["lotus", "exports"]] * 40, dtype=L)
    rates = mine_filler_tokens(s1, tg, min_count=30)
    assert rates["services"] > 0.95            # always added by noise
    assert rates["heritage"] < 0.05            # always shared with S1
    assert "nonexistent" not in rates


def test_filler_min_count_and_nulls():
    s1 = pl.Series([["a"], None, ["b"]], dtype=L)
    tg = pl.Series([["a", "x"], ["x"], None], dtype=L)
    assert mine_filler_tokens(s1, tg, min_count=3) == {}
    assert set(mine_filler_tokens(s1, tg, min_count=2)) == {"x"}
    with pytest.raises(ValueError):
        mine_filler_tokens(s1, tg.head(2))


def test_label_replacements_rules():
    s1 = pl.Series([["flint"], ["creative", "developers"], ["heritage", "interiors"], ["miami", "college"]], dtype=L)
    tg = pl.Series([["verazeph"], ["kriettiv", "devalapars"], ["heritage", "interiors"], ["miamicollege"]], dtype=L)
    raw = pl.Series(["Verazeph", "क्रिएटिव डेवलपर्स", "Heritage Interiors LLC", "miamicollege.com"])
    y = label_replacements(s1, tg, raw).to_list()
    assert y[0] == 1          # pseudo-word replacement
    assert y[1] == 0          # transliteration is NOT a replacement
    assert y[2] == 0          # ordinary near-identical name
    assert y[3] is None       # domain: ambiguous, excluded


def _syllable_words(n, rng):
    syl = ["zeph", "orbi", "cira", "lum", "nova", "delta", "mira", "vora", "pyra", "syn", "evo", "ecto"]
    return ["".join(rng.choice(syl) for _ in range(rng.randint(2, 4))) for _ in range(n)]


def test_replacement_model_ranks_pseudo_words_above_real_names():
    rng = random.Random(0)
    real = ["heritage", "interiors", "lotus", "exports", "summit", "ministries", "cardiology",
            "partners", "family", "prairie", "association", "urban", "cream", "royal", "consulting"]
    vocab = pl.Series([[w] for w in real * 3], dtype=L)
    fake = _syllable_words(300, rng)
    ordinary = [[rng.choice(real), rng.choice(real)] for _ in range(300)]
    # positives: S1 real names; targets either pseudo-word replacements or near-copies
    s1 = pl.Series(ordinary + ordinary, dtype=L)
    tg = pl.Series([[f] for f in fake] + ordinary, dtype=L)
    raw = pl.Series(fake + [" ".join(o) for o in ordinary])
    model = fit_name_lm(vocab, s1, tg, raw, pl.Series([[]] * 600, dtype=L))
    test_fake = pl.Series([[w] for w in _syllable_words(50, random.Random(1))], dtype=L)
    test_real = pl.Series([[rng.choice(real), rng.choice(real)] for _ in range(50)], dtype=L)
    empty = pl.Series([[]] * 50, dtype=L)
    pf, pr = name_replaced_score(model, test_fake, empty), name_replaced_score(model, test_real, empty)
    assert pf.dtype == np.float32 and ((pf >= 0) & (pf <= 1)).all()
    assert pf.mean() > 0.8 and pr.mean() < 0.2


def test_score_handles_nulls_empty_batches_and_roundtrip(tmp_path):
    vocab = pl.Series([["alpha"], ["alpha"], ["beta"], ["beta"]], dtype=L)
    model = fit_name_lm(vocab)              # vocabulary-only fallback
    core = pl.Series([["zzq"], ["alpha", "beta"], None, []], dtype=L)
    legal = pl.Series([[], ["llc"], None, []], dtype=L)
    s = name_replaced_score(model, core, legal, batch=3)
    assert s.shape == (4,) and s[0] == 1.0 and s[1] == 0.0 and s[2] == 0.0
    model.save(tmp_path / "m.pkl")
    again = NameReplacementModel.load(tmp_path / "m.pkl")
    assert np.array_equal(name_replaced_score(again, core, legal), s)


def test_split_filler_separates_english_filler_from_transliterations():
    from er.noise import split_filler
    rates = {"services": 0.97, "kanstrakshan": 1.0, "heritage": 0.01}
    filler, foreign = split_filler(rates, s1_vocab={"services", "heritage", "construction"})
    assert filler == {"services"} and foreign == {"kanstrakshan"}


def test_clean_negatives_protect_unseen_country_names():
    """Unseen-country words that look like pseudo-words must be pulled down by clean S1 names."""
    rng = random.Random(0)
    real = ["heritage", "interiors", "lotus", "exports", "summit", "ministries", "family", "urban"]
    french = ["amicale", "loisirs", "boulistes", "ecole", "federation", "pharmacie", "parents", "etoiles"]
    fake = _syllable_words(300, rng)
    ordinary = [[rng.choice(real), rng.choice(real)] for _ in range(300)]
    s1 = pl.Series(ordinary + ordinary, dtype=L)
    tg = pl.Series([[f] for f in fake] + ordinary, dtype=L)
    raw = pl.Series(fake + [" ".join(o) for o in ordinary])
    vocab = pl.Series([[w] for w in real * 3], dtype=L)
    probe = pl.Series([[w] for w in french], dtype=L)
    empty = pl.Series([[]] * len(french), dtype=L)
    base = fit_name_lm(vocab, s1, tg, raw)
    clean = pl.Series([[w] for w in french * 20] + [[rng.choice(real)] for _ in range(100)], dtype=L)
    adapted = fit_name_lm(vocab, s1, tg, raw, clean_core=clean)
    assert name_replaced_score(adapted, probe, empty).mean() < name_replaced_score(base, probe, empty).mean()
    assert name_replaced_score(adapted, probe, empty).mean() < 0.3
