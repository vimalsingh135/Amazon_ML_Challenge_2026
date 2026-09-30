import numpy as np
import polars as pl

from er.features import REC_COLS, coherence_features, name_freq, pair_features, token_idf
from er.normalize import normalize_record
from er.prep import SCHEMA


def _table(rows, country="India"):
    recs = [normalize_record(n, a, country) for n, a in rows]
    df = pl.DataFrame({k: list(v) for k, v in zip(SCHEMA, zip(*recs))}, schema=SCHEMA)
    return df.with_columns(pl.int_range(pl.len(), dtype=pl.UInt32).alias("idx"),
                           pl.lit(country).alias("country"),
                           pl.Series("business_name", [r[0] for r in rows]))


S1 = _table([("Creative Consulting Pvt Ltd", "12 MG Road, Pune, Maharashtra"),
             ("Lotus Exports Private Limited", "54, Rathi Nagar, Amravati, Maharashtra")])
S2 = _table([("Kriettiv Kansalting Pvt Ltd", "12 MG RD, PUNE, MH"),
             ("Lotus Exports Pvt Ltd", "54, AMRAVATI, Maharashtra"),
             ("Unrelated Traders", "99 Other St, Nagpur, MH")])
S3 = _table([("Creative Consulting Services", "12 Mg Rd, Pune, MH")])


def _run(enrich=None):
    tabs = {"1": S1, "2": S2, "3": S3}
    if enrich:
        tabs = {k: enrich(v) for k, v in tabs.items()}
    pairs = pl.DataFrame({"s1_idx": [0, 0, 1, 0], "t_idx": [0, 2, 1, 0], "src": [2, 2, 2, 3]},
                         schema={"s1_idx": pl.UInt32, "t_idx": pl.UInt32, "src": pl.UInt8})
    tg = pl.concat([tabs[s].with_columns(pl.lit(int(s), pl.UInt8).alias("src")) for s in ("2", "3")],
                   how="diagonal_relaxed")
    idf_c = token_idf(list(tabs.values()), "core")
    idf_a = token_idf(list(tabs.values()), "atoks")
    f = pair_features(pairs, tabs["1"], tg, idf_c, idf_a, name_freq(tabs["1"], "_1"), name_freq(tg, "_2"))
    return f, tabs


def test_pair_features_shape_order_and_no_list_leakage():
    f, _ = _run()
    assert f.height == 4 and f["s1_idx"].to_list() == [0, 0, 1, 0]      # row order preserved
    assert {c for c, t in f.schema.items() if not t.is_numeric()} == {"country"}  # dropped before training
    # the true match (Lotus) scores far above the distractor
    assert f["c_tset"][2] > 90 and f["c_tset"][1] < 50


def test_alias_enrichment_raises_similarity_for_transliterations():
    def enrich(t):
        return t.with_columns(
            pl.col("core").list.eval(pl.element().replace({"kriettiv": "creative", "kansalting": "consulting"}))
              .alias("core_a"),
            pl.col("atoks").alias("atoks_a"), pl.lit(0.0, pl.Float32).alias("nrep"))
    f, _ = _run(enrich)
    assert f["ca_tset"][0] == 100 and f["c_tset"][0] < f["ca_tset"][0]
    assert {"ca_jac", "aa_tset", "nrep_2"} <= set(f.columns)


def test_coherence_features_link_cross_source_duplicates():
    f, tabs = _run()
    anchors = pl.DataFrame({"s1_idx": [0, 0, 1], "src": [2, 3, 2], "a_idx": [2, 0, 1], "a_p1": [0.4, 0.9, 0.95]},
                           schema={"s1_idx": pl.UInt32, "src": pl.UInt8, "a_idx": pl.UInt32, "a_p1": pl.Float32})
    c = coherence_features(f, tabs, anchors)
    assert c.height == f.height
    # S2 'Kriettiv..' vs S3 anchor 'Creative Consulting Services': the addresses agree strongly
    assert c["coh_a_oth"][0] > 80
    # own-source anchor of pair 1 is itself -> masked
    assert c["coh_c_same"][1] == -1
    # S1 #1 has no S3 anchor -> masked
    assert c["coh_c_oth"][2] == -1 and np.isclose(c["coh_p_oth"][2], -1)
