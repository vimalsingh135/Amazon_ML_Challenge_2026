import polars as pl
import pytest

from er.output import check_invariants, id_lists, write_submission


def _pairs(rows):
    return pl.DataFrame(rows, schema=["s1", "tid"], orient="row")


def test_writer_format_sorted_empty_and_superset(tmp_path):
    s1 = pl.Series(["S1-2", "S1-1", "S1-3"])
    matches = _pairs([("S1-1", "S3-9"), ("S1-1", "S2-5"), ("S1-1", "S2-5")])
    cands = _pairs([("S1-1", "S2-5"), ("S1-2", "S3-1")])       # S3-9 missing: writer must add it
    write_submission(matches, cands, s1, tmp_path)
    m = (tmp_path / "matching_results.tsv").read_text().splitlines()
    c = (tmp_path / "candidate_pairs.tsv").read_text().splitlines()
    assert m == ["source1_entity_id\tmatched_entity_ids", "S1-1\tS2-5,S3-9", "S1-2\t", "S1-3\t"]
    assert c == ["source1_entity_id\tcandidate_entity_ids", "S1-1\tS2-5,S3-9", "S1-2\tS3-1", "S1-3\t"]


def test_writer_is_deterministic(tmp_path):
    s1 = pl.Series([f"S1-{i}" for i in range(50)])
    pairs = _pairs([(f"S1-{i}", f"S2-{j}") for i in range(50) for j in range(i % 4)])
    write_submission(pairs, pairs, s1, tmp_path / "a")
    write_submission(pairs.sample(fraction=1.0, shuffle=True, seed=1), pairs, s1, tmp_path / "b")
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes()


def test_invariants_catch_bad_ids_and_missing_rows():
    s1 = pl.Series(["S1-1", "S1-2"])
    good = id_lists(_pairs([("S1-1", "S2-1")]), s1, "matched_entity_ids")
    cand = id_lists(_pairs([("S1-1", "S2-1")]), s1, "candidate_entity_ids")
    check_invariants(good, cand, s1)
    bad = id_lists(_pairs([("S1-1", "S1-9")]), s1, "matched_entity_ids")
    with pytest.raises(AssertionError):
        check_invariants(bad, cand, s1)
    with pytest.raises(AssertionError):
        check_invariants(good.head(1), cand, s1)
    orphan = id_lists(_pairs([("S1-1", "S3-7")]), s1, "matched_entity_ids")
    with pytest.raises(AssertionError):
        check_invariants(orphan, cand, s1)
