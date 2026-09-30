import polars as pl
import pytest

from er.metric import macro_f05
from er.normalize import normalize_address, normalize_name
from er.text import skeleton, to_ascii


@pytest.mark.parametrize("raw,expected", [
    ("राम मार्केटिंग प्राइवेट लिमिटेड", "ram marketing praivet limited"),
    ("आदित्य", "aditya"),
    ("Léarning Ínteriors Sàrl", "learning interiors sarl"),
])
def test_to_ascii(raw, expected):
    assert to_ascii(raw) == expected


def test_skeleton_transliteration_invariance():
    assert skeleton("praivet") == skeleton("private")
    assert skeleton("limited") == skeleton("limited")
    assert skeleton("eksports") == skeleton("exports")


def test_normalize_name_legal_and_junk():
    norm, core, legal = normalize_name("Shandra Alfaro, P.A.,-M.D., P.C.")
    assert core == ["shandra", "alfaro"] and legal == ["md", "pa", "pc"]
    _, core, legal = normalize_name("meghanagold.com | www.meghanagol.com")
    assert core == ["meghanagold"] and legal == []
    _, core, legal = normalize_name("Heritage Ibtdeiirofs, [Llc]")
    assert core == ["heritage", "ibtdeiirofs"] and legal == ["llc"]
    _, core, _ = normalize_name("SHIVSHAKTI VIDYALAYA VIDYALAYA OVERSEAS CORPORATION")
    assert core == ["shivshakti", "vidyalaya", "overseas"]


def test_normalize_address_components():
    toks, nums, hno, state, loc, street = normalize_address(
        "9Th Street, Ferns Paradise, Bangalore North, No 755, Bangalore, Karnataka", "India")
    assert hno == "755" and state == "st_karnataka" and loc[0] == "bangalore"
    _, _, hno, state, loc, street = normalize_address("OH, Columbus, 5559 Orville Avenue", "US")
    assert (hno, state, loc, street) == ("5559", "st_ohio", ["columbus"], ["orville"])
    _, _, hno, state, _, _ = normalize_address("4113 16TH ST, NINENKAH, NC", "US")
    assert hno == "4113" and state == "st_northcarolina"
    # unseen country must not crash and must not invent a state
    _, nums, hno, state, loc, _ = normalize_address("Nº 8 PL PIERRE BÉRÉGOVOY, LILLE, Nord", "France")
    assert hno == "8" and state == "" and "lille" in loc
    assert normalize_address(None, "US")[0] == []


def test_macro_f05_rules():
    truth = pl.DataFrame({"s1": ["a", "a", "b"], "tid": ["x", "y", "z"]})
    pred = pl.DataFrame({"s1": ["a", "a", "a", "c"], "tid": ["x", "y", "w", "q"]})
    ids = pl.Series(["a", "b", "c", "d"])
    r = macro_f05(pred, truth, ids)
    # a: P=2/3 R=1 -> 0.714; b: miss -> 0; c: FP on singleton -> 0; d: empty/empty -> 1
    fa = 1.25 * (2 / 3) / (0.25 * (2 / 3) + 1)
    assert r["f05"] == pytest.approx((fa + 0 + 0 + 1) / 4)


def test_blocking_keys_survive_typos_and_name_swaps():
    from er.blocking import FAM_ID, make_keys
    df = pl.DataFrame({
        "idx": [0, 1, 2],
        "core": [["heritage", "interiors"], ["heritage", "interors"], ["lumzeta"]],
        "skel": [["rtj", "ntrs"], ["rtj", "ntrs"], ["lmjt"]],
        "loc": [["wooster"], ["wooster"], ["wooster"]],
        "street": [["stibbs"], ["stibbs"], ["stibbs"]],
        "nums": [["837", "12"], ["937"], ["837", "12"]],
    }, schema_overrides={"idx": pl.UInt32})
    k = make_keys(df)
    keys = {i: set(zip(g["key"], g["fam"])) for (i,), g in k.group_by("idx")}
    typo = keys[0] & keys[1]          # name typo + house-number typo
    assert {f for _, f in typo} >= {FAM_ID["n"], FAM_ID["s"], FAM_ID["nl"], FAM_ID["aa"]}
    swap = keys[0] & keys[2]          # completely different name, same address
    assert {f for _, f in swap} >= {FAM_ID["a1"], FAM_ID["a2"], FAM_ID["a4"]}
    assert not any(f in (FAM_ID["n"], FAM_ID["s"]) for _, f in swap)


def test_full_skeleton_key_matches_transliterations_not_different_businesses():
    from er.blocking import iter_key_families
    from er.normalize import normalize_record
    L = pl.List(pl.Utf8)

    def fs_keys(pairs):
        rows = [dict(zip(("core", "skel"), (lambda r: (r[1], r[3]))(normalize_record(n, "", "India"))))
                for a, b in pairs for n in (a, b)]
        df = (pl.DataFrame(rows, schema={"core": L, "skel": L}).with_row_index("idx")
                .with_columns([pl.lit([], dtype=L).alias(c) for c in ("loc", "street", "nums")]))
        fs = {f: k for f, k in iter_key_families(df)}["fs"]
        k = dict(zip(fs["idx"].to_list(), fs["key"].to_list()))
        return [k.get(2 * i) is not None and k.get(2 * i) == k.get(2 * i + 1) for i in range(len(pairs))]

    pos = [("City Media Private Limited", "सिटी मीडिया प्राइवेट लिमिटेड"),
           ("Seven Business Private Limited", "सेवन बिजनेस प्राइवेट लिमिटेड"),
           ("Tirupati Consulting Private Limited", "तिरुपति कंसल्टिंग प्राइवेट लिमिटेड")]
    neg = [("City Media Private Limited", "Metro Media Private Limited"),
           ("Lotus Exports Private Limited", "Lotus Imports Private Limited"),
           ("Tirupati Consulting Private Limited", "Tirupati Construction Private Limited")]
    assert all(fs_keys(pos)) and not any(fs_keys(neg))


def test_learned_admin_parts_are_dropped_for_unlisted_countries(monkeypatch):
    from er import normalize as N
    monkeypatch.setattr(N, "ADMIN_DROP", {"France": {"hauts de france", "nord"}})
    for a in ("20 Rue Parmentier, Dunkerque, Hauts-de-France", "20 R PARMENTIER, DUNKERQUE, Nord",
              "20 R Parmentier, Dunkerque"):
        toks, nums, hno, state, loc, street = N.normalize_address(a, "France")
        assert loc == ["dunkerque"] and state == "" and "france" not in toks and "nord" not in toks
    assert N.normalize_address("OH, Columbus, 5559 Orville Avenue", "US")[3] == "st_ohio"   # lexicon countries untouched
