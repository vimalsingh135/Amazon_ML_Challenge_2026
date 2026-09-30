import polars as pl
import pytest

from er.aliases import (mine_part_aliases, mine_token_aliases, norm_part, segment,
                        split_alias_markers)

L = pl.List(pl.Utf8)


@pytest.mark.parametrize("raw,main,alias", [
    ("evovio d/b/a commercial asset partners llc", "evovio", "commercial asset partners llc"),
    ("ectokelo f/k/a vaayuika developers pvt ltd", "ectokelo", "vaayuika developers pvt ltd"),
    ("pyravio formerly: grand intelligence industries", "pyravio", "grand intelligence industries"),
    ("fluxxylo co dba: auron imaging inc", "fluxxylo co", "auron imaging inc"),
    ("wexcalokor F/K/A TL Manufacturing LLC", "wexcalokor", "TL Manufacturing LLC"),
    ("lumnylabrix formerly known as zion consulting ltd", "lumnylabrix", "zion consulting ltd"),
])
def test_split_alias_markers(raw, main, alias):
    assert split_alias_markers(raw) == (main, alias)


@pytest.mark.parametrize("raw", ["dba organisers pvt ltd", "aka technologies", "aka medical llp",
                                 "summit ministries llc", "", None])
def test_split_alias_markers_leaves_real_names(raw):
    assert split_alias_markers(raw) == (raw or "", None)


def test_mine_token_aliases_transliterations_and_city_variants():
    s1 = pl.Series([["creative", "consulting"]] * 8 + [["kolkata", "trading"]] * 6 + [["alpha", "beta"]] * 8, dtype=L)
    tg = pl.Series([["kriettiv", "kansalting"]] * 8 + [["calcutta", "trading"]] * 6 + [["alpha", "betta"]] * 8, dtype=L)
    al = mine_token_aliases(s1, tg, min_count=5)
    assert al["kansalting"] == "consulting"
    assert al["calcutta"] == "kolkata"
    assert al["betta"] == "beta"
    assert "trading" not in al                      # shared tokens never become aliases


def test_mine_token_aliases_rejects_impure_and_rare():
    # 'zorbo' aligns to different S1 tokens each time -> impure; 'rarex' too rare
    s1 = pl.Series([["zorba"], ["zorbu"], ["zorbi"], ["zorby"], ["zorbe"], ["rare"]], dtype=L)
    tg = pl.Series([["zorbo"]] * 5 + [["rarex"]], dtype=L)
    al = mine_token_aliases(s1, tg, min_count=2, min_purity=0.6)
    assert "zorbo" not in al and "rarex" not in al


def test_mine_token_aliases_respects_vocab_frequency():
    s1 = pl.Series([["heritage"]] * 6, dtype=L)
    tg = pl.Series([["heritages"]] * 6, dtype=L)
    assert mine_token_aliases(s1, tg, min_count=5)["heritages"] == "heritage"
    # a common real S1 word must not be remapped
    assert mine_token_aliases(s1, tg, min_count=5, s1_vocab_freq={"heritages": 50}) == {}


def test_mine_part_aliases_native_script_states():
    s1 = pl.Series(["12 anna salai, chennai, tamil nadu", "4 mount road, madurai, tamil nadu",
                    "9 gandhi st, salem, tamil nadu", "1 beach rd, trichy, tamil nadu",
                    "7 lake view, vellore, tamil nadu", "22 mg road, bengaluru, karnataka"] * 2)
    tg = pl.Series(["12 anna salai, chennai, தமிழ்நாடு", "4 mount rd, madurai, தமிழ்நாடு",
                    "9 gandhi st, salem, தமிழ்நாடு", "1 beach rd, trichy, தமிழ்நாடு",
                    "7 lake view, vellore, தமிழ்நாடு", "22 mg rd, bengaluru, ಕರ್ನಾಟಕ"] * 2)
    al = mine_part_aliases(s1, tg, min_count=5)
    assert al[norm_part("தமிழ்நாடு")] == "tamil nadu"
    assert norm_part("ಕರ್ನಾಟಕ") not in al           # only 2 occurrences < min_count
    assert all(not any(ch.isdigit() for ch in k) for k in al)


def test_segment():
    vocab = {"meghana": 40, "gold": 500, "nantes": 30, "federation": 60, "services": 900,
             "miami": 80, "college": 300, "golden": 90}
    assert segment("meghanagold", vocab) == ["meghana", "gold"]
    assert segment("nantesfederationeurl", vocab) == ["nantes", "federation", "eurl"]
    assert segment("bsservicescom", vocab) == ["bs", "services", "com"]
    assert segment("miamicollege", vocab) == ["miami", "college"]
    assert segment("college", vocab) == ["college"]           # common word stays whole
    assert segment("zqxwvjkp", vocab) == ["zqxwvjkp"]           # nothing confident
    assert segment("golden", vocab) == ["golden"]


def test_ocr_fix_repairs_digit_swaps_but_keeps_real_alphanumerics():
    from er.auxfit import ocr_fix
    toks = ["g1obal", "c0astal", "5uper", "roya1", "pinnac1e", "b2b", "24x7", "3m", "1st", "41st",
            "d2o", "h0tel", "12", "global"]
    out = pl.DataFrame({"t": toks}).select(ocr_fix(pl.col("t")))["t"].to_list()
    assert out == ["global", "coastal", "super", "royal", "pinnacle", "b2b", "24x7", "3m", "1st", "41st",
                   "d2o", "hotel", "12", "global"]
