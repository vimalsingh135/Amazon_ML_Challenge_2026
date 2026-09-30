"""Label-free search for more France false-pair groups in the LB-0.988 file (Balaji's pifit), beyond his one-word
category-swap rule. Groups of SELECTED France pairs in the same building (same house number, address token_set >= 80):
  G1 disjoint core names, all words in the top-300 French vocabulary   (plausibly different co-located businesses)
  G2 disjoint core names, some word outside the vocabulary              (generator pseudo-word renames -> likely true)
  G3 names differ by 2+ words, all differing words in vocab (multi-word category change)
  G4 one name's core is a strict subset of the other (partial name)
pifit = share of FALSE pairs in a group (count-distribution mixture fit vs untouched French S1). Removal helps F0.5 when
the false share is above ~0.25; ship only well above that."""
import io
import json
import re
import unicodedata
import zipfile

import pandas as pd
import polars as pl
from rapidfuzz import fuzz, process

R = "E:/projects/Amazon ML challenge"
W = R + "/.worktrees/v6val/work_v6"
V = set(json.load(open(R + "/.worktrees/v6val/tmp/fr_vocab300.json")))
D = {"developpement", "fils", "services", "service", "groupe", "st", "saint"}
LEGAL = {"private", "pvt", "pv", "limited", "ltd", "llp", "llc", "inc", "incorporated", "corporation", "corp", "company", "co", "pc", "pa", "plc",
         "lp", "ltda", "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "gmbh", "opc", "public", "pllc", "lc", "cie", "md", "dds", "cpa", "ei"}
STOP = {"and", "the", "of", "et", "de", "du", "des", "la", "le", "les", "l", "d", "a", "an", "dba", "m/s", "ms", "india", "france", "usa", "us"}
DOM = re.compile(r"^(https?://)?(www\.)?|\.(co\.in|com|in|net|org|fr|biz|info|co|io|us)\b")


def core(name):   # Balaji's core() from france_rule_apply.py
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    s = next((p.strip() for p in s.split("|") if p.strip()), "")
    m = re.search(r"\b(d/b/a|dba|doing business as)\b", s)
    if m and s[:m.start()].strip(" :-,") and s[m.end():].strip(" :-,"):
        s = s[m.end():]
    s = DOM.sub(" ", s)
    s = re.sub(r"\b([a-z])\.", r"\1", s).replace("&", " and ")
    toks = re.sub(r"[^a-z0-9]+", " ", s).split()
    c = {t for t in toks if t not in LEGAL and t not in STOP}
    return c or {t for t in toks if t not in STOP} or set(toks)


def load(path):
    raw = open(path, "rb").read()
    if raw[:2] == b"PK":
        z = zipfile.ZipFile(io.BytesIO(raw))
        raw = z.read(next(n for n in z.namelist() if n.endswith(".tsv")))
    d = pl.read_csv(io.BytesIO(raw), separator="\t", quote_char=None, infer_schema=False)
    d = d.rename({d.columns[0]: "s1", d.columns[1]: "m"})
    return d.with_columns(pl.col("m").fill_null("").str.split(",")).explode("m").filter(pl.col("m") != "").select("s1", pl.col("m").alias("t"))


S = pl.read_parquet(f"{W}/test_s1.parquet", columns=["entity_id", "business_name", "addr_norm", "hno", "country"]).filter(pl.col("country") == "France").select(
    pl.col("entity_id").alias("s1"), pl.col("business_name").alias("b1"), pl.col("addr_norm").fill_null("").alias("a1"), pl.col("hno").fill_null("").str.strip_chars().alias("h1"))
T = pl.concat([pl.read_parquet(f"{W}/test_s{s}.parquet", columns=["entity_id", "business_name", "addr_norm", "hno"]) for s in (2, 3)]).select(
    pl.col("entity_id").alias("t"), pl.col("business_name").alias("b2"), pl.col("addr_norm").fill_null("").alias("a2"), pl.col("hno").fill_null("").str.strip_chars().alias("h2"))
p = load(R + "/matching_results_0.988_balaji").join(S.select("s1"), on="s1")
x = p.join(S, on="s1").join(T, on="t")
x = x.with_columns(pl.Series("sa", process.cpdist(x["a1"].to_list(), x["a2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)))
x = x.filter((pl.col("h1") != "") & (pl.col("h1") == pl.col("h2")) & (pl.col("sa") >= 80))
pdx = x.select("s1", "t", "b1", "b2").to_pandas()
grp = []
for b1, b2 in zip(pdx.b1, pdx.b2):
    A, B = core(b1), core(b2)
    if not (A & B):
        grp.append("G1_disjoint_vocab" if (A | B) <= V else "G2_disjoint_pseudo")
    elif A < B or B < A:
        grp.append("G4_partial")
    else:
        dA, dB = A - B, B - A
        if len(dA) >= 2 or len(dB) >= 2:
            grp.append("G3_multiword_vocab" if (dA | dB) <= V and not ((dA | dB) & D) else "other_multi")
        else:
            grp.append("other")
pdx["g"] = grp
nsel = S.select("s1").join(p.group_by("s1").len("n"), on="s1", how="left").fill_null(0).to_pandas().set_index("s1")["n"]


def pifit(flagged):
    K = 9
    ref = nsel.drop(flagged.index).clip(upper=K).value_counts(normalize=True).reindex(range(K + 1), fill_value=0)
    obs = nsel.loc[flagged.index].clip(upper=K).value_counts(normalize=True).reindex(range(K + 1), fill_value=0)
    k = int(round(flagged.mean()))
    sb = ref * ref.index
    sb = sb / sb.sum()
    alt = ref.shift(k, fill_value=0)
    da, do = alt - sb, obs - sb
    return float((da * do).sum() / (da * da).sum())


rows = []
for gname, q in pdx.groupby("g"):
    fl = q.groupby("s1").size()
    rows.append({"group": gname, "pairs": len(q), "s1": len(fl), "pifit_share_false": round(pifit(fl), 3)})
out = pd.DataFrame(rows).sort_values("pifit_share_false", ascending=False)
print(out.to_string(index=False))
pdx.to_parquet(R + "/.worktrees/v6val/out/fr_groups.parquet")
for gname in ("G1_disjoint_vocab", "G3_multiword_vocab", "G4_partial"):
    q = pdx[pdx.g == gname].sample(min(6, (pdx.g == gname).sum()), random_state=0)
    print(f"\n{gname} samples:\n" + "\n".join(f"  {a[:45]:45s} | {b[:45]}" for a, b in zip(q.b1, q.b2)))
