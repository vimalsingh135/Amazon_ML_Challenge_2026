"""Record-level normalisation of business names and addresses.

Everything here uses only the provided data plus general linguistic knowledge
(legal-form abbreviations, street-type abbreviations, state abbreviations). No
entity-level external data is used.
"""
from __future__ import annotations

import os
import re

from .aliases import split_alias_markers
from .text import skeleton, to_ascii

# --- names ------------------------------------------------------------------
LEGAL = {
    "private": "pvt", "pvt": "pvt", "pv": "pvt", "limited": "ltd", "ltd": "ltd", "limted": "ltd",
    "llp": "llp", "llc": "llc", "inc": "inc", "incorporated": "inc", "corporation": "corp",
    "corp": "corp", "company": "co", "co": "co", "pc": "pc", "pa": "pa", "plc": "plc",
    "lp": "lp", "ltda": "ltd", "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa",
    "eurl": "eurl", "sci": "sci", "snc": "snc", "gmbh": "gmbh", "opc": "opc", "public": "public",
    "pllc": "pllc", "lc": "llc", "cie": "cie", "md": "md", "dds": "dds", "cpa": "cpa",
    # transliterated legal forms (Indic-script names romanised by er.text)
    "praivet": "pvt", "praivat": "pvt", "privet": "pvt", "prayvet": "pvt", "limited": "ltd",
    "limitad": "ltd", "limited.": "ltd", "elelpi": "llp", "elaelapi": "llp", "elelapi": "llp", "kampani": "co", "kampni": "co",
    "korporeshan": "corp", "karporeshan": "corp", "inkarporetad": "inc",
}
NAME_STOP = {"and", "the", "of", "et", "de", "du", "des", "la", "le", "les", "l", "d", "a", "an",
             "dba", "m/s", "ms", "india", "france", "usa", "us"}
_DOMAIN = re.compile(r"^(https?://)?(www\.)?|\.(co\.in|com|in|net|org|fr|biz|info|co|io|us)\b")
_ONE_LETTER_DOT = re.compile(r"\b([a-z])\.")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_name(raw: str | None):
    s = to_ascii(raw or "")
    parts = [p.strip() for p in s.split("|")]
    s = next((p for p in parts if p), "")
    # "<pseudo-word> d/b/a <real name>": the identity-bearing name is the alias (see er.aliases)
    main, alias = split_alias_markers(s)
    if alias:
        s = alias
    s = _DOMAIN.sub(" ", s)
    if " " not in s.strip() and len(s.strip()) >= 9 and s.strip().endswith("com"):
        s = s.strip()[:-3]  # glued domain without a dot: "bsservicescom" -> "bsservices"
    s = _ONE_LETTER_DOT.sub(r"\1", s)
    s = s.replace("&", " and ")
    toks = _NON_ALNUM.sub(" ", s).split()
    seen, uniq = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    legal = sorted({LEGAL[t] for t in uniq if t in LEGAL})
    core = [t for t in uniq if t not in LEGAL and t not in NAME_STOP]
    if not core:  # name made only of legal/stop words: keep them rather than nothing
        core = [t for t in uniq if t not in NAME_STOP] or uniq
    return " ".join(uniq), core, legal


# --- addresses ---------------------------------------------------------------
ABBR = {
    # street types (US / India / France); 'saint' is merged with 'st' on purpose (noise maps St->Saint)
    "street": "st", "st": "st", "str": "st", "saint": "st", "road": "rd", "rd": "rd",
    "avenue": "ave", "ave": "ave", "av": "ave", "boulevard": "blvd", "blvd": "blvd", "bd": "blvd",
    "bld": "blvd", "drive": "dr", "dr": "dr", "lane": "ln", "ln": "ln", "court": "ct", "ct": "ct",
    "place": "pl", "pl": "pl", "highway": "hwy", "hwy": "hwy", "parkway": "pkwy", "pkwy": "pkwy",
    "circle": "cir", "cir": "cir", "trail": "trl", "trl": "trl", "terrace": "ter", "ter": "ter",
    "square": "sq", "sq": "sq", "way": "way", "plaza": "plz", "plz": "plz", "mount": "mt",
    "mt": "mt", "fort": "ft", "ft": "ft", "center": "ctr", "centre": "ctr", "ctr": "ctr",
    "north": "n", "n": "n", "south": "s", "s": "s", "east": "e", "e": "e", "west": "w", "w": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "apartment": "unit", "apt": "unit", "unit": "unit", "suite": "unit", "ste": "unit",
    "floor": "fl", "fl": "fl", "flr": "fl", "building": "bldg", "bldg": "bldg",
    "rue": "rue", "r": "rue", "quai": "quai", "q": "quai", "chemin": "ch", "ch": "ch",
    "allee": "all", "all": "all", "impasse": "imp", "imp": "imp", "route": "rte", "rte": "rte",
    "near": "nr", "nr": "nr", "opposite": "opp", "opp": "opp", "behind": "bh", "beside": "bs",
    "district": "dist", "dist": "dist", "sector": "sec", "sec": "sec", "phase": "ph", "ph": "ph",
    "marg": "marg", "nagar": "ngr", "ngr": "ngr", "colony": "col", "col": "col",
    "first": "1st", "second": "2nd", "third": "3rd",
}
ADDR_MARKERS = {"no", "h", "hno", "door", "plot", "flat", "shop", "house", "box", "po", "pmb",
                "c", "o", "co", "city", "of", "county", "icty", "null", "none", "na", "the",
                "and", "de", "du", "des", "la", "le", "les", "l", "d", "bis", "ter", "half"}
_TYPES = set(ABBR.values())

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland", "ma": "massachusetts",
    "mi": "michigan", "mn": "minnesota", "ms": "mississippi", "mo": "missouri", "mt": "montana",
    "ne": "nebraska", "nv": "nevada", "nh": "new hampshire", "nj": "new jersey",
    "nm": "new mexico", "ny": "new york", "nc": "north carolina", "nd": "north dakota",
    "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania", "ri": "rhode island",
    "sc": "south carolina", "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah",
    "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia", "pr": "puerto rico",
}
IN_STATES = {
    "ap": "andhra pradesh", "ar": "arunachal pradesh", "as": "assam", "br": "bihar",
    "cg": "chhattisgarh", "ct": "chhattisgarh", "ga": "goa", "gj": "gujarat", "hr": "haryana",
    "hp": "himachal pradesh", "jh": "jharkhand", "ka": "karnataka", "kl": "kerala",
    "mp": "madhya pradesh", "mh": "maharashtra", "mn": "manipur", "ml": "meghalaya",
    "mz": "mizoram", "nl": "nagaland", "od": "odisha", "or": "odisha", "orissa": "odisha",
    "pb": "punjab", "rj": "rajasthan", "sk": "sikkim", "tn": "tamil nadu", "tg": "telangana",
    "ts": "telangana", "tr": "tripura", "up": "uttar pradesh", "uk": "uttarakhand",
    "ut": "uttarakhand", "uttaranchal": "uttarakhand", "wb": "west bengal", "dl": "delhi",
    "jk": "jammu and kashmir", "ch": "chandigarh", "py": "puducherry", "pondicherry": "puducherry",
    "la": "ladakh", "dn": "dadra and nagar haveli", "an": "andaman and nicobar islands",
}


def _state_map(abbr: dict[str, str]) -> dict[str, str]:
    m = {}
    for k, v in abbr.items():
        canon = "st_" + v.replace(" ", "")
        m[k] = canon
        m[v] = canon
        m[v.replace(" ", "")] = canon
    return m


STATE_MAPS = {"US": _state_map(US_STATES), "India": _state_map(IN_STATES)}
# learned aliases (e.g. native-script state names) are merged in at runtime
STATE_ALIASES: dict[str, dict[str, str]] = {}
# learned administrative parts (region / department ...) for countries without a state lexicon;
# dropped from tokens and locality (state stays unknown). Loaded from ER_ADMIN_PARTS (JSON file,
# {country: [normalised part, ...]}) so spawned worker processes see them too. See er.admin.
ADMIN_DROP: dict[str, set[str]] = {}
if os.environ.get("ER_ADMIN_PARTS") and os.path.exists(os.environ["ER_ADMIN_PARTS"]):
    import json as _json
    ADMIN_DROP = {c: set(v) for c, v in _json.load(open(os.environ["ER_ADMIN_PARTS"], encoding="utf-8")).items()}

_HALF = re.compile(r"(\d)\s*1/2\b")
_NUM = re.compile(r"\d+")
_ORDINAL = re.compile(r"^\d+(st|nd|rd|th)$")


def normalize_address(raw: str | None, country: str):
    """Return (tokens, nums, hno, state, loc_tokens, street_tokens)."""
    s = to_ascii(raw or "")
    s = _HALF.sub(r"\1 half", s)
    smap = STATE_MAPS.get(country, {})
    alias = STATE_ALIASES.get(country, {})
    drop = ADMIN_DROP.get(country, ())
    tokens: list[str] = []
    loc_parts: list[list[str]] = []
    street: list[str] = []
    nums: list[str] = []
    hno = ""
    state = ""
    for part in s.split(","):
        ptoks = _NON_ALNUM.sub(" ", part).split()
        if not ptoks:
            continue
        pj = " ".join(ptoks)
        if pj in drop:            # learned admin part (e.g. French region / department)
            continue
        st = smap.get(pj) or alias.get(pj)
        if st and not state:
            state = st
            tokens.append(st)
            continue
        has_digit = any(c.isdigit() for c in pj)
        ploc: list[str] = []
        for t in ptoks:
            if any(c.isdigit() for c in t):
                ordinal = bool(_ORDINAL.match(t))
                for n in _NUM.findall(t):
                    n = n.lstrip("0") or "0"
                    nums.append(n)
                    if not hno and not ordinal:
                        hno = n
                tokens.append(t)
                continue
            t = ABBR.get(t, t)
            if t in ADDR_MARKERS:
                continue
            tokens.append(t)
            if t in _TYPES or len(t) < 2:
                continue
            (street if has_digit else ploc).append(t)
        if ploc:
            loc_parts.append(ploc)
    # locality tokens ordered from the end of the address (city/district sit near the end)
    loc = [t for p in reversed(loc_parts) for t in p]
    if not hno and nums:
        hno = nums[0]
    return tokens, list(dict.fromkeys(nums)), hno, state, loc, street


def normalize_record(name, addr, country):
    name_norm, core, legal = normalize_name(name)
    toks, nums, hno, state, loc, street = normalize_address(addr, country)
    skel = [skeleton(t) for t in core if not t.isdigit()]
    return (name_norm, core, legal, skel, " ".join(toks), toks, nums, hno, state,
            list(dict.fromkeys(loc)), list(dict.fromkeys(street)))
