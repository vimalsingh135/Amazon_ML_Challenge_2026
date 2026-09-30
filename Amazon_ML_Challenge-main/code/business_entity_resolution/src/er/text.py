"""Scalar text utilities: script folding, Indic transliteration, phonetic skeletons.

These operate on single strings and are only applied to the (minority of) values that
need them; bulk normalisation is vectorised in ``er.prep`` with Polars.
"""
from __future__ import annotations

import re
import unicodedata

from anyascii import anyascii

# ---------------------------------------------------------------------------
# Indic transliteration. All Brahmic scripts in Unicode share the Devanagari
# layout at a fixed block offset, so we fold every script onto Devanagari
# offsets and transliterate with a single table.
# ---------------------------------------------------------------------------
_INDIC_BLOCKS = (0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00)

_CONS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh",
    0x1C: "j", 0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh",
    0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r",
    0x31: "r", 0x32: "l", 0x33: "l", 0x34: "zh", 0x35: "v", 0x36: "sh", 0x37: "sh",
    0x38: "s", 0x39: "h",
    0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "d", 0x5D: "dh", 0x5E: "f", 0x5F: "y",
}
_VOWELS = {
    0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri",
    0x0C: "l", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o",
    0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "l",
}
_MATRAS = {
    0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri",
    0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o",
    0x4C: "au", 0x62: "l", 0x63: "l",
}
_VIRAMA, _NUKTA = 0x4D, 0x3C
_NASAL = {0x01: "n", 0x02: "n", 0x03: "h"}


def _indic_offset(ch: str) -> int | None:
    cp = ord(ch)
    for base in _INDIC_BLOCKS:
        if base <= cp < base + 0x80:
            return cp - base
    return None


def transliterate_indic(text: str) -> str:
    """Rule-based transliteration of any Brahmic script to lowercase Latin.

    Implements inherent-vowel insertion with word-final schwa deletion, which is
    what romanised Indian business names typically look like (राम -> ram).
    """
    out: list[str] = []
    pending_schwa = False  # a consonant was emitted and may take an inherent 'a'
    conjunct = False  # current consonant closes a conjunct cluster (keeps final 'a')
    after_virama = False
    for ch in text:
        off = _indic_offset(ch)
        if off is None:
            if ch in "‌‍":
                continue
            if pending_schwa and not ch.isalpha():
                if conjunct:
                    out.append("a")  # aditya, not adity
                pending_schwa = False  # word end: drop schwa
            out.append(ch)
            continue
        if off in _CONS:
            if pending_schwa:
                out.append("a")
            out.append(_CONS[off])
            pending_schwa = True
            conjunct = after_virama
            after_virama = False
        elif off in _MATRAS:
            out.append(_MATRAS[off])
            pending_schwa = False
        elif off == _VIRAMA:
            pending_schwa = False
            after_virama = True
        elif off == _NUKTA:
            continue
        elif off in _VOWELS:
            if pending_schwa:
                out.append("a")
            out.append(_VOWELS[off])
            pending_schwa = False
        elif off in _NASAL:
            if pending_schwa:
                out.append("a")
            out.append(_NASAL[off])
            pending_schwa = False
        elif 0x66 <= off <= 0x6F:
            if pending_schwa:
                out.append("a")
            out.append(str(off - 0x66))
            pending_schwa = False
        else:  # danda, avagraha, misc signs
            if pending_schwa:
                pending_schwa = False
            out.append(" ")
    if pending_schwa and conjunct:
        out.append("a")
    return "".join(out)


_INDIC_RE = re.compile("[ऀ-ൿ]")


def to_ascii(text: str) -> str:
    """Fold any string to lowercase ASCII (accents stripped, Indic transliterated)."""
    if text.isascii():
        return text.lower()
    if _INDIC_RE.search(text):
        text = transliterate_indic(text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    if not text.isascii():
        text = anyascii(text)
    return text.lower()


# ---------------------------------------------------------------------------
# Phonetic consonant skeleton: robust to vowel noise, transliteration and
# aspiration differences (praivet ~ private -> "prvt").
# ---------------------------------------------------------------------------
_SKEL_SUBS = (
    ("ph", "f"), ("ck", "k"), ("x", "ks"), ("q", "k"), ("w", "v"), ("z", "j"), ("b", "v"), ("g", "j"),
)
_C_SOFT = re.compile(r"c(?=[eiy])")
_G_SOFT = re.compile(r"g(?=[ei])")
_VOWEL_H = re.compile(r"[aeiouyh]")
_REPEAT = re.compile(r"(.)\1+")


def skeleton(token: str) -> str:
    t = token
    for a, b in _SKEL_SUBS:
        t = t.replace(a, b)
    t = _C_SOFT.sub("s", t)
    t = t.replace("c", "k")
    head, tail = t[:1], _VOWEL_H.sub("", t[1:])
    if head in "aeiouyh":
        head = ""
    t = _REPEAT.sub(r"\1", head + tail)
    return t
