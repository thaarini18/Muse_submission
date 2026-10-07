"""Text normalisation for business names and addresses.

Everything here is rule-based and language-agnostic enough to handle
US / India / France (and other unseen countries) without per-country code paths:
  * any script -> ASCII via `anyascii` (ISC licence, local, no lookups)
  * abbreviation canonicalisation (street types, legal forms, states)
  * a crude phonetic "skeleton" so transliterated Indic names line up with
    their Latin spellings (e.g. 'laiph investmemts' ~ 'life investments').
"""
import re
from functools import lru_cache
from anyascii import anyascii

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_DIGITS = re.compile(r"\d+")

# ---------------------------------------------------------------- names
LEGAL = {
    # generic / US
    "inc", "incorporated", "corp", "corporation", "co", "company", "llc", "llp", "lp", "ltd",
    "limited", "plc", "pc", "pllc", "pa", "psc", "lc", "l", "c", "p", "incorporation",
    # India
    "pvt", "private", "prvt", "pvtltd", "opc",
    # France / EU
    "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "scop", "scp", "selarl", "gmbh", "ag", "bv", "nv", "srl",
}
HONORIFIC = {"the", "sri", "shri", "shree", "smt", "dr", "mr", "mrs", "ms", "m", "s", "la", "le", "les",
             "a", "an", "of", "and", "et", "de", "des", "du", "d"}
# tokens that introduce an alias (DBA / formerly / trading as)
ALIAS_MARKERS = [" doing business as ", " dba ", " formerly known as ", " formerly ", " fka ",
                 " aka ", " also known as ", " trading as ", " t a ", " previously "]
# generic business words: kept for features, down-weighted by IDF for blocking
GENERIC = {"services", "service", "center", "centre", "group", "enterprises", "enterprise", "holdings",
           "holding", "solutions", "international", "industries", "traders", "trading", "ventures",
           "associates", "partners", "global", "india", "america", "usa", "france", "national"}


@lru_cache(maxsize=1 << 18)
def to_ascii(s: str) -> str:
    if not s:
        return ""
    if s.isascii():
        return s.lower()
    return anyascii(s).lower()


def basic_tokens(s: str):
    s = to_ascii(s).replace("&", " and ")
    return [t for t in _NON_ALNUM.split(s) if t]


_SKEL_SUBS = [("ph", "f"), ("sh", "s"), ("ch", "c"), ("th", "t"), ("kh", "k"), ("gh", "g"),
              ("bh", "b"), ("dh", "d"), ("jh", "j"), ("ck", "k"), ("q", "k"), ("w", "v"), ("z", "s"),
              ("x", "ks"), ("y", "i")]
_VOWELS = re.compile(r"[aeiouh]")
_REPEAT = re.compile(r"(.)\1+")
_M_BEFORE_CONS = re.compile(r"m(?=[cdfgjklqrstvxz])")


@lru_cache(maxsize=1 << 18)
def skeleton(tok: str) -> str:
    """Phonetic consonant skeleton; keeps first letter, drops vowels / h, collapses repeats."""
    if tok.isdigit() or len(tok) <= 2:
        return tok
    t = _M_BEFORE_CONS.sub("n", tok)
    for a, b in _SKEL_SUBS:
        t = t.replace(a, b)
    t = t.replace("c", "k")
    first = t[0]
    rest = _VOWELS.sub("", t[1:])
    # a leading vowel is dropped too: transliteration often changes it (universal / yunivrsl)
    if first in "aeiouhy":
        first = ""
    out = _REPEAT.sub(r"\1", first + rest)
    return out or tok[:2]


def split_alias(name_ascii: str):
    """Return list of name variants (full name plus alias parts for DBA/formerly)."""
    s = " " + name_ascii + " "
    parts = [s]
    for mk in ALIAS_MARKERS:
        if mk in s:
            a, b = s.split(mk, 1)
            parts = [a, b]
            break
    return [p.strip() for p in parts if p.strip()]


@lru_cache(maxsize=1 << 18)
def name_core_tokens(name: str):
    """Tokens of a business name with legal forms / honorifics removed."""
    toks = basic_tokens(name)
    core = [t for t in toks if t not in LEGAL and t not in HONORIFIC]
    return tuple(core) if core else tuple(toks)


# ---------------------------------------------------------------- addresses
STREET = {
    "street": "st", "st": "st", "saint": "st", "str": "st", "road": "rd", "rd": "rd", "avenue": "ave",
    "ave": "ave", "av": "ave", "drive": "dr", "dr": "dr", "court": "ct", "ct": "ct", "circle": "cir",
    "cir": "cir", "lane": "ln", "ln": "ln", "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "bld": "blvd",
    "highway": "hwy", "hwy": "hwy", "place": "pl", "pl": "pl", "parkway": "pkwy", "pkwy": "pkwy",
    "terrace": "ter", "ter": "ter", "trail": "trl", "trl": "trl", "way": "way", "square": "sq", "sq": "sq",
    "rue": "rue", "r": "rue", "allee": "all", "all": "all", "impasse": "imp", "imp": "imp", "chemin": "ch",
    "che": "ch", "route": "rte", "rte": "rte", "quai": "qu", "cours": "crs", "marg": "marg", "nagar": "ngr",
    "ngr": "ngr", "north": "n", "south": "s", "east": "e", "west": "w", "n": "n", "s": "s", "e": "e", "w": "w",
    "apartment": "apt", "apt": "apt", "suite": "ste", "ste": "ste", "unit": "unit", "floor": "fl", "fl": "fl",
    "building": "bldg", "bldg": "bldg", "number": "no", "no": "no", "house": "h", "h": "h", "hno": "h",
    "near": "nr", "nr": "nr", "opp": "opp", "opposite": "opp", "sector": "sec", "sec": "sec",
    "bis": "bis", "ter_": "ter", "etage": "fl", "eme": "", "er": "", "th": "", "nd": "", "rd_": "", "po": "po",
    "box": "box", "door": "door", "block": "blk", "blk": "blk", "plot": "plot", "shop": "shop", "flat": "flat",
    "office": "off", "off": "off", "gali": "gali", "road_": "rd",
}
# tokens that carry little address identity (used to down-weight / drop in blocking)
ADDR_STOP = {"st", "rd", "ave", "dr", "ct", "cir", "ln", "blvd", "hwy", "pl", "pkwy", "ter", "trl", "way", "sq",
             "rue", "all", "imp", "ch", "rte", "qu", "crs", "n", "s", "e", "w", "apt", "ste", "unit", "fl", "bldg",
             "no", "h", "nr", "opp", "sec", "po", "box", "door", "blk", "plot", "shop", "flat", "off", "c", "o",
             "and", "the", "de", "des", "du", "la", "le", "of", "near", "a", "b", "d", "", "na", "city", "town"}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh",
    "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "west bengal": "wb", "delhi": "dl", "chandigarh": "ch",
    "jammu and kashmir": "jk", "puducherry": "py", "pondicherry": "py",
}


@lru_cache(maxsize=1 << 18)
def addr_tokens(addr: str):
    """Canonical address tokens (abbreviations unified, ordinal suffixes split from numbers)."""
    s = to_ascii(addr).replace("&", " and ")
    s = re.sub(r"(\d+)(st|nd|rd|th|er|eme|e)\b", r"\1", s)  # 152nd -> 152, 2eme -> 2
    out = []
    for t in _NON_ALNUM.split(s):
        if not t:
            continue
        c = STREET.get(t)
        if c is not None:
            if c:
                out.append(c)
            continue
        # split alnum like 'b20' / '78c' into number part and letters
        if not t.isdigit() and not t.isalpha():
            nums = _DIGITS.findall(t)
            letters = _DIGITS.sub(" ", t).split()
            out.extend(nums)
            out.extend(l for l in letters if len(l) > 1)
            continue
        out.append(t)
    return tuple(out)


def addr_numbers(tokens):
    return [t for t in tokens if t.isdigit()]


def addr_words(tokens):
    return [t for t in tokens if not t.isdigit() and t not in ADDR_STOP]
