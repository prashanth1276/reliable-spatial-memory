"""Natural-language command parser using spaCy."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import re

try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
    _SPACY_OK = True
except Exception:
    _NLP = None
    _SPACY_OK = False


COLORS = {
    "red", "blue", "green", "yellow", "black", "white",
    "brown", "orange", "purple", "pink", "gray", "grey",
}

# Keywords that map to the object type we actually store
OBJECT_KEYWORDS = {
    "mug": "Mug", "cup": "Mug",
    "red mug": "RedMug", "red cup": "RedMug", "redmug": "RedMug",
    "apple": "Apple",
    "table": "Table", "dining table": "Table",
    "counter": "Counter", "countertop": "Counter",
    "fridge": "Fridge", "refrigerator": "Fridge",
    "sofa": "Sofa", "couch": "Sofa",
    "tv": "TV", "television": "TV",
    "coffee table": "CoffeeTable",
}

ACTION_KEYWORDS = {
    "go": "navigate", "navigate": "navigate", "move": "navigate",
    "reach": "navigate", "head": "navigate", "walk": "navigate",
    "find": "find", "locate": "find", "search": "find",
    "bring": "fetch", "fetch": "fetch", "get": "fetch",
}

SPATIAL_PREPS = {"near", "beside", "next to", "by", "on", "under"}


@dataclass
class ParsedCommand:
    action: str
    object_type: Optional[str]
    color: Optional[str] = None
    spatial_relation: Optional[Tuple[str, str]] = None  # (prep, anchor_type)
    raw: str = ""
    parse_succeeded: bool = False


# ---------- spaCy-based parser ----------

def _parse_spacy(text: str) -> ParsedCommand:
    doc = _NLP(text.lower())
    tokens = [t for t in doc if not t.is_punct and not t.is_space]

    action = "navigate"
    obj_type: Optional[str] = None
    color: Optional[str] = None
    spatial: Optional[Tuple[str, str]] = None

    # 1. Action: first verb
    for t in tokens:
        if t.pos_ == "VERB" and t.lemma_ in ACTION_KEYWORDS:
            action = ACTION_KEYWORDS[t.lemma_]
            break
        if t.pos_ == "VERB" and t.text in ACTION_KEYWORDS:
            action = ACTION_KEYWORDS[t.text]
            break

    # 2. Color: first adjective that's in COLORS
    for t in tokens:
        if t.pos_ == "ADJ" and t.lemma_.lower() in COLORS:
            color = t.lemma_.lower()
            break

    # 3. Object type: first noun that maps to a known object
    for t in tokens:
        if t.pos_ in ("NOUN", "PROPN"):
            key = t.lemma_.lower()
            if key in OBJECT_KEYWORDS:
                obj_type = OBJECT_KEYWORDS[key]
                break

    # 4. Spatial relation: prep + following noun
    for i, t in enumerate(tokens):
        if t.pos_ == "ADP" and t.text in SPATIAL_PREPS:
            for j in range(i + 1, min(i + 4, len(tokens))):
                if tokens[j].pos_ in ("NOUN", "PROPN"):
                    anchor_key = tokens[j].lemma_.lower()
                    if anchor_key in OBJECT_KEYWORDS:
                        spatial = (t.text, OBJECT_KEYWORDS[anchor_key])
                        break
            if spatial:
                break

    return ParsedCommand(
        action=action,
        object_type=obj_type,
        color=color,
        spatial_relation=spatial,
        raw=text,
        parse_succeeded=obj_type is not None,
    )


# ---------- Regex fallback ----------

def _parse_regex(text: str) -> ParsedCommand:
    t = text.lower().strip()
    action = "navigate"
    for kw, a in ACTION_KEYWORDS.items():
        if kw in t:
            action = a
            break

    color = None
    for c in COLORS:
        if re.search(rf"\b{c}\b", t):
            color = c
            break

    obj_type = None
    # longest-first so "coffee table" beats "table"
    for key in sorted(OBJECT_KEYWORDS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(key)}s?\b", t):
            obj_type = OBJECT_KEYWORDS[key]
            break

    spatial = None
    for prep in SPATIAL_PREPS:
        m = re.search(rf"\b{re.escape(prep)}\s+the\s+([a-z ]+)", t)
        if m:
            anchor = m.group(1).strip()
            for key in sorted(OBJECT_KEYWORDS, key=len, reverse=True):
                if key in anchor:
                    spatial = (prep, OBJECT_KEYWORDS[key])
                    break
        if spatial:
            break

    return ParsedCommand(
        action=action,
        object_type=obj_type,
        color=color,
        spatial_relation=spatial,
        raw=text,
        parse_succeeded=obj_type is not None,
    )


def parse_command(text: str) -> ParsedCommand:
    """Parse a natural-language command. Uses spaCy, falls back to regex."""
    if _SPACY_OK:
        try:
            result = _parse_spacy(text)
        except Exception:
            result = _parse_regex(text)
    else:
        result = _parse_regex(text)

    # ---- Post-parse safety net for color extraction ----
    # spaCy sometimes tags colors as NOUN or misses them in short commands.
    if result.color is None:
        t = text.lower()
        for c in sorted(COLORS, key=len, reverse=True):
            import re as _re
            if _re.search(rf"\b{c}\b", t):
                result.color = c
                break

    return result