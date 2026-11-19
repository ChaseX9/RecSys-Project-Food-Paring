from __future__ import annotations

import re
from pathlib import Path

import yaml

_R_ITEM = re.compile(r'"((?:[^"\\]|\\.)*)"')
_PAREN = re.compile(r"\([^)]*\)")
_NON_ALPHA = re.compile(r"[^a-z\- ]")

# Preparation / state words that do not change the flavor of an ingredient
DESCRIPTORS = {
    "fresh", "freshly", "chopped", "minced", "sliced", "diced", "grated", "shredded", "crushed",
    "large", "small", "medium", "extra", "extra-large", "whole", "optional", "divided", "packed",
    "softened", "melted", "cooked", "uncooked", "raw", "frozen", "canned", "dried", "boneless",
    "skinless", "lean", "thinly", "finely", "coarsely", "roughly", "ripe", "unsalted", "salted",
    "organic", "good", "quality", "pure", "real", "prepared", "drained", "rinsed", "peeled",
    "seeded", "cubed", "halved", "sifted", "toasted", "roasted",
    "light", "lite", "lowfat", "nonfat", "jumbo",
}

# Multi-word modifiers that do not change the flavor (diet versions, cooking state).
# "vegan" / "vegetarian" are NOT here: they change what the diet filter must do.
_MODIFIER_PHRASES = re.compile(
    r"\b(?:low[- ]?sodium|reduced[- ]sodium|no[- ]salt[- ]added|no[- ]added[- ]salt|fat[- ]free"
    r"|low[- ]?fat|reduced[- ]fat|non[- ]?fat|reduced[- ]calorie|hard[- ](?:boiled|cooked))\b")

# Brand names stripped from the start of an ingredient ("heinz ketchup" -> "ketchup").
# Apostrophes are already turned into spaces ("hellmann's" -> "hellmann s").
# Products that ARE the brand (cool whip, bisquick, tabasco, velveeta) are not listed.
_BRANDS = ("hellmann s", "hellmanns", "best foods", "heinz", "kikkoman", "philadelphia",
           "lea perrins", "grey poupon", "gold medal", "swanson", "hunt s", "hunts", "ragu",
           "planters", "nestle", "uncle ben s", "uncle bens")
_BRAND_PREFIX = re.compile(r"^(?:" + "|".join(re.escape(b) for b in _BRANDS) + r")\s+")

_IRREGULAR = {"leaves": "leaf", "halves": "half", "loaves": "loaf", "tomatoes": "tomato",
              "potatoes": "potato", "mangoes": "mango", "avocados": "avocado"}
_KEEP_S = {"molasses", "hummus", "couscous", "asparagus", "grits", "swiss", "watercress",
           "cress", "citrus", "lemongrass", "bass", "series", "species"}


def parse_r_vector(value) -> list[str]:
    """Turn an R vector string like 'c("a", "b")' into ['a', 'b']."""
    if not isinstance(value, str):
        return []
    return [m.replace('\\"', '"') for m in _R_ITEM.findall(value)]


def _singularize(word: str) -> str:
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if word in _KEEP_S or len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "sses", "xes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def load_aliases(path: str | Path | None) -> dict[str, str]:
    if not path or not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def normalize_ingredient(raw: str, aliases: dict[str, str] | None = None) -> str:
    """Canonical form of a raw ingredient name, or '' if nothing usable is left."""
    text = raw.lower()
    text = _PAREN.sub(" ", text)
    text = text.split(",")[0]  # "lemons, rind of" -> "lemons"
    text = _NON_ALPHA.sub(" ", text)
    text = " ".join(text.split())
    text = _MODIFIER_PHRASES.sub(" ", text).strip()
    text = _BRAND_PREFIX.sub("", " ".join(text.split()))
    tokens = [t.strip("-") for t in text.split()]
    tokens = [t for t in tokens if t and t not in DESCRIPTORS]
    if not tokens:
        return ""
    tokens[-1] = _singularize(tokens[-1])
    name = " ".join(tokens)
    if aliases:
        name = aliases.get(name, name)
    return name


def normalize_recipe(raw_vector, aliases: dict[str, str] | None = None) -> list[str]:
    """Deduplicated list of canonical ingredients, order preserved."""
    seen, out = set(), []
    for raw in parse_r_vector(raw_vector):
        name = normalize_ingredient(raw, aliases)
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out
