import numpy as np
import pandas as pd
import pytest

from src.ranking import HybridRanker

VOCAB = [
    # nuts and look-alikes
    "walnut", "peanut butter", "pine nut", "pine nuts", "hazelnut", "cashew",
    "nutmeg", "coconut", "water chestnut", "nutritional yeast", "minute rice",
    # eggs and look-alikes
    "egg", "hard-boiled egg", "taleggio", "eggplant",
    # meat / fish and look-alikes
    "ham", "chicken breast", "salmon", "graham cracker",
    # seafood, cured meat, other animal-derived products
    "sea scallop", "pancetta", "unflavored gelatin", "worcestershire sauce", "mahi mahi fillet",
    "gooseberry",
    # gluten
    "buckwheat noodle", "buckwheat groat",
    # other animal products
    "butter", "honey",
    # neutral query ingredient
    "tomato",
]


@pytest.fixture
def ranker():
    rng = np.random.default_rng(0)
    vocab = pd.DataFrame({"ingredient_id": range(len(VOCAB)), "name": VOCAB})
    return HybridRanker(rng.normal(size=(len(VOCAB), 8)), vocab)


def _names(ranker, **filters):
    recs = ranker.recommend(["tomato"], top_k=len(VOCAB), **filters)
    return {r["name"] for r in recs}


def test_nut_allergy_whole_word(ranker):
    kept = _names(ranker, allergies=["nut"])
    for banned in ("walnut", "peanut butter", "pine nut", "pine nuts", "hazelnut", "cashew"):
        assert banned not in kept
    for ok in ("nutmeg", "coconut", "water chestnut", "nutritional yeast", "minute rice"):
        assert ok in kept


def test_egg_allergy_whole_word(ranker):
    kept = _names(ranker, allergies=["egg"])
    assert "egg" not in kept
    assert "hard-boiled egg" not in kept
    assert "taleggio" in kept
    assert "eggplant" in kept


def test_vegetarian(ranker):
    kept = _names(ranker, diet="vegetarian")
    for banned in ("ham", "chicken breast", "salmon"):
        assert banned not in kept
    assert "graham cracker" in kept


def test_vegetarian_seafood_cured_meat_and_animal_derived(ranker):
    kept = _names(ranker, diet="vegetarian")
    for banned in ("sea scallop", "pancetta", "unflavored gelatin", "worcestershire sauce",
                   "mahi mahi fillet"):
        assert banned not in kept
    assert "gooseberry" in kept  # "goose" matches whole words only


def test_vegan_is_stricter_than_vegetarian(ranker):
    kept = _names(ranker, diet="vegan")
    for banned in ("sea scallop", "pancetta", "unflavored gelatin", "worcestershire sauce"):
        assert banned not in kept


def test_gluten_buckwheat_noodle(ranker):
    kept = _names(ranker, allergies=["gluten"])
    assert "buckwheat noodle" not in kept
    assert "buckwheat groat" in kept


def test_vegan(ranker):
    kept = _names(ranker, diet="vegan")
    for banned in ("butter", "egg", "honey", "ham", "salmon"):
        assert banned not in kept
    assert "peanut butter" in kept


def test_vegan_with_nut_allergy_still_bans_peanut_butter(ranker):
    kept = _names(ranker, diet="vegan", allergies=["nut"])
    assert "peanut butter" not in kept


def test_no_filters_keeps_everything_but_query(ranker):
    assert _names(ranker) == set(VOCAB) - {"tomato"}
