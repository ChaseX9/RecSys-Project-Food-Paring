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


# --------------------------------------------------------------- scoring options
import scipy.sparse as sp

from src.ranking import RankingConfig


def _tiny(config, cooc=None):
    vocab = pd.DataFrame({"ingredient_id": range(4), "name": ["tomato", "basil", "rare herb", "garlic"]})
    # basil: same direction as tomato, large norm (popular); rare herb: same direction, small norm
    emb = np.array([[1.0, 0.0], [3.0, 0.2], [0.3, 0.0], [0.0, 1.0]])
    return HybridRanker(emb, vocab, config=config, cooccurrence=cooc)


def test_cosine_ignores_norm_and_dot_keeps_it():
    only_emb = dict(beta_chem=0.0, gamma_pref=0.0)
    cos = _tiny(RankingConfig(emb_similarity="cosine", **only_emb)).recommend(["tomato"], top_k=3)
    dot = _tiny(RankingConfig(emb_similarity="dot", **only_emb)).recommend(["tomato"], top_k=3)
    assert cos[0]["name"] == "rare herb"        # cosine 1.0 beats basil's 0.998
    assert dot[0]["name"] == "basil"            # dot keeps the popularity carried by the norm


def test_normalize_puts_components_in_unit_range():
    recs = _tiny(RankingConfig(emb_similarity="dot", normalize=True, beta_chem=0.0,
                               gamma_pref=0.0)).recommend(["tomato"], top_k=3)
    assert all(0.0 <= r["emb_score"] <= 1.0 for r in recs)
    assert recs[0]["emb_score"] == pytest.approx(1.0)


def test_cooccurrence_term_reranks():
    cooc = sp.csr_matrix(np.array([[0, 0, 0, 50], [0, 0, 0, 0], [0, 0, 0, 0], [50, 0, 0, 0]], dtype=float))
    cfg = RankingConfig(alpha_emb=0.0, beta_chem=0.0, gamma_pref=0.0, delta_cooc=1.0, normalize=True)
    assert _tiny(cfg, cooc).recommend(["tomato"], top_k=1)[0]["name"] == "garlic"


def test_unknown_similarity_rejected():
    with pytest.raises(ValueError):
        _tiny(RankingConfig(emb_similarity="euclid"))
