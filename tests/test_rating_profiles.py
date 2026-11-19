import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.data.rating_profiles import RatingProfiles, user_recipe_weights
from src.ranking import HybridRanker, RankingConfig


def test_centered_shrunk_weights():
    # user 1 rates everything 5 -> small positive weights, never an empty profile
    # user 2: a 4 from a generous user is NOT a dislike, a 2 is
    train = pd.DataFrame({"user_id": [1, 1, 2, 2, 2], "recipe_id": [10, 11, 10, 11, 12],
                          "rating": [5, 5, 5, 4, 2]})
    uids, W, n = user_recipe_weights(train, {10: 0, 11: 1, 12: 2}, 3, shrink_k=5, dislike_max_rating=3)
    W = W.toarray()
    assert uids.tolist() == [1, 2] and n.tolist() == [2, 3]
    assert (W[0, :2] > 0).all()
    assert W[1, 1] == 0.0          # 4/5 below the user's mean, but above dislike_max_rating
    assert W[1, 2] < 0             # 2/5 -> dislike
    assert W[1, 0] > 0


def _profiles():
    # 2 users x 4 ingredients; user 7 likes ingredient 1, dislikes ingredient 2
    like = sp.csr_matrix(np.array([[0, 1.0, 0, 0], [0, 0, 0, 1.0]]))
    dislike = sp.csr_matrix(np.array([[0, 0, 1.0, 0], [0, 0, 0, 0]]))
    signal = sp.csr_matrix(np.array([[0, 1.0, -1.0, 0], [0, 0, 0, 1.0]]))
    return RatingProfiles(np.array([7, 8]), like, dislike, signal, np.array([1.0, 1.0]), np.array([20, 20]))


def _ranker(**cfg):
    vocab = pd.DataFrame({"ingredient_id": range(4), "name": ["tomato", "basil", "cilantro", "garlic"]})
    emb = np.array([[1.0, 0.0], [0.9, 0.1], [0.9, 0.1], [0.0, 1.0]])
    base = dict(beta_chem=0.0, emb_similarity="cosine", normalize=True, pref_source="rating")
    base.update(cfg)
    return HybridRanker(emb, vocab, config=RankingConfig(**base), rating_profiles=_profiles())


def test_rating_preference_signs():
    pref = _ranker().preference(7)
    assert pref[1] > 0 and pref[2] < 0
    assert not _ranker().preference(999).any()          # unknown user -> no preference
    assert not _ranker().preference("app-user").any()   # non-numeric app id -> no preference


def test_like_and_dislike_reorder_equal_pairings():
    # basil and cilantro are tied on pairing; user 7 likes basil, dislikes cilantro
    recs = [r["name"] for r in _ranker(gamma_pref=0.5).recommend(["tomato"], user_id=7, top_k=3)]
    assert recs.index("basil") < recs.index("cilantro")


def test_multiplicative_keeps_poor_pairings_poor():
    # garlic is orthogonal to tomato (pairing 0): even a strong taste cannot lift it
    r = _ranker(gamma_pref=5.0, pref_mode="multiplicative")
    assert r.recommend(["tomato"], user_id=8, top_k=3)[-1]["name"] == "garlic"
    a = _ranker(gamma_pref=5.0, pref_mode="additive")
    assert a.recommend(["tomato"], user_id=8, top_k=1)[0]["name"] == "garlic"


def test_embedding_user_vector_shape():
    rp = _profiles()
    u = rp.user_vectors(np.eye(4))
    assert u.shape == (2, 4)
    assert np.linalg.norm(u, axis=1) == pytest.approx([1.0, 1.0])
    assert u[0, 1] > 0 and u[0, 2] < 0          # liked direction +, disliked direction -


def test_rating_source_requires_profiles():
    vocab = pd.DataFrame({"ingredient_id": range(2), "name": ["a", "b"]})
    with pytest.raises(ValueError):
        HybridRanker(np.eye(2), vocab, config=RankingConfig(pref_source="rating"))
