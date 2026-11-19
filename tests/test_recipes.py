"""Recipe ratings in the app: catalog, storage, and their effect on the taste profile."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from fastapi.testclient import TestClient

from api import recipes
from api.main import create_app
from api.recommender import ModelRecommender
from src.data.rating_profiles import RatingData, profiles_from_weights
from src.ranking import HybridRanker, RankingConfig

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
VOCAB = ["tomato", "basil", "oregano", "cilantro", "cumin", "garlic", "mozzarella cheese", "flour"]
V = {n: i for i, n in enumerate(VOCAB)}
RECIPES = [  # recipe_id, name, ingredients, n train ratings
    (1, "Margherita Pizza", ["tomato", "basil", "oregano", "mozzarella cheese", "flour"], 50),
    (2, "Chicken Curry", ["cilantro", "cumin", "garlic"], 40),          # meat only in the title
    (3, "Tomato &amp; Basil Pasta Sauce", ["tomato", "garlic", "basil"], 30),
    (4, "Fresh Salsa", ["tomato", "cilantro", "cumin"], 20),
    (5, "Garlic Bread", ["garlic", "flour", "oregano"], 10),
    (6, "Two Things", ["tomato", "basil"], 5),                          # < 3 ingredients
]


def make_catalog() -> recipes.RecipeCatalog:
    df = pd.DataFrame({"recipe_id": [r[0] for r in RECIPES], "name": [r[1] for r in RECIPES],
                       "ingredient_ids": [[V[i] for i in r[2]] for r in RECIPES]})
    ids = [r[0] for r in RECIPES]
    return recipes.RecipeCatalog(df, VOCAB, pd.Series([r[3] for r in RECIPES], index=ids),
                                 pd.Series([4.5] * len(ids), index=ids))


def make_model() -> ModelRecommender:
    rows, cols = [], []
    for k, r in enumerate(RECIPES):
        for name in r[2]:
            rows.append(k); cols.append(V[name])
    ri = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(RECIPES), len(VOCAB)))
    data = RatingData(ri=ri, recipe_index={r[0]: k for k, r in enumerate(RECIPES)},
                      idf=np.ones(len(VOCAB)), global_mean=4.5,
                      train_ratings=sp.csr_matrix((0, len(RECIPES))), train_user_row={},
                      shrink_k=5.0, dislike_below_rating=3.0, conf_k=10.0, alpha=2.0)
    # one Food.com-like user with no ratings, only so the "rating" source has its matrices
    empty = profiles_from_weights([999], sp.csr_matrix((1, len(RECIPES))), np.array([0]), ri,
                                  np.ones(len(VOCAB)))
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(len(VOCAB), 4))
    cfg = RankingConfig(beta_chem=0.0, gamma_pref=0.5, normalize=True, pref_source="rating",
                        pref_mode="multiplicative", profile_combine="net")
    vocab = pd.DataFrame({"ingredient_id": range(len(VOCAB)), "name": VOCAB})
    ranker = HybridRanker(emb, vocab, config=cfg, rating_profiles=empty, rating_data=data)
    return ModelRecommender(processed_dir=Path("does-not-exist"), ranker=ranker)


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(str(tmp_path / "r.db"), recommender=make_model(),
                               catalog=make_catalog())) as c:
        yield c


# ------------------------------------------------------------------ catalog
def test_search_matches_every_word_most_rated_first():
    cat = make_catalog()
    assert [r["name"] for r in cat.search("tomato")] == ["Tomato & Basil Pasta Sauce"]  # unescaped
    assert [r["recipe_id"] for r in cat.search("sa")] == [3, 4]          # most rated first
    assert cat.search("pizza margherita")[0]["ingredients"][:2] == ["tomato", "basil"]
    assert cat.search("   ") == []


def test_title_rules_catch_meat_missing_from_ingredients():
    cat = make_catalog()
    veg = recipes.Ban(title=recipes.title_ban("vegetarian"))
    assert 2 not in [r["recipe_id"] for r in cat.search("curry", ban=veg)]
    assert recipes.title_ban(None) is None
    for title in ("porketta roast", "bbq ribs", "buffalo wings", "baked salmon", "cod cakes"):
        assert recipes.title_ban("vegetarian").search(title), title
    assert not recipes.title_ban("pescatarian").search("baked salmon")
    for title in ("ribbon cake", "codfish-free salad", "chickpea curry", "graham crackers"):
        assert not recipes.title_ban("vegetarian").search(title), title


def test_onboarding_is_popular_varied_and_skips_rated():
    cat = make_catalog()
    ids = [r["recipe_id"] for r in cat.onboarding(n=3)]
    assert ids[0] == 1 and len(set(ids)) == 3 and 6 not in ids   # most rated first, no tiny recipe
    assert 2 in ids                      # nothing in common with the pizza: picked early
    assert 1 not in [r["recipe_id"] for r in cat.onboarding(rated=[1], n=3)]


def test_next_teaches_new_ingredients_and_varies():
    cat = make_catalog()
    card = cat.next_to_rate(rated=[1, 3], seed=1, top=1)
    assert card["recipe_id"] in (2, 4) and "cumin" in card["new_ingredients"]
    assert cat.next_to_rate(rated=[1, 2, 3, 4], skip=[5]) is None
    seen = {cat.next_to_rate(seed=s)["recipe_id"] for s in range(20)}
    assert len(seen) > 1


# -------------------------------------------------------------------- API
def test_rate_list_update_delete(client):
    assert client.put("/recipe_ratings/lea/1", json={"rating": 5}).status_code == 204
    assert client.put("/recipe_ratings/lea/1", json={"rating": 4}).status_code == 204  # replaces
    assert client.put("/recipe_ratings/lea/2", json={"rating": 6}).status_code == 422
    assert client.put("/recipe_ratings/lea/42", json={"rating": 3}).status_code == 404
    rows = client.get("/recipe_ratings/lea").json()
    assert [(r["recipe_id"], r["user_rating"], r["name"]) for r in rows] == [(1, 4, "Margherita Pizza")]
    assert client.get("/recipes/search", params={"q": "pizza", "user_id": "lea"}).json()[0]["user_rating"] == 4
    assert client.delete("/recipe_ratings/lea/1").status_code == 204
    assert client.delete("/recipe_ratings/lea/1").status_code == 404
    assert client.get("/recipe_ratings/lea").json() == []


def test_routes_apply_diet_and_skip_rated(client):
    client.put("/recipe_ratings/lea/1", json={"rating": 5})
    ob = client.get("/recipes/onboarding", params={"user_id": "lea", "diet": "vegetarian"}).json()
    assert {r["recipe_id"] for r in ob} == {3, 4, 5}            # no pizza (rated), no curry (meat)
    nxt = client.get("/recipes/next", params={"user_id": "lea", "skip": [3, 4], "seed": 0}).json()
    assert nxt["recipe_id"] in (2, 5)
    assert client.get("/recipes/next", params={"user_id": "lea", "skip": [2, 3, 4, 5]}).status_code == 404


def test_allergic_feedback_also_filters_recipes(client):
    client.post("/log_interaction", json={"user_id": "lea", "query_ingredients": ["salad"],
                                          "recommended_ingredient": "cumin", "feedback": "allergic"})
    ids = {r["recipe_id"] for r in client.get("/recipes/onboarding", params={"user_id": "lea"}).json()}
    assert 2 not in ids and 4 not in ids


def test_ratings_shape_the_profile_and_the_recommendations(client):
    def scores():
        body = client.post("/recommend", json={"user_id": "lea", "ingredients": ["tomato"],
                                               "top_k": 7}).json()
        return {r["name"]: r["pref_score"] for r in body["recommendations"]}

    assert set(scores().values()) == {0.0}                      # nothing known yet
    assert client.get("/profile/lea").json()["has_rating_history"] is False
    client.put("/recipe_ratings/lea/1", json={"rating": 5})     # loves the pizza
    client.put("/recipe_ratings/lea/2", json={"rating": 1})     # hates the curry
    after = scores()
    assert after["basil"] > 0 and after["oregano"] > 0 and after["cumin"] < 0 and after["cilantro"] < 0
    prof = client.get("/profile/lea").json()
    assert prof["has_rating_history"] and prof["n_recipe_ratings"] == 2
    assert "basil" in {d["name"] for d in prof["likes"]} and "cumin" in {d["name"] for d in prof["dislikes"]}
    assert set(scores(  # another user is not affected
    ).keys()) and client.post("/recommend", json={"user_id": "other", "ingredients": ["tomato"]}
                              ).json()["recommendations"][0]["pref_score"] == 0.0


def test_mock_and_no_catalog(tmp_path):
    with TestClient(create_app(str(tmp_path / "m.db"), recommender=None, catalog=None)) as c:
        assert c.get("/recipes/search", params={"q": "pizza"}).status_code == 503
        assert c.get("/recipe_ratings/lea").json() == []


def test_deleting_an_account_deletes_its_recipe_ratings(client):
    client.put("/recipe_ratings/camille/1", json={"rating": 5})
    assert client.delete("/admin/users/camille", params={"actor": "admin"}).status_code == 204
    assert client.get("/recipe_ratings/camille").json() == []


# ------------------------------------------------------- same formula as offline
@pytest.mark.skipif(not (PROCESSED / "interactions_train.parquet").exists(),
                    reason="needs data/processed")
def test_profile_from_replayed_ratings_equals_the_precomputed_one():
    from src.data.rating_profiles import build_rating_profiles, load_rating_data

    rp = build_rating_profiles(PROCESSED, alpha=5.0)
    data = load_rating_data(PROCESSED, alpha=5.0)
    for row in (0, len(rp.user_ids) // 2):
        uid = int(rp.user_ids[row])
        one = data.profile_for(data.ratings_of(uid))
        a = rp.net.getrow(row).toarray().ravel()
        b = one.net.getrow(0).toarray().ravel()
        assert np.abs(a - b).max() < 1e-9 and one.confidence[0] == pytest.approx(rp.confidence[row])
