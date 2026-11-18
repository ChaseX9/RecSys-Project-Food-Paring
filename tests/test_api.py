import pytest
from fastapi.testclient import TestClient

from api.main import create_app


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(str(tmp_path / "test.db"), recommender=None)) as c:  # mock contract
        yield c


def test_log_and_read_interaction(client):
    payload = {"user_id": "u1", "query_ingredients": ["salmon"],
               "recommended_ingredient": "dill", "feedback": "like"}
    r = client.post("/log_interaction", json=payload)
    assert r.status_code == 201 and r.json()["status"] == "logged"

    rows = client.get("/interactions", params={"user_id": "u1"}).json()
    assert len(rows) == 1
    assert rows[0]["query_ingredients"] == ["salmon"] and rows[0]["feedback"] == "like"
    assert client.get("/interactions", params={"user_id": "other"}).json() == []


def test_rejects_invalid_feedback_and_empty_query(client):
    bad_feedback = {"user_id": "u1", "query_ingredients": ["salmon"], "feedback": "love"}
    assert client.post("/log_interaction", json=bad_feedback).status_code == 422
    empty_query = {"user_id": "u1", "query_ingredients": [], "feedback": "like"}
    assert client.post("/log_interaction", json=empty_query).status_code == 422


def test_free_text_cooked_log(client):
    payload = {"user_id": "u2", "query_ingredients": ["chicken", "lemon"],
               "feedback": "cooked", "note": "roasted with thyme"}
    assert client.post("/log_interaction", json=payload).status_code == 201


REC_KEYS = {"rank", "name", "score", "emb_score", "chem_score", "pref_score", "explanation", "feedback"}


def test_recommend_format_and_top_k(client):
    payload = {"user_id": "u1", "ingredients": ["Garlic", "lemon"], "top_k": 7}
    r = client.post("/recommend", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == ["Garlic", "lemon"] and body["mock"] is True
    recs = body["recommendations"]
    assert len(recs) == 7
    assert all(set(rec) == REC_KEYS for rec in recs)
    assert [rec["rank"] for rec in recs] == list(range(1, 8))
    scores = [rec["score"] for rec in recs]
    assert scores == sorted(scores, reverse=True)
    assert not {rec["name"] for rec in recs} & {"garlic", "lemon"}


def test_recommend_default_top_k_and_deterministic(client):
    payload = {"user_id": "u1", "ingredients": ["salmon"]}
    first = client.post("/recommend", json=payload).json()
    assert len(first["recommendations"]) == 10
    assert client.post("/recommend", json=payload).json() == first


def test_recommend_smaller_top_k_is_prefix_of_larger(client):
    base = {"user_id": "u1", "ingredients": ["tomato", "basil"]}
    by_k = {k: client.post("/recommend", json={**base, "top_k": k}).json()["recommendations"]
            for k in (5, 10, 50)}
    assert by_k[5] == by_k[10][:5]
    assert by_k[10] == by_k[50][:10]


def test_recommend_rejects_empty_ingredients_and_bad_top_k(client):
    base = {"user_id": "u1", "ingredients": ["salmon"]}
    assert client.post("/recommend", json={**base, "ingredients": []}).status_code == 422
    assert client.post("/recommend", json={**base, "top_k": 0}).status_code == 422
    assert client.post("/recommend", json={**base, "top_k": 51}).status_code == 422


def test_mock_recommend_falls_back_without_csv(tmp_path):
    from api.recommender import FALLBACK_VOCAB, mock_recommend

    recs = mock_recommend(["garlic"], top_k=5, vocab_path=tmp_path / "missing.csv")
    assert len(recs) == 5
    names = {rec["name"] for rec in recs}
    assert names <= set(FALLBACK_VOCAB) and "garlic" not in names


# ------------------------------------------------------------------ feedback loop
import numpy as np
import pandas as pd

from api import db
from api.recommender import ModelRecommender
from src.ranking import HybridRanker, RankingConfig


def _log(client, item, feedback, query=("tomato",), user="u1"):
    r = client.post("/log_interaction", json={"user_id": user, "query_ingredients": list(query),
                                              "recommended_ingredient": item, "feedback": feedback})
    assert r.status_code == 201


def test_feedback_state_latest_wins_and_not_for_me_is_query_scoped(tmp_path):
    path = str(tmp_path / "fb.db")
    db.init_db(path)
    db.insert_interaction(path, "u", ["tomato"], "basil", "dislike", None)
    db.insert_interaction(path, "u", ["tomato"], "basil", "like", None)       # changed their mind
    db.insert_interaction(path, "u", ["tomato"], "cilantro", "dislike", None)
    db.insert_interaction(path, "u", ["salmon"], "dill", "not_for_me", None)  # only with salmon
    db.insert_interaction(path, "u", ["tomato"], "garlic", "click", None)     # ignored
    assert db.feedback_for_user(path, "u", ["tomato"]) == {"liked": ["basil"], "disliked": ["cilantro"],
                                                           "allergic": []}
    assert db.feedback_for_user(path, "u", ["Salmon"])["disliked"] == ["cilantro", "dill"]
    assert db.feedback_for_user(path, "other", ["tomato"]) == {"liked": [], "disliked": [], "allergic": []}


def test_allergy_is_permanent(tmp_path):
    path = str(tmp_path / "fb.db")
    db.init_db(path)
    db.insert_interaction(path, "u", ["salad"], "peanut", "allergic", None)
    db.insert_interaction(path, "u", ["salad"], "peanut", "like", None)        # misclick: must not lift it
    fb = db.feedback_for_user(path, "u", ["salad"])
    assert fb["allergic"] == ["peanut"] and "peanut" not in fb["liked"]


def test_old_database_is_migrated_for_allergic(tmp_path):
    import sqlite3
    path = tmp_path / "old.db"
    old_schema = db.SCHEMA.replace(", 'allergic'", "")
    with sqlite3.connect(path) as conn:
        conn.executescript(old_schema)
        conn.execute("INSERT INTO interactions (user_id, query_ingredients, recommended_ingredient, feedback, "
                     "note, created_at) VALUES ('u', '[\"salad\"]', 'dill', 'like', NULL, '2026-11-01')")
    db.init_db(path)                                                           # migration
    db.insert_interaction(path, "u", ["salad"], "peanut", "allergic", None)
    rows = db.list_interactions(path, "u")
    assert [r["feedback"] for r in rows] == ["allergic", "like"] and rows[1]["created_at"] == "2026-11-01"
    db.init_db(path)                                                           # idempotent


def test_mock_dislike_removes_item_on_next_call(client):
    payload = {"user_id": "u1", "ingredients": ["salmon"]}
    first = client.post("/recommend", json=payload).json()["recommendations"]
    target = first[0]["name"]
    _log(client, target, "dislike", query=("salmon",))
    body = client.post("/recommend", json=payload).json()
    assert target not in [r["name"] for r in body["recommendations"]]
    assert body["applied_feedback"]["disliked"] == [target]


VOCAB = ["tomato", "basil", "oregano", "cilantro", "cumin", "garlic"]


@pytest.fixture
def model_client(tmp_path):
    # basil ~ oregano (close), cilantro ~ cumin (close), garlic far from everything
    emb = np.array([[1, 0, 0], [0.9, 0.4, 0], [0.88, 0.45, 0], [0.9, 0, 0.4],
                    [0.88, 0, 0.45], [0.3, 0.3, 0.9]], dtype=float)
    vocab = pd.DataFrame({"ingredient_id": range(len(VOCAB)), "name": VOCAB})
    cfg = RankingConfig(beta_chem=0.0, gamma_pref=0.2, normalize=True, feedback_sim_threshold=0.9)
    rec = ModelRecommender(processed_dir=tmp_path, ranker=HybridRanker(emb, vocab, config=cfg))
    with TestClient(create_app(str(tmp_path / "m.db"), recommender=rec)) as c:
        yield c


def _names(client, user="u1"):
    body = client.post("/recommend", json={"user_id": user, "ingredients": ["tomato"], "top_k": 5}).json()
    assert body["mock"] is False
    return [r["name"] for r in body["recommendations"]], body


def test_model_dislike_bans_only_that_item(model_client):
    before, body0 = _names(model_client)
    _log(model_client, "cilantro", "dislike")
    after, body = _names(model_client)
    assert "cilantro" in before and "cilantro" not in after
    score = lambda b, n: next(r["score"] for r in b["recommendations"] if r["name"] == n)
    assert score(body, "cumin") == pytest.approx(score(body0, "cumin"))   # neighbour untouched
    assert all(r["feedback"] is None for r in body["recommendations"])
    assert _names(model_client, user="someone_else")[0] == before  # feedback is per user


def test_model_like_boosts_item_and_neighbours(model_client):
    before, _ = _names(model_client)
    _log(model_client, "oregano", "like")
    after, body = _names(model_client)
    assert after.index("oregano") <= before.index("oregano")
    assert after.index("basil") <= before.index("basil")
    assert next(r for r in body["recommendations"] if r["name"] == "oregano")["feedback"] == "you liked it"
    score = lambda b, n: next(r["score"] for r in b["recommendations"] if r["name"] == n)
    assert score(body, "cumin") == pytest.approx(score(_names(model_client, user="fresh")[1], "cumin"))


def test_profile_is_updated_by_feedback(model_client):
    # new app user: no rating history, the profile is built from feedback only (cold start)
    p0 = model_client.get("/profile/newbie").json()
    assert p0["likes"] == [] and p0["dislikes"] == [] and p0["has_rating_history"] is False
    _log(model_client, "oregano", "like", user="newbie")
    _log(model_client, "cilantro", "dislike", user="newbie")
    p1 = model_client.get("/profile/newbie").json()
    likes = {d["name"]: d["value"] for d in p1["likes"]}
    assert likes["oregano"] == pytest.approx(1.5) and 0 < likes["basil"] < 1.5   # liked + close neighbour
    assert p1["dislikes"] == [{"name": "cilantro", "value": -1.0}]                # no spread to cumin
    _log(model_client, "cilantro", "like", user="newbie")                         # changed their mind
    p2 = model_client.get("/profile/newbie").json()
    assert "cilantro" in {d["name"] for d in p2["likes"]} and p2["dislikes"] == []



def test_model_allergic_bans_ingredient_and_its_variants(tmp_path):
    vocab = ["tomato", "peanut", "peanut butter", "peanut oil", "basil", "water chestnut"]
    emb = np.array([[1, 0], [0.9, 0.1], [0.85, 0.2], [0.8, 0.3], [0.95, 0.05], [0.7, 0.3]], dtype=float)
    ranker = HybridRanker(emb, pd.DataFrame({"ingredient_id": range(6), "name": vocab}),
                          config=RankingConfig(beta_chem=0.0, gamma_pref=0.2, normalize=True))
    rec = ModelRecommender(processed_dir=tmp_path, ranker=ranker)
    with TestClient(create_app(str(tmp_path / "a.db"), recommender=rec)) as c:
        _log(c, "peanut", "allergic")
        _log(c, "peanut", "like")                                    # cannot lift the allergy
        body = c.post("/recommend", json={"user_id": "u1", "ingredients": ["tomato"]}).json()
        names = {r["name"] for r in body["recommendations"]}
        assert names == {"basil", "water chestnut"}                  # every peanut variant is gone
        assert body["applied_feedback"]["allergic"] == ["peanut"]
        prof = c.get("/profile/u1").json()
        assert {"name": "peanut", "value": -1.0} in prof["dislikes"]
