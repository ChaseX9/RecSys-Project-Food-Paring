import pytest
from fastapi.testclient import TestClient

from api.main import create_app


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(str(tmp_path / "test.db"))) as c:
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


REC_KEYS = {"rank", "name", "score", "emb_score", "chem_score", "pref_score", "explanation"}


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


def test_recommend_candidates_restricts_output(client):
    fridge = ["lemon", "dill", "butter", "spinach", "garlic", "salmon"]
    payload = {"user_id": "u1", "ingredients": ["salmon"], "top_k": 10, "candidates": fridge}
    recs = client.post("/recommend", json=payload).json()["recommendations"]
    names = [r["name"] for r in recs]
    assert names and set(names) <= set(fridge) - {"salmon"}  # never the query itself
    assert len(names) == len(fridge) - 1


def test_recommend_empty_candidates_returns_nothing(client):
    payload = {"user_id": "u1", "ingredients": ["salmon"], "candidates": []}
    assert client.post("/recommend", json=payload).json()["recommendations"] == []
