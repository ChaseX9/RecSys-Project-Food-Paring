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
