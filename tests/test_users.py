"""Accounts API (api/users.py): storage, password hashing, rules, demo seed."""
import sqlite3

import pytest
from fastapi.testclient import TestClient

from api import db, users
from api.main import create_app


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "users.db")


@pytest.fixture
def client(db_path):
    with TestClient(create_app(db_path, recommender=None)) as c:
        yield c


def _new(client, username="lea.m", email="lea@exemple.fr", **extra):
    return client.post("/users", json={"username": username, "display_name": "Léa M",
                                       "email": email, "password": "motdepasse", **extra})


def test_demo_accounts_seeded_once(db_path, client):
    names = {u["username"] for u in client.get("/users").json()}
    assert names == {u for u, _, _ in users.DEMO_ACCOUNTS}
    client.delete("/admin/users/camille", params={"actor": "admin"})
    users.init_db(db_path)                           # restart: table not empty, no re-seed
    assert "camille" not in {u["username"] for u in client.get("/users").json()}


def test_no_seed_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("FOODPAIRING_SEED_DEMO", "0")
    with TestClient(create_app(str(tmp_path / "empty.db"), recommender=None)) as c:
        assert c.get("/users").json() == []


def test_password_is_salted_and_hashed_never_returned(db_path, client):
    body = _new(client).json()
    assert "pw_hash" not in body and "salt" not in body and "password" not in body
    with sqlite3.connect(db_path) as conn:
        salt, pw_hash = conn.execute("SELECT salt, pw_hash FROM users WHERE username='lea.m'").fetchone()
        other_salt = conn.execute("SELECT salt FROM users WHERE username='demo'").fetchone()[0]
    assert "motdepasse" not in (salt, pw_hash) and len(pw_hash) == 64 and salt != other_salt


def test_sign_up_login_and_profile(client):
    assert _new(client).status_code == 201
    assert _new(client).json()["detail"] == "Cet identifiant est déjà pris."
    assert _new(client, "autre", "LEA@exemple.fr").status_code == 409   # e-mail, any case
    assert client.post("/login", json={"username": "lea.m", "password": "faux"}).status_code == 401
    u = client.post("/login", json={"username": " Lea.M ", "password": "motdepasse"}).json()
    assert u["last_login"] and u["role"] == "user" and u["fridge"] == []

    r = client.patch("/users/lea.m", json={"diet": "vegan", "fridge": ["egg", "egg", "milk"]})
    assert r.json()["diet"] == "vegan" and r.json()["fridge"] == ["egg", "milk"]
    assert client.patch("/users/lea.m", json={"diet": None}).json()["diet"] is None
    assert client.patch("/users/lea.m", json={"role": "admin"}).json()["role"] == "user"  # ignored
    assert client.patch("/users/lea.m", json={"top_k": 0}).status_code == 422
    assert client.get("/users/nobody").status_code == 404


def test_only_admins_create_admins(client):
    assert _new(client, role="admin").status_code == 403
    assert _new(client, role="admin", actor="demo").status_code == 403
    assert _new(client, role="admin", actor="admin").json()["role"] == "admin"


def test_last_active_admin_is_protected(client):
    act = lambda what, target, **f: client.post(f"/admin/users/{target}/{what}",
                                                json={"actor": "admin", **f})
    assert act("role", "admin", role="user").status_code == 409
    assert act("active", "admin", active=False).status_code == 409
    assert act("role", "demo", role="admin").json()["role"] == "admin"
    assert act("role", "admin", role="user").status_code == 200       # demo is admin now
    assert client.post("/admin/users/camille/active",
                       json={"actor": "admin", "active": False}).status_code == 403  # no longer admin


def test_deactivated_account_cannot_log_in(client):
    client.post("/admin/users/camille/active", json={"actor": "admin", "active": False})
    r = client.post("/login", json={"username": "camille", "password": "camille1234"})
    assert r.status_code == 403 and "désactivé" in r.json()["detail"]


def test_reset_password(client):
    temp = client.post("/admin/users/demo/reset_password",
                       json={"actor": "admin"}).json()["temporary_password"]
    assert client.post("/login", json={"username": "demo", "password": temp}).status_code == 200
    assert client.post("/login", json={"username": "demo", "password": "demo1234"}).status_code == 401


def test_delete_removes_the_accounts_feedback(db_path, client):
    client.post("/log_interaction", json={"user_id": "camille", "query_ingredients": ["salad"],
                                          "recommended_ingredient": "peanut", "feedback": "allergic"})
    assert client.delete("/admin/users/camille", params={"actor": "demo"}).status_code == 403
    assert client.delete("/admin/users/camille", params={"actor": "admin"}).status_code == 204
    assert db.list_interactions(db_path, "camille") == []
    # a new account with the same username starts from scratch
    _new(client, "camille", "new@exemple.fr")
    assert client.get("/feedback/camille").json()["allergic"] == []


def test_change_password(client):
    r = client.post("/users/demo/password", json={"current": "faux", "new": "nouveau123"})
    assert r.status_code == 401
    assert client.post("/users/demo/password", json={"current": "demo1234", "new": "court"}).status_code == 400
    assert client.post("/users/demo/password",
                       json={"current": "demo1234", "new": "nouveau123"}).status_code == 204
