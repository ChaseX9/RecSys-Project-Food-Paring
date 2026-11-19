"""app/auth.py on top of the accounts API: French messages, profile, password and admin rules."""
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from app import auth
from tests.conftest import route_api_client


@pytest.fixture(autouse=True)
def api(tmp_path, monkeypatch):
    auth.store.clear()
    with TestClient(create_app(str(tmp_path / "auth.db"), recommender=None)) as client:
        route_api_client(client, monkeypatch)
        yield client
    auth.store.clear()


def _user(api, username):
    return api.get(f"/users/{username}").json()


@pytest.mark.parametrize("args, msg", [
    (("Ab", "A", "a@b.fr", "12345678", "12345678"), "Identifiant"),
    (("newbie", "N", "pas-un-mail", "12345678", "12345678"), "e-mail"),
    (("newbie", "N", "a@b.fr", "court", "court"), "au moins"),
    (("newbie", "N", "a@b.fr", "12345678", "87654321"), "correspondent"),
    (("demo", "N", "a@b.fr", "12345678", "12345678"), "déjà pris"),
    (("newbie", "N", "ALEX@foodpairing.local", "12345678", "12345678"), "déjà utilisée"),
])
def test_register_validation(args, msg):
    with pytest.raises(auth.AuthError, match=msg):
        auth.register(*args)


def test_register_and_update_profile(api):
    auth.register("marie.d", "Marie D", "marie@exemple.fr", "motdepasse", "motdepasse")
    u = auth.update_profile("marie.d", diet="vegan", allergies=["soy"], top_k=5,
                            food_com_id="1533")
    assert (u.diet, u.allergies, u.top_k, u.role) == ("vegan", ["soy"], 5, "user")
    assert auth.history_user_id(u) == "1533" and auth.log_user_id(u) == "marie.d"
    assert _user(api, "marie.d")["allergies"] == ["soy"]          # stored by the API
    with pytest.raises(auth.AuthError, match="nombre"):
        auth.update_profile("marie.d", food_com_id="abc")
    with pytest.raises(ValueError):
        auth.update_profile("marie.d", role="admin")  # not self-editable
    assert auth.update_profile("marie.d", diet=None, food_com_id=None).diet is None


def test_change_password(api):
    with pytest.raises(auth.AuthError, match="actuel"):
        auth.change_password("demo", "faux", "nouveau123", "nouveau123")
    with pytest.raises(auth.AuthError, match="correspondent"):
        auth.change_password("demo", "demo1234", "nouveau123", "autre1234")
    auth.change_password("demo", "demo1234", "nouveau123", "nouveau123")
    assert api.post("/login", json={"username": "demo", "password": "nouveau123"}).status_code == 200
    assert api.post("/login", json={"username": "demo", "password": "demo1234"}).status_code == 401


def test_admin_actions_require_admin():
    with pytest.raises(auth.AuthError, match="administrateurs"):
        auth.admin_delete("demo", "camille")
    with pytest.raises(auth.AuthError, match="administrateurs"):
        auth.register("x.y", "X", "x@y.fr", "12345678", "12345678", role="admin", actor="demo")


def test_admin_cannot_remove_last_admin_or_self():
    with pytest.raises(auth.AuthError):
        auth.admin_set_active("admin", "admin", False)
    with pytest.raises(auth.AuthError):
        auth.admin_delete("admin", "admin")
    with pytest.raises(auth.AuthError, match="au moins un"):
        auth.admin_set_role("admin", "admin", "user")
    auth.admin_set_role("admin", "demo", "admin")
    auth.admin_set_role("admin", "admin", "user")  # fine now: demo is admin too


def test_admin_reset_deactivate_delete(api):
    temp = auth.admin_reset_password("admin", "camille")
    assert api.post("/login", json={"username": "camille", "password": temp}).status_code == 200
    auth.admin_set_active("admin", "camille", False)
    assert not _user(api, "camille")["active"]
    with pytest.raises(auth.AuthError, match="désactivé"):
        auth.login("camille", temp)
    auth.admin_delete("admin", "camille")
    assert api.get("/users/camille").status_code == 404
    assert any("supprimé" in e["message"] for e in auth.events())


def test_admin_set_api_url():
    with pytest.raises(auth.AuthError):
        auth.admin_set_api_url("admin", "pas une url")
    with pytest.raises(auth.AuthError, match="administrateurs"):
        auth.admin_set_api_url("demo", "http://127.0.0.1:9000")
    auth.admin_set_api_url("admin", "http://127.0.0.1:9000/")
    assert auth.store().settings["api_url"] == "http://127.0.0.1:9000"


def test_api_down_gives_a_clear_message(monkeypatch):
    from app import api_client

    def down(*a, **k):
        raise api_client.APIError("down")

    monkeypatch.setattr(api_client, "_request", down)
    with pytest.raises(auth.AuthError, match="ne répond pas"):
        auth.register("lea.m", "Léa", "lea@exemple.fr", "motdepasse", "motdepasse")
