"""Mock account store: registration, profile, password and admin rules."""
import pytest

from app import auth


@pytest.fixture(autouse=True)
def fresh_accounts():
    auth.store.clear()
    yield
    auth.store.clear()


def test_passwords_are_hashed_and_salted():
    u = auth.store().users["demo"]
    assert "demo1234" not in (u.pw_hash, u.salt) and u.check_password("demo1234")
    assert not u.check_password("demo12345")


@pytest.mark.parametrize("args, msg", [
    (("Ab", "A", "a@b.fr", "12345678", "12345678"), "Identifiant"),
    (("newbie", "N", "pas-un-mail", "12345678", "12345678"), "e-mail"),
    (("newbie", "N", "a@b.fr", "court", "court"), "au moins"),
    (("newbie", "N", "a@b.fr", "12345678", "87654321"), "correspondent"),
    (("demo", "N", "a@b.fr", "12345678", "12345678"), "déjà pris"),
    (("newbie", "N", "alex@foodpairing.local", "12345678", "12345678"), "déjà utilisée"),
])
def test_register_validation(args, msg):
    with pytest.raises(auth.AuthError, match=msg):
        auth.register(*args)


def test_register_and_update_profile():
    auth.register("marie.d", "Marie D", "marie@exemple.fr", "motdepasse", "motdepasse")
    u = auth.update_profile("marie.d", diet="vegan", allergies=["soy"], top_k=5,
                            food_com_id="1533")
    assert (u.diet, u.allergies, u.top_k, u.role) == ("vegan", ["soy"], 5, "user")
    assert auth.history_user_id(u) == "1533" and auth.log_user_id(u) == "marie.d"
    with pytest.raises(auth.AuthError):
        auth.update_profile("marie.d", food_com_id="abc")
    with pytest.raises(ValueError):
        auth.update_profile("marie.d", role="admin")  # not self-editable


def test_change_password():
    with pytest.raises(auth.AuthError, match="actuel"):
        auth.change_password("demo", "faux", "nouveau123", "nouveau123")
    auth.change_password("demo", "demo1234", "nouveau123", "nouveau123")
    assert auth.store().users["demo"].check_password("nouveau123")


def test_admin_actions_require_admin():
    with pytest.raises(auth.AuthError, match="administrateurs"):
        auth.admin_delete("demo", "camille")


def test_admin_cannot_remove_last_admin_or_self():
    with pytest.raises(auth.AuthError):
        auth.admin_set_active("admin", "admin", False)
    with pytest.raises(auth.AuthError):
        auth.admin_delete("admin", "admin")
    with pytest.raises(auth.AuthError, match="au moins un"):
        auth.admin_set_role("admin", "admin", "user")
    auth.admin_set_role("admin", "demo", "admin")
    auth.admin_set_role("admin", "admin", "user")  # fine now: demo is admin too


def test_admin_reset_deactivate_delete():
    temp = auth.admin_reset_password("admin", "camille")
    assert auth.store().users["camille"].check_password(temp)
    auth.admin_set_active("admin", "camille", False)
    assert not auth.store().users["camille"].active
    auth.admin_delete("admin", "camille")
    assert "camille" not in auth.store().users
    assert any("supprimé" in e["message"] for e in auth.events())


def test_admin_set_api_url():
    with pytest.raises(auth.AuthError):
        auth.admin_set_api_url("admin", "pas une url")
    auth.admin_set_api_url("admin", "http://127.0.0.1:9000/")
    assert auth.store().settings["api_url"] == "http://127.0.0.1:9000"
