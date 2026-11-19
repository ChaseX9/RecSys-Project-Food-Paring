"""Streamlit UI driven end-to-end against the real FastAPI app (in-process, no server needed)."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from app import api_client, auth

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

ENTRY = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


@pytest.fixture(autouse=True)
def fresh_accounts():
    auth.store.clear()  # demo accounts re-created for every test
    yield
    auth.store.clear()


@pytest.fixture
def api(tmp_path, monkeypatch):
    """Route every api_client call through a FastAPI TestClient backed by a temp SQLite DB."""
    with TestClient(create_app(str(tmp_path / "test.db"))) as client:
        def fake_request(method, base_url, path, **kwargs):
            r = client.request(method, path, **kwargs)
            if r.status_code >= 400:
                raise api_client.APIError(f"HTTP {r.status_code}")
            return r

        monkeypatch.setattr(api_client, "_request", fake_request)
        yield client


def _login(username: str | None = "demo", password: str | None = None) -> AppTest:
    at = AppTest.from_file(ENTRY, default_timeout=30).run()
    assert not at.exception, at.exception
    if username is None:
        _button(at, "Continuer sans compte").click().run()
    else:
        pw = password or next(p for u, p, _ in auth.DEMO_ACCOUNTS if u == username)
        # the login tab comes first, so its two fields are the first matches
        inputs = [t for t in at.text_input if t.label in ("Identifiant", "Mot de passe")]
        inputs[0].input(username)
        inputs[1].input(pw)
        _button(at, "Se connecter").click().run()
    assert not at.exception, at.exception
    return at


def _goto(at: AppTest, page: str) -> AppTest:
    at.switch_page(f"views/{page}").run()
    assert not at.exception, at.exception
    return at


def _button(at, prefix):
    return next(b for b in at.button if b.label.startswith(prefix))


def test_login_required_then_wrong_password():
    at = AppTest.from_file(ENTRY, default_timeout=30).run()
    assert at.title[0].value.startswith("👋")
    at = _login("demo", "mauvais")
    assert any("incorrect" in e.value for e in at.error)
    assert auth.events()[0]["message"] == "Échec de connexion"


def test_pairings_flow_logs_feedback_under_account(api):
    at = _login("demo")
    assert at.title[0].value.startswith("🍽️")
    _button(at, "Saumon + citron").click().run()
    at.multiselect(key="f_allergies").set_value(["dairy"]).run()
    _button(at, "🔍").click().run()
    assert not at.exception

    res = at.session_state.results
    assert len(res["recs"]) == 10 and res["allergies"] == ["dairy"]
    _button(at, "👍").click().run()
    rows = api.get("/interactions").json()
    assert rows[0]["user_id"] == "demo" and rows[0]["feedback"] == "like"


def test_profile_preferences_prefill_search_and_not_for_me_excludes(api):
    at = _login("camille")  # vegetarian, nut allergy, dislikes cilantro
    assert at.session_state.f_diet == "Végétarien"
    assert at.session_state.f_allergies == ["nut"]
    _button(at, "Tomate + basilic").click().run()
    _button(at, "🔍").click().run()
    res = at.session_state.results
    assert res["diet"] == "vegetarian" and "cilantro" not in {r["name"] for r in res["recs"]}

    first = res["recs"][0]["name"]
    _button(at, "🚫").click().run()
    assert first in auth.store().users["camille"].disliked
    _button(at, "🔍").click().run()
    assert first not in {r["name"] for r in at.session_state.results["recs"]}


def test_cooked_note_is_read_at_click_time(api):
    at = _login("demo")
    _button(at, "Tomate + basilic").click().run()
    _button(at, "🔍").click().run()
    at.text_area[0].input("bruschetta").run()
    _button(at, "Enregistrer").click().run()
    row = api.get("/interactions").json()[0]
    assert row["feedback"] == "cooked" and row["note"] == "bruschetta"


def test_guest_has_no_profile_or_admin_and_logs_as_anonymous(api):
    at = _login(None)
    links = {p.label for p in at.get("page_link")} if at.get("page_link") else set()
    assert not links & {"Mon profil", "Administration", "Mon historique"}
    _button(at, "Poulet").click().run()
    _button(at, "🔍").click().run()
    _button(at, "👍").click().run()
    assert api.get("/interactions").json()[0]["user_id"] == auth.GUEST


def test_logout_returns_to_login(api):
    at = _login("demo")
    _button(at, "Se déconnecter").click().run()
    assert not at.exception
    assert "auth_user" not in at.session_state
    assert at.title[0].value.startswith("👋")


def test_api_down_shows_error(monkeypatch):
    def down(*a, **k):
        raise api_client.APIError("down")

    monkeypatch.setattr(api_client, "_request", down)
    at = _login("demo")
    _button(at, "Poulet").click().run()
    _button(at, "🔍").click().run()
    assert any("ne répond pas" in e.value for e in at.error)


def test_theme_toggle_switches_config():
    from streamlit import config

    at = AppTest.from_file(ENTRY, default_timeout=30).run()
    at.toggle(key="dark_mode").set_value(False).run()
    assert not at.exception and config.get_option("theme.base") == "light"
    at.toggle(key="dark_mode").set_value(True).run()
    assert config.get_option("theme.base") == "dark"


@pytest.mark.parametrize("page", ["frigo.py", "historique.py", "profil.py", "a_propos.py"])
def test_user_pages_render(api, page):
    _goto(_login("demo"), page)


def test_admin_page_renders_for_admin_only(api):
    at = _goto(_login("admin"), "admin.py")
    assert at.title[0].value.startswith("🛠️")
    _button(at, "🔌 Tester la connexion").click().run()
    assert not at.exception

    at = _login("demo")
    with pytest.raises(Exception):
        _goto(at, "admin.py")  # not in a regular user's navigation


def test_fridge_around_one_ingredient_only_suggests_fridge_items(api):
    at = _goto(_login("demo"), "frigo.py")
    fridge = set(auth.store().users["demo"].fridge)
    assert set(at.session_state.fridge_sel) == fridge
    at.selectbox[0].set_value("salmon").run()  # base ingredient, from the fridge
    _button(at, "🔍 Trouver les associations").click().run()
    assert not at.exception
    res = at.session_state.fridge_results
    names = {r["name"] for r in res["recs"]}
    assert names and names <= fridge - {"salmon"}
    assert not {r["name"] for r in res["to_buy"]} & fridge  # shopping ideas are new items

    _button(at, "👍").click().run()
    row = api.get("/interactions").json()[0]
    assert row["query_ingredients"] == ["salmon"] and row["recommended_ingredient"] in fridge


def test_fridge_best_pairs_and_persistence(api):
    at = _goto(_login("camille"), "frigo.py")  # vegetarian, nut allergy
    at.radio[0].set_value("Les meilleures associations de mon frigo").run()
    _button(at, "🔍 Trouver les meilleurs duos").click().run()
    assert not at.exception
    res = at.session_state.fridge_results
    fridge = set(auth.store().users["camille"].fridge)
    assert res["pairs"] and all({p["a"], p["b"]} <= fridge for p in res["pairs"])
    assert len({frozenset((p["a"], p["b"])) for p in res["pairs"]}) == len(res["pairs"])
    assert res["star"] in fridge

    _button(at, "+ ").click().run()  # quick add is saved to the profile
    assert len(auth.store().users["camille"].fridge) == len(fridge) + 1


def test_fridge_needs_two_items_for_guest(api):
    at = _goto(_login(None), "frigo.py")
    assert at.session_state.fridge_sel == []
    assert any("au moins" in i.value for i in at.info)


def test_explanations_are_translated():
    from app.common import explanation_fr

    out = explanation_fr("often paired together in recipes (NPMI=0.30) + share aromatic "
                         "compounds (3,4-Dihydroxybenzoic Acid (+2 more))")
    assert out == ("Souvent cuisinés ensemble dans les recettes et "
                   "partagent des molécules aromatiques.")
    assert "exemple" in explanation_fr("[mock] dill is often paired with salmon.")


def test_client_reports_unreachable_api():
    with pytest.raises(api_client.APIError):
        api_client.recommend(["salmon"], "u1", base_url="http://127.0.0.1:9")
    assert api_client.health("http://127.0.0.1:9") is False
    assert api_client.CALLS[0]["ok"] is False


def test_demo_accounts_shown_with_one_click_login(api):
    at = AppTest.from_file(ENTRY, default_timeout=30).run()
    shown = " ".join(m.value for m in at.markdown)
    for username, password, _ in auth.DEMO_ACCOUNTS:
        assert f"`{username}`" in shown and f"`{password}`" in shown
    next(b for b in at.button if b.key == "demo_camille").click().run()
    assert not at.exception and at.session_state.auth_user == "camille"


def test_ingredient_images_photo_or_emoji():
    from app.common import ingredient_emoji, ingredient_img

    assert "data:image/webp;base64," in ingredient_img("salmon")  # TheMealDB photo
    assert ingredient_emoji("smoked haddock fish") == "🐟"
    assert ingredient_emoji("unknownium") == "🍽️"
    assert "🐟" in ingredient_img("zzz fish")  # no photo -> family emoji
