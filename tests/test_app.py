"""Streamlit UI driven end-to-end against the real FastAPI app (in-process, no server needed)."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from app import api_client, auth
from tests.conftest import route_api_client

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

ENTRY = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


@pytest.fixture(autouse=True)
def fresh_app_state():
    auth.store.clear()  # app event log / settings re-created for every test
    yield
    auth.store.clear()


@pytest.fixture
def api(tmp_path, monkeypatch):
    """Route every api_client call through a FastAPI TestClient backed by a temp SQLite DB
    (accounts included: the demo accounts are created on startup)."""
    with TestClient(create_app(str(tmp_path / "test.db"))) as client:
        route_api_client(client, monkeypatch)
        yield client


def _user(api, username: str) -> dict:
    return api.get(f"/users/{username}").json()


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


def test_login_required_then_wrong_password(api):
    at = AppTest.from_file(ENTRY, default_timeout=30).run()
    assert at.title[0].value.startswith("👋")
    at = _login("demo", "mauvais")
    assert any("incorrect" in e.value for e in at.error)
    assert auth.events()[0]["message"].startswith("Échec de connexion")


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


def _names(at) -> list[str]:
    return [r["name"] for r in at.session_state.results["recs"]]


def test_profile_preferences_prefill_search_and_dislike_is_applied_by_api(api):
    at = _login("camille")  # vegetarian, nut allergy, dislikes cilantro
    assert at.session_state.f_diet == "Végétarien"
    assert at.session_state.f_allergies == ["nut"]
    _button(at, "Tomate + basilic").click().run()
    _button(at, "🔍").click().run()
    res = at.session_state.results
    assert res["diet"] == "vegetarian" and "cilantro" not in _names(at)  # sent as `exclude`
    assert len(res["recs"]) == 10                                         # no client-side cut

    first = res["recs"][0]["name"]
    _button(at, "👎").click().run()
    assert _user(api, "camille")["disliked"] == ["cilantro"]  # profile list untouched
    _button(at, "🔍").click().run()
    assert first not in _names(at)
    assert at.session_state.results["body"]["applied_feedback"]["disliked"] == [first]


def test_not_for_me_only_applies_to_the_same_search(api):
    at = _login("demo")
    _button(at, "Tomate + basilic").click().run()
    _button(at, "🔍").click().run()
    first = _names(at)[0]
    _button(at, "🚫").click().run()
    _button(at, "🔍").click().run()
    assert first not in _names(at)
    fb = api.get("/feedback/demo").json()
    assert first not in fb["disliked"]  # outside this search it is still allowed


def test_feedback_of_food_com_linked_account_is_applied(api):
    """demo is linked to a Food.com profile: ratings come from it, feedback from the account."""
    at = _login("demo")
    assert _user(api, "demo")["food_com_id"]
    _button(at, "Saumon + citron").click().run()
    _button(at, "🔍").click().run()
    first = _names(at)[0]
    _button(at, "👎").click().run()
    assert api.get("/interactions").json()[0]["user_id"] == "demo"
    _button(at, "🔍").click().run()
    assert first not in _names(at)


def test_allergy_button_bans_the_ingredient(api):
    at = _login("demo")
    _button(at, "Poulet").click().run()
    _button(at, "🔍").click().run()
    first = _names(at)[0]
    _button(at, "Oui, j'y suis allergique").click().run()
    assert api.get("/interactions").json()[0]["feedback"] == "allergic"
    _button(at, "🔍").click().run()
    assert first not in _names(at)
    assert at.session_state.results["body"]["applied_feedback"]["allergic"] == [first]


def test_profile_shows_feedback_and_lets_the_user_change_their_mind(api):
    at = _login("demo")
    _button(at, "Poulet").click().run()
    _button(at, "🔍").click().run()
    first = _names(at)[0]
    _button(at, "👎").click().run()
    at = _goto(at, "profil.py")
    _button(at, "👍 En fait j'aime").click().run()
    assert not at.exception
    assert api.get("/feedback/demo").json()["liked"] == [first]
    assert api.get("/interactions").json()[0]["query_ingredients"] == ["(profil)"]


def test_cooked_note_is_read_at_click_time(api):
    at = _login("demo")
    _button(at, "Tomate + basilic").click().run()
    _button(at, "🔍").click().run()
    at.text_area[0].input("bruschetta").run()
    _button(at, "Enregistrer").click().run()
    row = api.get("/interactions").json()[0]
    assert row["feedback"] == "cooked" and row["note"] == "bruschetta"


def test_guest_has_no_profile_or_admin_and_each_guest_has_own_id(api):
    at = _login(None)
    links = {p.label for p in at.get("page_link")} if at.get("page_link") else set()
    assert not links & {"Mon profil", "Administration", "Mon historique"}
    _button(at, "Poulet").click().run()
    _button(at, "🔍").click().run()
    first = _names(at)[0]
    _button(at, "👎").click().run()
    guest = api.get("/interactions").json()[0]["user_id"]
    assert guest.startswith(auth.GUEST_PREFIX)

    other = _login(None)  # another browser session: the first guest's dislike is not applied
    _button(other, "Poulet").click().run()
    _button(other, "🔍").click().run()
    assert other.session_state.guest_id != guest and first in _names(other)


def test_logout_returns_to_login(api):
    at = _login("demo")
    _button(at, "Se déconnecter").click().run()
    assert not at.exception
    assert "auth_user" not in at.session_state
    assert at.title[0].value.startswith("👋")


def test_api_down_shows_error(api, monkeypatch):
    at = _login("demo")

    def down(*a, **k):
        raise api_client.APIError("down")

    monkeypatch.setattr(api_client, "_request", down)  # the API goes down after login
    _button(at, "Poulet").click().run()
    _button(at, "🔍").click().run()
    assert any("ne répond pas" in e.value for e in at.error)
    assert at.session_state.auth_user == "demo"  # still logged in (last known account)


def test_theme_toggle_switches_config(api):
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
    fridge = set(_user(api, "demo")["fridge"])
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
    fridge = set(_user(api, "camille")["fridge"])
    assert res["pairs"] and all({p["a"], p["b"]} <= fridge for p in res["pairs"])
    assert len({frozenset((p["a"], p["b"])) for p in res["pairs"]}) == len(res["pairs"])
    assert res["star"] in fridge

    _button(at, "+ ").click().run()  # quick add is saved to the profile
    assert len(_user(api, "camille")["fridge"]) == len(fridge) + 1


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
    assert explanation_fr("share aromatic compounds (x) + close to basil, which you liked") == \
        "Partagent des molécules aromatiques et proche de basilic, que tu as aimé."
    assert explanation_fr("suggested by the pairing model + you liked it") == \
        "Suggéré par le modèle et tu l'as aimé."


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


def test_recipe_images_photo_collage_or_emoji():
    import re

    from app.common import _image_uri, _image_paths, recipe_img

    photo = recipe_img(54, ["carrot", "flour"])            # "Carrot Cake": TheMealDB photo
    assert photo.count("<img") == 1 and "object-fit:cover" in photo
    collage = recipe_img(-1, ["salt", "salmon", "dill", "lemon", "butter", "garlic"])
    srcs = re.findall(r"src='([^']+)'", collage)
    assert len(srcs) == 4                                  # 2 x 2 at most
    assert srcs[0] == _image_uri(_image_paths()["salmon"])  # main ingredients before staples
    staples = [_image_uri(_image_paths()[s]) for s in ("salt", "butter") if s in _image_paths()]
    assert not set(staples) & set(srcs)                   # 4 main ingredients have a photo
    assert "🐟" in recipe_img(-1, ["zzz fish"])            # no photo at all -> emoji
    assert recipe_img(-1, []).count("<img") == 0


def test_accounts_survive_an_app_restart(tmp_path, monkeypatch):
    """Accounts are in the API database: a new sign-up is still there after a restart."""
    db_path = str(tmp_path / "persist.db")
    with TestClient(create_app(db_path, recommender=None)) as client:
        route_api_client(client, monkeypatch)
        at = AppTest.from_file(ENTRY, default_timeout=30).run()
        auth.register("lea.m", "Léa M", "lea@exemple.fr", "motdepasse", "motdepasse")
    auth.store.clear()
    with TestClient(create_app(db_path, recommender=None)) as client:
        route_api_client(client, monkeypatch)
        at = _login("lea.m", "motdepasse")
        assert at.session_state.auth_user == "lea.m"


def test_sign_up_page_creates_account_in_api(api):
    at = AppTest.from_file(ENTRY, default_timeout=30).run()
    signup = [t for t in at.text_input if t.label == "Identifiant"][1]  # 2nd tab
    signup.input("lea.m")
    for label, value in [("Prénom et nom", "Léa M"), ("Adresse e-mail", "lea@exemple.fr"),
                         ("Confirmer le mot de passe", "motdepasse")]:
        next(t for t in at.text_input if t.label == label).input(value)
    [t for t in at.text_input if t.label == "Mot de passe"][1].input("motdepasse")
    _button(at, "Créer mon compte").click().run()
    assert not at.exception and at.session_state.auth_user == "lea.m"
    assert _user(api, "lea.m")["display_name"] == "Léa M"
    assert at.title[0].value.startswith("⭐")  # first login: sent to the recipe ratings


# ------------------------------------------------------------- recipe ratings
def _new_account(api, username="lea.m") -> AppTest:
    api.post("/users", json={"username": username, "display_name": "Léa M",
                             "email": f"{username}@exemple.fr", "password": "motdepasse"})
    return _login(username, "motdepasse")


def _stars(at, section: str):
    return [f for f in at.feedback if f.key.startswith(f"stars|{section}|")]


def test_first_login_onboarding_rate_then_finish(api):
    at = _new_account(api)
    assert at.title[0].value.startswith("⭐")
    starters = _stars(at, "onb")
    assert len(starters) == 10
    starters[0].set_value(4).run()                      # 5 stars
    starters = _stars(at, "onb")
    starters[1].set_value(0).run()                      # 1 star
    rows = api.get("/recipe_ratings/lea.m").json()
    assert sorted(r["user_rating"] for r in rows) == [1, 5]
    assert len(_stars(at, "mine")) == 2
    _button(at, "C'est bon pour l'instant").click().run()
    assert _user(api, "lea.m")["onboarding_done"] is True
    assert api.get("/profile/lea.m").json()["n_recipe_ratings"] == 2

    at = _login("lea.m", "motdepasse")                  # next login: no redirect any more
    assert at.title[0].value.startswith("🍽️")


def test_skip_onboarding(api):
    at = _new_account(api)
    _button(at, "Passer").click().run()
    assert _user(api, "lea.m")["onboarding_done"] is True and \
        api.get("/recipe_ratings/lea.m").json() == []


def test_recipe_to_rate_changes_and_can_be_skipped(api):
    at = _new_account(api)
    first = at.session_state.rate_next["recipe_id"]
    _button(at, "🤷").click().run()
    second = at.session_state.rate_next["recipe_id"]
    assert second != first
    _stars(at, "next")[0].set_value(3).run()
    rows = api.get("/recipe_ratings/lea.m").json()
    assert {r["recipe_id"]: r["user_rating"] for r in rows} == {second: 4}
    assert at.session_state.rate_next["recipe_id"] != second   # a new one once rated


def test_search_rate_and_remove(api):
    at = _new_account(api)
    at.text_input(key="rate_query").input("chocolate chip cookies").run()
    found = _stars(at, "search")
    assert found
    found[0].set_value(4).run()
    rows = api.get("/recipe_ratings/lea.m").json()
    assert len(rows) == 1 and "cookie" in rows[0]["name"].lower()
    _button(at, "🗑️").click().run()
    assert api.get("/recipe_ratings/lea.m").json() == []


def test_rating_page_respects_diet(api):
    at = _login("camille")                              # vegetarian
    at = _goto(at, "recettes.py")
    names = " ".join(at.session_state.rate_next["name"].lower().split())
    for r in api.get("/recipes/onboarding", params={"user_id": "camille",
                                                    "diet": "vegetarian"}).json():
        names += " " + r["name"].lower()
    assert not any(w in names for w in ("chicken", "pork", "beef", "ribs", "salmon", "shrimp"))
