"""Page "Noter des recettes": the user rates Food.com recipes 1-5 to teach the model their taste.

The ratings go to the API (PUT /recipe_ratings) and are folded into the user's rating profile,
the same one Food.com users get from their history (src/data/rating_profiles.py): the next
/recommend is personalized with them, no retraining needed.

Sections: the 10 starter recipes (shown on first login, can be skipped), one recipe to rate at
each visit, a search, and the list of the user's ratings (editable).
"""
from __future__ import annotations

import random

import streamlit as st

from app import api_client, auth
from app.i18n import t
from app.common import API_DOWN_MSG, api_url, fr, ingredient_img, recipe_img

GOAL = 10  # ratings after which the profile is considered "started"

user = auth.current_user()
uid = auth.log_user_id(user)
diet = user.diet if user else None
allergies = list(user.allergies) if user else []
# New seed at every visit of the page (page_changed is set by streamlit_app.py), so "a recipe
# to rate" is a different one each time the user comes back.
if st.session_state.get("page_changed") or "rate_seed" not in st.session_state:
    st.session_state.rate_seed = random.randrange(1_000_000)
    st.session_state.rate_skip = []
    st.session_state.pop("rate_next", None)


def _fail() -> None:
    st.toast(t("Impossible d'enregistrer ta note : le serveur ne répond pas."), icon="❌")


def on_stars(key: str, recipe_id: int, name: str) -> None:
    """st.feedback callback: 0-4 stars index -> rating 1-5; None (unselected) removes the rating."""
    value = st.session_state.get(key)
    try:
        if value is None:
            api_client.unrate_recipe(uid, recipe_id, base_url=api_url())
            st.toast(t("Note retirée : {name}", name=name))
        else:
            api_client.rate_recipe(uid, recipe_id, value + 1, base_url=api_url())
            st.toast(t("{stars} pour {name}. Merci, c'est pris en compte !",
                       stars="⭐" * (value + 1), name=name))
    except api_client.APIError:
        _fail()
        return
    st.session_state.pop("rate_next", None)  # the next recipe depends on what was rated


def remove(recipe_id: int, name: str) -> None:
    try:
        api_client.unrate_recipe(uid, recipe_id, base_url=api_url())
        st.toast(t("Note retirée : {name}", name=name))
    except api_client.APIError:
        _fail()


def skip_next(recipe_id: int) -> None:
    st.session_state.rate_skip = st.session_state.rate_skip + [recipe_id]
    st.session_state.pop("rate_next", None)


def finish_onboarding() -> None:
    u = auth.current_user()
    if u:
        try:
            auth.update_profile(u.username, onboarding_done=True)
        except auth.AuthError as e:
            st.toast(t(str(e)), icon="❌")
    st.session_state.onboarding_hidden = True


def recipe_card(r: dict, section: str, rating: int | None, compact: bool = False) -> None:
    """Name, ingredients (in French, with photos), Food.com stats and the 5 stars."""
    rid = r["recipe_id"]
    ings = r.get("ingredients") or []
    with st.container(border=True):
        pic, body = st.columns([1, 3] if compact else [1, 4], vertical_alignment="center")
        pic.markdown(recipe_img(rid, ings, 96 if compact else 120), unsafe_allow_html=True)
        with body:
            st.markdown(f"**{r['name']}**")
            shown = ings[:6] if compact else ings[:10]
            st.markdown(" ".join(f"<span style='white-space:nowrap;margin-right:.5rem'>"
                                 f"{ingredient_img(i, 22)}{fr(i)}</span>" for i in shown)
                        + (f" <span style='opacity:.6'>+{len(ings) - len(shown)}</span>"
                           if len(ings) > len(shown) else ""), unsafe_allow_html=True)
            if r.get("n_ratings"):
                mean = f"{r['mean_rating']:.1f} ★ · " if r.get("mean_rating") else ""
                st.caption(t("Sur Food.com : {mean}{n} avis", mean=mean, n=r["n_ratings"]))
            # the rating is in the key: the widget is rebuilt when the rating changes elsewhere
            key = f"stars|{section}|{rid}|{rating}"
            st.feedback("stars", key=key, default=None if rating is None else rating - 1,
                        on_change=on_stars, args=(key, rid, r["name"]))


# ---------------------------------------------------------------- data
try:
    mine = api_client.recipe_ratings(uid, base_url=api_url())
except api_client.APIError:
    st.error(t(API_DOWN_MSG))
    st.stop()
rated = {r["recipe_id"]: r["user_rating"] for r in mine}
n = len(rated)

# ---------------------------------------------------------------- header
st.title(t("⭐ Noter des recettes"))
st.markdown(t("Le modèle apprend **tes goûts à partir des recettes que tu notes**, comme pour les "
            "utilisateurs de Food.com : les ingrédients des recettes que tu adores remontent dans "
            "tes suggestions, ceux des recettes que tu n'aimes pas descendent. Plus tu notes, "
            "plus c'est précis."))
st.progress(min(n / GOAL, 1.0),
            text=t("**{n} recette(s) notée(s)**", n=n)
            + (t(" · encore {n} pour bien démarrer", n=GOAL - n) if n < GOAL
               else t(" · ton profil est lancé ✨")))
if user is None:
    st.caption(t("💡 En invité, tes notes ne valent que pour cette visite. Crée un compte pour "
                 "les garder."))
st.caption(t("Les recettes viennent de Food.com (noms en anglais). Clique sur les étoiles : "
             "1 = pas du tout, 5 = j'adore. Clique à nouveau sur la même étoile pour retirer la note."))

# --------------------------------------------------------------- onboarding
onboarding_open = n < GOAL and not (user and user.onboarding_done) and \
    not st.session_state.get("onboarding_hidden")
with st.expander(t("🚀 Pour commencer : 10 recettes populaires et variées"), expanded=onboarding_open):
    try:
        starters = api_client.recipes_onboarding(uid, diet, allergies, base_url=api_url())
    except api_client.APIError:
        starters = []
        st.error(t(API_DOWN_MSG))
    if starters:
        st.markdown(t("Note celles que tu connais (ou dont l'idée te plaît ou te déplaît) : elles "
                      "ont été choisies pour couvrir un maximum de goûts différents")
                    + (t(", en respectant ton régime et tes allergies.") if diet or allergies else "."))
        cols = st.columns(2)
        for k, r in enumerate(starters):
            with cols[k % 2]:
                recipe_card(r, "onb", rated.get(r["recipe_id"]), compact=True)
    else:
        st.success(t("Tu as déjà noté toutes les recettes de départ 🎉"))
    if onboarding_open:
        st.button(t("C'est bon pour l'instant") if n else t("Passer, je noterai plus tard"),
                  on_click=finish_onboarding, type="primary" if n else "secondary")

# ------------------------------------------------------- one recipe per visit
st.subheader(t("🎲 Une recette à noter"))
st.caption(t("À chaque visite, une nouvelle recette choisie pour découvrir tes goûts sur des "
             "ingrédients qu'on ne connaît pas encore."))
if "rate_next" not in st.session_state:
    try:
        st.session_state.rate_next = api_client.recipe_next(
            uid, diet, allergies, skip=st.session_state.rate_skip,
            seed=st.session_state.rate_seed + n, base_url=api_url())
    except api_client.APIError:
        st.session_state.rate_next = None
        st.error(t(API_DOWN_MSG))
nxt = st.session_state.rate_next
if nxt:
    left, right = st.columns([3, 1], vertical_alignment="center")
    with left:
        recipe_card(nxt, "next", rated.get(nxt["recipe_id"]))
        if nxt.get("new_ingredients"):
            st.caption(t("Pour en savoir plus sur tes goûts en : ")
                       + ", ".join(fr(i) for i in nxt["new_ingredients"]) + ".")
    right.button(t("🤷 Je ne connais pas, une autre"), on_click=skip_next,
                 args=(nxt["recipe_id"],), width="stretch")
else:
    st.info(t("Plus aucune recette à te proposer pour l'instant. Bravo !"))

# ------------------------------------------------------------------- search
st.subheader(t("🔍 Chercher une recette"))
query = st.text_input(t("Nom de la recette (en anglais)"), key="rate_query",
                      placeholder=t("ex. lasagna, chocolate cake, guacamole…"))
if len(query.strip()) >= 2:
    try:
        found = api_client.recipes_search(query.strip(), uid, diet, allergies, limit=10,
                                          base_url=api_url())
    except api_client.APIError:
        found = None
        st.error(t(API_DOWN_MSG))
    if found == []:
        st.info(t("Aucune recette trouvée (pense à chercher en anglais : « pancake », « soup »…)."))
    for r in found or []:
        recipe_card(r, "search", rated.get(r["recipe_id"]), compact=True)

# ---------------------------------------------------------------- my ratings
st.subheader(t("📋 Mes notes ({n})", n=n))
if not mine:
    st.caption(t("Tu n'as encore noté aucune recette."))
for r in mine:
    c0, c1, c2, c3 = st.columns([1, 5, 2, 1], vertical_alignment="center")
    c0.markdown(recipe_img(r["recipe_id"], r.get("ingredients") or [], 56), unsafe_allow_html=True)
    c1.markdown(f"**{r['name']}**  \n<span style='opacity:.7;font-size:.85em'>"
                + ", ".join(fr(i) for i in (r.get("ingredients") or [])[:6]) + "</span>",
                unsafe_allow_html=True)
    key = f"stars|mine|{r['recipe_id']}|{r['user_rating']}"
    with c2:
        st.feedback("stars", key=key, default=r["user_rating"] - 1, on_change=on_stars,
                    args=(key, r["recipe_id"], r["name"]))
    c3.button("🗑️", key=f"del|{r['recipe_id']}", on_click=remove, args=(r["recipe_id"], r["name"]),
              help=t("Retirer cette note"))
