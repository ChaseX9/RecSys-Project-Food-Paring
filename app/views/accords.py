"""Page "Trouver des accords": ingredients in, pairing suggestions out, feedback logged.

Filters (diet, allergies, disliked ingredients, number of results) come from the user's profile
and can be changed for a single search without touching the profile.
"""
from __future__ import annotations

import streamlit as st

from app import api_client, auth
from app.common import (ALLERGIES, API_DOWN_MSG, DIET_LABELS, DIETS, EXAMPLES, FEEDBACK, api_url,
                        explanation_fr, fr, ingredient_img, ingredient_label, load_vocab,
                        mode_badge)
from src.ranking.hybrid import RankingConfig

WEIGHTS = RankingConfig()  # same weights the ranker uses, to split the score bar
PARTS = {  # score component -> (colour, what it means for a user)
    "emb": ("#7FAF3A", "Habitudes culinaires"),
    "chem": ("#E0A040", "Chimie des arômes"),
    "pref": ("#5B9BD5", "Tes goûts"),
}

user = auth.current_user()
st.session_state.setdefault("ingredients", [])
st.session_state.setdefault("feedback", {})  # "query|ingredient" -> (feedback, row id)

# Search filters start from the profile; reloaded whenever the profile changes.
profile_sig = (user.username, user.diet, tuple(user.allergies), tuple(user.disliked), user.top_k) \
    if user else None
# (Streamlit also drops widget state when leaving the page, hence the second check.)
F_KEYS = ("f_allergies", "f_diet", "f_disliked", "f_top_k")
if st.session_state.get("filters_sig") != profile_sig or \
        any(k not in st.session_state for k in F_KEYS):
    st.session_state.filters_sig = profile_sig
    st.session_state.f_allergies = list(user.allergies) if user else []
    st.session_state.f_diet = DIET_LABELS.get(user.diet, "Aucun") if user else "Aucun"
    st.session_state.f_disliked = list(user.disliked) if user else []
    st.session_state.f_top_k = user.top_k if user else 10


# ----------------------------------------------------------------- callbacks
def set_query(ingredients: list[str]) -> None:
    st.session_state.ingredients = list(ingredients)


def add_to_query(name: str, query: list[str]) -> None:
    """Chain exploration: add a suggestion to the list (logged as a 'click')."""
    if name not in st.session_state.ingredients:
        st.session_state.ingredients = st.session_state.ingredients + [name]
    send_feedback(query, name, "click", toast=False)


def send_feedback(query: list[str], name: str, feedback: str, note_key: str | None = None,
                  toast: bool = True) -> None:
    # Read at click time: the note typed just before clicking would be stale if bound earlier.
    note = st.session_state.get(note_key) if note_key else None
    u = auth.current_user()
    try:
        row_id = api_client.log_interaction(auth.log_user_id(u), query, feedback, name, note,
                                            base_url=api_url())
    except api_client.APIError:
        st.toast("Impossible d'enregistrer ton avis : le serveur ne répond pas.", icon="❌")
        return
    if feedback != "click":
        st.session_state.feedback[f"{'+'.join(query)}|{name}"] = (feedback, row_id)
    if feedback == "not_for_me" and u and name not in u.disliked:
        auth.update_profile(u.username, disliked=u.disliked + [name])
        st.toast(f"{fr(name).capitalize()} ne te sera plus proposé (modifiable dans ton profil).",
                 icon="🚫")
        return
    if toast:
        icon, label = FEEDBACK[feedback]
        st.toast(f"Merci ! Avis « {label} » enregistré pour {fr(name)}.", icon=icon)


# ---------------------------------------------------------------------- main
st.title("🍽️ Qu'est-ce qui va bien avec… ?")
st.markdown("Choisis un ou plusieurs ingrédients que tu as sous la main : on te propose "
            "ceux qui s'accordent le mieux avec, en respectant tes allergies et ton régime.")

with st.expander("Comment ça marche ?"):
    st.markdown(
        "1. **Choisis tes ingrédients** dans la liste (tape en français ou en anglais).\n"
        "2. Vérifie **tes contraintes** juste en dessous : elles viennent de ton profil, "
        "et tu peux les changer pour une seule recherche.\n"
        "3. Clique sur **Trouver des accords**.\n"
        "4. Donne ton avis sur les suggestions (👍 👎 🚫 🍳) : il est enregistré et servira à "
        "améliorer les recommandations. « 🚫 Pas pour moi » retire l'ingrédient de tes "
        "prochaines suggestions."
    )

st.multiselect("Tes ingrédients", load_vocab(), key="ingredients", format_func=ingredient_label,
               placeholder="Ex. saumon, tomate, chocolat…")
ex_cols = st.columns([1.2] + [1] * len(EXAMPLES))
ex_cols[0].markdown("Idées :")
for col, (label, ings) in zip(ex_cols[1:], EXAMPLES.items()):
    col.button(label, on_click=set_query, args=(ings,), width="stretch")

# ------------------------------------------------------------- constraints
allergies, diet = st.session_state.f_allergies, DIETS[st.session_state.f_diet]
disliked, top_k = st.session_state.f_disliked, st.session_state.f_top_k
chips = ([f"🥗 {st.session_state.f_diet}"] if diet else []) + \
        [f"⚠️ sans {ALLERGIES[a].lower()}" for a in allergies] + \
        ([f"🚫 {len(disliked)} ingrédient(s) évité(s)"] if disliked else []) + \
        [f"🔢 {top_k} suggestions"]
st.markdown("**Tes contraintes :** " + " · ".join(chips))
with st.expander("Modifier pour cette recherche"):
    c1, c2 = st.columns(2)
    c1.multiselect("Allergies", list(ALLERGIES), key="f_allergies",
                   format_func=ALLERGIES.get, placeholder="Aucune")
    c2.selectbox("Régime alimentaire", list(DIETS), key="f_diet")
    c1.multiselect("Ingrédients à éviter", load_vocab(), key="f_disliked",
                   format_func=ingredient_label, placeholder="Aucun")
    c2.slider("Nombre de suggestions", 1, 30, key="f_top_k")
    if user:
        st.page_link("views/profil.py", label="Changer mes préférences par défaut (profil)",
                     icon="👤")
    else:
        st.caption("💡 Crée un compte pour enregistrer ces préférences une fois pour toutes.")

query = st.session_state.ingredients
if st.button("🔍 Trouver des accords", type="primary", disabled=not query):
    try:
        with st.spinner("On cherche les meilleurs accords…"):
            # Ask for extra results to make up for the disliked ingredients removed below.
            # TODO: pass `exclude` to /recommend once the API supports it (HybridRanker does).
            body, latency = api_client.recommend(query, auth.api_user_id(user),
                                                 min(top_k + len(disliked), 50), allergies, diet,
                                                 base_url=api_url())
        recs = [r for r in body["recommendations"] if r["name"] not in set(disliked)][:top_k]
        st.session_state.results = {"body": body, "recs": recs, "latency": latency,
                                    "query": list(query), "allergies": allergies, "diet": diet}
    except api_client.APIError:
        st.session_state.pop("results", None)
        st.error(API_DOWN_MSG)
if not query:
    st.caption("👆 Ajoute au moins un ingrédient, ou clique sur une idée.")


def score_breakdown(rec: dict) -> str:
    """Coloured bar showing how much each component contributes to the score."""
    parts = {"emb": WEIGHTS.alpha_emb * rec["emb_score"],
             "chem": WEIGHTS.beta_chem * rec["chem_score"],
             "pref": WEIGHTS.gamma_pref * rec["pref_score"]}
    parts = {k: max(v, 0.0) for k, v in parts.items()}
    total = sum(parts.values()) or 1.0
    segs = "".join(f'<div style="width:{100 * v / total:.1f}%;background:{PARTS[k][0]}"></div>'
                   for k, v in parts.items() if v > 0)
    legend = " &nbsp; ".join(
        f'<span style="color:{PARTS[k][0]}">●</span> {PARTS[k][1]} {100 * v / total:.0f} %'
        for k, v in parts.items())
    return (f'<div style="display:flex;height:8px;border-radius:4px;overflow:hidden;'
            f'background:rgba(128,128,128,.25);margin:2px 0 4px">{segs}</div>'
            f'<div style="font-size:.85em;opacity:.8">{legend} &nbsp;·&nbsp; '
            f'score {rec["score"]:.3f}</div>')


def render_card(rank: int, rec: dict, q: list[str], best: float, details: bool) -> None:
    name = rec["name"]
    key = f"{'+'.join(q)}|{name}"
    given = st.session_state.feedback.get(key)
    with st.container(border=True):
        title, match = st.columns([3, 2], vertical_alignment="center")
        english = f" <span style='opacity:.55;font-size:.7em'>({name})</span>" \
            if fr(name) != name else ""
        title.markdown(f"#### {ingredient_img(name, 48)}{rank}. {fr(name).capitalize()}{english}",
                       unsafe_allow_html=True)
        level = max(0.0, min(1.0, rec["score"] / best)) if best > 0 else 0.0
        match.progress(level, text="Niveau d'accord")
        st.markdown(f"💡 {explanation_fr(rec['explanation'])}")
        if details:
            st.markdown(score_breakdown(rec), unsafe_allow_html=True)

        if given:
            icon, label = FEEDBACK[given[0]]
            st.caption(f"{icon} Ton avis « {label} » est enregistré.")
            return
        b = st.columns(5)
        b[0].button("👍 J'aime", key=f"like|{key}", on_click=send_feedback,
                    args=(q, name, "like"), width="stretch")
        b[1].button("👎 Bof", key=f"dislike|{key}", on_click=send_feedback,
                    args=(q, name, "dislike"), width="stretch")
        b[2].button("🚫 Pas pour moi", key=f"nfm|{key}", on_click=send_feedback,
                    args=(q, name, "not_for_me"), width="stretch",
                    help="Je n'aime pas / je ne mange pas cet ingrédient : "
                         "il ne me sera plus proposé")
        with b[3].popover("🍳 Je l'ai cuisiné", width="stretch"):
            st.text_area("Qu'as-tu préparé ?", key=f"note|{key}",
                         placeholder="Ex. saumon au four, sauce aneth et citron")
            st.button("Enregistrer", key=f"cooked|{key}", on_click=send_feedback,
                      args=(q, name, "cooked"), kwargs={"note_key": f"note|{key}"},
                      type="primary")
        if name not in st.session_state.ingredients:
            b[4].button("➕ Ajouter à ma liste", key=f"add|{key}", on_click=add_to_query,
                        args=(name, q), width="stretch",
                        help="Ajoute cet ingrédient à ta recherche pour aller plus loin")


res = st.session_state.get("results")
if res:
    recs = res["recs"]
    st.divider()
    head, side = st.columns([4, 1], vertical_alignment="center")
    head.markdown(f"### {len(recs)} suggestions pour {' + '.join(fr(i) for i in res['query'])}")
    info = []
    if user and user.food_com_id:
        info.append("✨ personnalisé avec ton profil")
    head.caption(" · ".join(info + [f"réponse en {res['latency'] * 1000:.0f} ms"]))
    with side:
        mode_badge(res["body"].get("mock"))
        details = st.toggle("Détail des scores", value=False,
                            help="D'où vient chaque suggestion : habitudes culinaires, "
                                 "chimie des arômes, tes goûts.")
    if not recs:
        st.warning("Aucune suggestion ne respecte toutes tes contraintes. "
                   "Essaie de retirer une allergie ou de changer de régime.")
    best = recs[0]["score"] if recs else 1.0
    for i, rec in enumerate(recs, 1):
        render_card(i, rec, res["query"], best, details)
