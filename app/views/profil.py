"""Page "Mon profil": personal info, food preferences, what the model learned, password."""
from __future__ import annotations

from datetime import datetime

import streamlit as st

from app import api_client, auth
from app.common import (ALLERGIES, API_DOWN_MSG, DIET_LABELS, DIETS, PROFILE_QUERY, api_url,
                        food_com_label, fr, ingredient_img, ingredient_label, load_demo_users,
                        load_vocab)

user = auth.current_user()
if user is None:  # guests never see this page in the menu, but a stale URL could land here
    st.warning("Connecte-toi pour accéder à ton profil.")
    st.stop()


def fmt_date(iso: str | None) -> str:
    return datetime.fromisoformat(iso).astimezone().strftime("%d/%m/%Y à %H:%M") if iso else "—"


def save(**changes) -> None:
    try:
        auth.update_profile(user.username, **changes)
        st.session_state.flash = "Modifications enregistrées ✅"
        st.rerun()
    except auth.AuthError as e:
        st.error(str(e))


head, badge = st.columns([4, 1], vertical_alignment="center")
head.title(f"👤 {user.display_name}")
head.caption(f"@{user.username} · {auth.ROLES[user.role]} · membre depuis le "
             f"{fmt_date(user.created_at)[:10]} · dernière connexion le {fmt_date(user.last_login)}")

t_food, t_learned, t_perso, t_info, t_security = st.tabs(
    ["🥗 Préférences alimentaires", "🧠 Ce que le modèle sait de toi", "✨ Personnalisation",
     "📇 Informations", "🔒 Sécurité"])

# ----------------------------------------------------------- food preferences
with t_food:
    st.markdown("Ces préférences sont appliquées **automatiquement** à chacune de tes recherches. "
                "Les ingrédients incompatibles ne te seront jamais proposés.")
    with st.form("food"):
        c1, c2 = st.columns(2)
        diet_label = c1.selectbox("Régime alimentaire", list(DIETS),
                                  index=list(DIETS).index(DIET_LABELS.get(user.diet, "Aucun")))
        top_k = c2.slider("Nombre de suggestions par recherche", 1, 30, user.top_k)
        allergies = st.multiselect("Allergies et intolérances", list(ALLERGIES),
                                   default=user.allergies, format_func=ALLERGIES.get,
                                   placeholder="Aucune")
        disliked = st.multiselect("Ingrédients que je n'aime pas", load_vocab(),
                                  default=[d for d in user.disliked if d in set(load_vocab())],
                                  format_func=ingredient_label, placeholder="Aucun",
                                  help="Ils ne te seront jamais proposés. Le bouton « 👎 Je "
                                       "n'aime pas » sur une suggestion fait la même chose "
                                       "(voir l'onglet « Ce que le modèle sait de toi »).")
        if st.form_submit_button("Enregistrer mes préférences", type="primary"):
            save(diet=DIETS[diet_label], allergies=allergies, disliked=disliked, top_k=top_k)

# --------------------------------------------------------- what the model learned
def change_feedback(name: str, feedback: str, message: str) -> None:
    """Feedback given from the profile (no search): logged like any other, so the API applies it."""
    try:
        api_client.log_interaction(auth.log_user_id(user), [PROFILE_QUERY], feedback, name,
                                   base_url=api_url())
        st.toast(message)
    except api_client.APIError:
        st.toast("Impossible d'enregistrer : le serveur ne répond pas.", icon="❌")


def chips(items: list[dict], color: str) -> str:
    """Ingredient chips; the bar under each one shows how strong the preference is."""
    return "<div style='display:flex;flex-wrap:wrap;gap:.6rem'>" + "".join(
        f"<div style='display:flex;flex-direction:column;align-items:center;width:5rem;"
        f"font-size:.78rem;text-align:center;line-height:1.1'>"
        f"{ingredient_img(d['name'], 40).replace('margin-right:.4rem', 'margin:0 0 .2rem')}"
        f"{fr(d['name'])}<div style='height:4px;width:{max(8, min(100, abs(d['value']) / 1.5 * 100)):.0f}%;"
        f"background:{color};border-radius:2px;margin-top:.2rem'></div></div>"
        for d in items) + "</div>"


def feedback_list(names: list[str], button: str, feedback: str, message: str, key: str) -> None:
    for name in names:
        left, right = st.columns([4, 2], vertical_alignment="center")
        left.markdown(f"{ingredient_img(name, 28)}{fr(name).capitalize()}",
                      unsafe_allow_html=True)
        right.button(button, key=f"{key}|{name}", on_click=change_feedback,
                     args=(name, feedback, message.format(fr(name))), width="stretch")


with t_learned:
    st.markdown("Le modèle ajuste ses suggestions selon **tes avis** (👍 👎 ⚠️ sur les "
                "suggestions)" + (" et **les recettes notées par ton profil Food.com**"
                                  if user.food_com_id else "") + ". Voici ce qu'il retient.")
    try:
        prof = api_client.profile(auth.log_user_id(user), auth.history_user_id(user), n=12,
                                  base_url=api_url())
    except api_client.APIError:
        st.error(API_DOWN_MSG)
        prof = None
    if prof:
        fb = prof.get("feedback") or {}
        if prof.get("mock"):
            st.info("🧪 Le vrai modèle n'est pas chargé sur l'API : seuls tes avis sont listés.")
        else:
            liked_col, disliked_col = st.columns(2)
            with liked_col:
                st.markdown("##### 💚 Ce qu'il pense que tu aimes")
                if prof["likes"]:
                    st.markdown(chips(prof["likes"], "#7FAF3A"), unsafe_allow_html=True)
                    st.caption("Tes 👍 et les ingrédients qui leur ressemblent"
                               + (", plus tes recettes Food.com" if prof.get("has_rating_history")
                                  else "") + ".")
                else:
                    st.caption("Rien pour l'instant : mets 👍 sur des suggestions.")
            with disliked_col:
                st.markdown("##### 💔 Ce qu'il pense que tu n'aimes pas")
                if prof["dislikes"]:
                    st.markdown(chips(prof["dislikes"], "#D9534F"), unsafe_allow_html=True)
                else:
                    st.caption("Rien pour l'instant.")
            if user.food_com_id and not prof.get("has_rating_history"):
                st.warning("Ton profil Food.com n'a pas de recettes notées connues du modèle : "
                           "la personnalisation ne vient que de tes avis.")

        st.divider()
        st.markdown("##### Tes avis, un par un")
        st.caption("Tu peux changer d'avis à tout moment : c'est pris en compte dès ta "
                   "prochaine recherche.")
        a, b, c = st.columns(3)
        with a:
            st.markdown(f"**👍 Aimés ({len(fb.get('liked', []))})**")
            feedback_list(fb.get("liked", []), "👎 Plus maintenant", "dislike",
                          "{} ne te sera plus proposé.", "unlike")
        with b:
            st.markdown(f"**👎 Écartés ({len(fb.get('disliked', []))})**")
            feedback_list(fb.get("disliked", []), "👍 En fait j'aime", "like",
                          "{} peut de nouveau t'être proposé.", "undislike")
        with c:
            st.markdown(f"**⚠️ Allergies ({len(fb.get('allergic', []))})**")
            feedback_list(fb.get("allergic", []), "✅ Pas allergique", "not_allergic",
                          "Allergie à {} retirée.", "unallergic")
        if not any(fb.get(k) for k in ("liked", "disliked", "allergic")):
            st.info("Aucun avis pour l'instant. Va sur **Trouver des accords** et donne ton avis "
                    "sur les suggestions.")
        st.caption("Les « 🚫 Pas avec ça » ne comptent que pour la recherche où tu les as "
                   "donnés : ils sont dans ton historique.")

# ------------------------------------------------------------ personalization
with t_perso:
    st.markdown(
        "Le modèle apprend les goûts des utilisateurs à partir de leurs **recettes notées sur "
        "Food.com**. En reliant ton compte à un profil Food.com, les suggestions sont réordonnées "
        "selon ces goûts.")
    demo = load_demo_users()
    options = ["none"] + demo + ["custom"]
    current = user.food_com_id
    idx = options.index(current) if current in demo else (len(options) - 1 if current else 0)
    with st.form("perso"):
        choice = st.radio("Profil Food.com", options, index=idx, format_func=lambda o: {
            "none": "Aucun : suggestions générales", "custom": "Autre identifiant Food.com…"}
            .get(o) or food_com_label(o))
        custom = st.text_input("Identifiant Food.com", value=current if current and
                               current not in demo else "", placeholder="ex. 1533",
                               help="Utilisé seulement si « Autre identifiant » est coché.")
        if st.form_submit_button("Enregistrer", type="primary"):
            fid = None if choice == "none" else (custom.strip() or None) \
                if choice == "custom" else choice
            if choice == "custom" and not fid:
                st.error("Indique un identifiant Food.com.")
            else:
                save(food_com_id=fid)
    if current:
        st.success(f"✨ Relié à : {food_com_label(current)}")

# -------------------------------------------------------------------- info
with t_info:
    with st.form("info"):
        name = st.text_input("Prénom et nom", value=user.display_name)
        email = st.text_input("Adresse e-mail", value=user.email)
        st.text_input("Identifiant de connexion", value=user.username, disabled=True,
                      help="L'identifiant ne peut pas être modifié.")
        if st.form_submit_button("Enregistrer", type="primary"):
            save(display_name=name, email=email)

# ---------------------------------------------------------------- security
with t_security:
    with st.form("password", clear_on_submit=True):
        st.markdown("**Changer mon mot de passe**")
        current_pw = st.text_input("Mot de passe actuel", type="password")
        new_pw = st.text_input("Nouveau mot de passe", type="password",
                               help=f"Au moins {auth.MIN_PASSWORD} caractères")
        confirm_pw = st.text_input("Confirmer le nouveau mot de passe", type="password")
        if st.form_submit_button("Changer le mot de passe", type="primary"):
            try:
                auth.change_password(user.username, current_pw, new_pw, confirm_pw)
                st.success("Mot de passe changé ✅")
            except auth.AuthError as e:
                st.error(str(e))
    st.caption("🔒 Ton mot de passe n'est jamais stocké en clair : seule une empreinte salée "
               "est gardée.")
