"""Page "Mon profil": personal info, food preferences, personalization link, password."""
from __future__ import annotations

from datetime import datetime

import streamlit as st

from app import auth
from app.common import (ALLERGIES, DIET_LABELS, DIETS, food_com_label, ingredient_label,
                        load_demo_users, load_vocab)

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

t_food, t_perso, t_info, t_security = st.tabs(
    ["🥗 Préférences alimentaires", "✨ Personnalisation", "📇 Informations", "🔒 Sécurité"])

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
                                  help="Ils ne te seront plus proposés. Le bouton « 🚫 Pas pour "
                                       "moi » sur une suggestion les ajoute ici.")
        if st.form_submit_button("Enregistrer mes préférences", type="primary"):
            save(diet=DIETS[diet_label], allergies=allergies, disliked=disliked, top_k=top_k)

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
    st.caption("🧪 Pour l'instant les comptes sont gardés en mémoire (pas encore de base de "
               "données) : ils sont remis à zéro au redémarrage du serveur.")
