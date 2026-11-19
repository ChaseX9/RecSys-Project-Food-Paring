"""Page "Connexion": sign in, sign up, continue as a guest, or use a demo account."""
from __future__ import annotations

import streamlit as st

from app import auth

DESCRIPTIONS = {"admin": "🛠️ Accès au panneau d'administration",
                "demo": "✨ Relié à un profil Food.com, frigo déjà rempli",
                "camille": "🥗 Végétarienne, allergique aux fruits à coque"}


def demo_login(username: str, password: str) -> None:
    try:  # can fail if an admin deactivated or changed this demo account
        u = auth.login(username, password)
        st.session_state.flash = f"Bonjour {u.display_name} ! 👋"
    except auth.AuthError as e:
        st.session_state.flash = f"❌ {e}"


st.title("👋 Bienvenue !")
st.markdown("Connecte-toi pour retrouver **tes préférences** (régime, allergies…), **ton frigo** "
            "et **ton historique**. Tu peux aussi essayer l'application sans compte.")

form_col, demo_col = st.columns([3, 2], gap="large")

with form_col:
    login_tab, signup_tab = st.tabs(["Se connecter", "Créer un compte"])

    with login_tab:
        with st.form("login", border=False):
            username = st.text_input("Identifiant", placeholder="ex. demo")
            password = st.text_input("Mot de passe", type="password")
            if st.form_submit_button("Se connecter", type="primary", width="stretch"):
                try:
                    u = auth.login(username, password)
                    st.session_state.flash = f"Bonjour {u.display_name} ! 👋"
                    st.rerun()
                except auth.AuthError as e:
                    st.error(str(e))

    with signup_tab:
        with st.form("signup", border=False):
            c1, c2 = st.columns(2)
            new_user = c1.text_input("Identifiant", placeholder="ex. marie.d",
                                     help="3 à 30 caractères : minuscules, chiffres, . _ -")
            display = c2.text_input("Prénom et nom", placeholder="ex. Marie Dupont")
            email = st.text_input("Adresse e-mail", placeholder="marie@exemple.fr")
            c3, c4 = st.columns(2)
            pw = c3.text_input("Mot de passe", type="password",
                               help=f"Au moins {auth.MIN_PASSWORD} caractères")
            pw2 = c4.text_input("Confirmer le mot de passe", type="password")
            if st.form_submit_button("Créer mon compte", type="primary", width="stretch"):
                try:
                    auth.register(new_user, display, email, pw, pw2)
                    u = auth.login(new_user, pw)
                    st.session_state.flash = (f"Compte créé, bienvenue {u.display_name} ! "
                                              "Pense à remplir ton profil (régime, allergies).")
                    st.rerun()
                except auth.AuthError as e:
                    st.error(str(e))

    st.divider()
    st.button("Continuer sans compte", icon="🚶", on_click=auth.login_as_guest, width="stretch",
              help="Tu pourras chercher des accords, mais pas enregistrer tes préférences.")

with demo_col:
    st.subheader("🧪 Comptes de démonstration")
    st.caption("Recopie un identifiant et son mot de passe, ou clique sur « Se connecter ». "
               "Ces comptes sont créés au premier démarrage de l'API.")
    for username, password, extra in auth.DEMO_ACCOUNTS:
        with st.container(border=True):
            info, action = st.columns([3, 2], vertical_alignment="center")
            info.markdown(
                f"**{extra['display_name']}** · <span style='font-size:.85rem;opacity:.7'>"
                f"{auth.ROLES[extra.get('role', 'user')]}</span>  \n"
                f"Identifiant : `{username}`  \nMot de passe : `{password}`  \n"
                f"<span style='font-size:.85rem;opacity:.7'>{DESCRIPTIONS.get(username, '')}"
                f"</span>", unsafe_allow_html=True)
            action.button("Se connecter", key=f"demo_{username}", on_click=demo_login,
                          args=(username, password), width="stretch")
