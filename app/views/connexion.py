"""Page "Connexion": sign in, sign up, continue as a guest, or use a demo account."""
from __future__ import annotations

import streamlit as st

from app import auth
from app.i18n import t

DESCRIPTIONS = {"admin": "🛠️ Accès au panneau d'administration",
                "demo": "✨ Relié à un profil Food.com, frigo déjà rempli",
                "camille": "🥗 Végétarienne, allergique aux fruits à coque"}


def demo_login(username: str, password: str) -> None:
    try:  # can fail if an admin deactivated or changed this demo account
        u = auth.login(username, password)
        st.session_state.flash = t("Bonjour {name} ! 👋", name=u.display_name)
    except auth.AuthError as e:
        st.session_state.flash = f"❌ {t(str(e))}"


st.title(t("👋 Bienvenue !"))
st.markdown(t("Connecte-toi pour retrouver **tes préférences** (régime, allergies…), **ton frigo** "
              "et **ton historique**. Tu peux aussi essayer l'application sans compte."))

form_col, demo_col = st.columns([3, 2], gap="large")

with form_col:
    login_tab, signup_tab = st.tabs([t("Se connecter"), t("Créer un compte")])

    with login_tab:
        with st.form("login", border=False):
            username = st.text_input(t("Identifiant"), placeholder=t("ex. demo"))
            password = st.text_input(t("Mot de passe"), type="password")
            if st.form_submit_button(t("Se connecter"), type="primary", width="stretch"):
                try:
                    u = auth.login(username, password)
                    st.session_state.flash = t("Bonjour {name} ! 👋", name=u.display_name)
                    st.rerun()
                except auth.AuthError as e:
                    st.error(t(str(e)))

    with signup_tab:
        with st.form("signup", border=False):
            c1, c2 = st.columns(2)
            new_user = c1.text_input(t("Identifiant"), placeholder=t("ex. marie.d"),
                                     help=t("3 à 30 caractères : minuscules, chiffres, . _ -"))
            display = c2.text_input(t("Prénom et nom"), placeholder=t("ex. Marie Dupont"))
            email = st.text_input(t("Adresse e-mail"), placeholder=t("marie@exemple.fr"))
            c3, c4 = st.columns(2)
            pw = c3.text_input(t("Mot de passe"), type="password",
                               help=t("Au moins {n} caractères", n=auth.MIN_PASSWORD))
            pw2 = c4.text_input(t("Confirmer le mot de passe"), type="password")
            if st.form_submit_button(t("Créer mon compte"), type="primary", width="stretch"):
                try:
                    auth.register(new_user, display, email, pw, pw2)
                    u = auth.login(new_user, pw)
                    st.session_state.flash = t("Compte créé, bienvenue {name} ! 🎉",
                                               name=u.display_name)
                    st.session_state.onboarding = True  # first stop: the profile (allergies)
                    st.rerun()
                except auth.AuthError as e:
                    st.error(t(str(e)))

    st.divider()
    st.button(t("Continuer sans compte"), icon="🚶", on_click=auth.login_as_guest, width="stretch",
              help=t("Tu pourras chercher des accords, mais pas enregistrer tes préférences."))

with demo_col:
    st.subheader(t("🧪 Comptes de démonstration"))
    st.caption(t("Recopie un identifiant et son mot de passe, ou clique sur « Se connecter ». "
                 "Ces comptes sont créés au premier démarrage de l'API."))
    for username, password, extra in auth.DEMO_ACCOUNTS:
        with st.container(border=True):
            info, action = st.columns([3, 2], vertical_alignment="center")
            info.markdown(
                f"**{extra['display_name']}** · <span style='font-size:.85rem;opacity:.7'>"
                f"{t(auth.ROLES[extra.get('role', 'user')])}</span>  \n"
                + t("Identifiant : `{u}`  \nMot de passe : `{p}`", u=username, p=password) + "  \n"
                f"<span style='font-size:.85rem;opacity:.7'>{t(DESCRIPTIONS.get(username, ''))}"
                f"</span>", unsafe_allow_html=True)
            action.button(t("Se connecter"), key=f"demo_{username}", on_click=demo_login,
                          args=(username, password), width="stretch")
