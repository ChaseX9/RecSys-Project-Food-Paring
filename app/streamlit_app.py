"""Food Pairing — Streamlit UI entry point: auth gate, role-based navigation, theme, sidebar.

Run (from the repo root, with the API running):
    uvicorn api.main:app
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app import auth
from app.common import api_status, api_url, ensure_theme, logo_data_uri

st.set_page_config(page_title="Food Pairing", page_icon="🍽️", layout="wide")

st.session_state.setdefault("dark_mode", True)
ensure_theme(st.session_state.dark_mode)

user = auth.current_user()
guest = auth.is_guest()

# ------------------------------------------------------------- pages by role
P = {
    "login": st.Page("views/connexion.py", title="Connexion", icon="🔑", default=True),
    "accords": st.Page("views/accords.py", title="Trouver des accords", icon="🍽️",
                       default=True, url_path="accords"),
    "frigo": st.Page("views/frigo.py", title="Mon frigo", icon="🧊"),
    "historique": st.Page("views/historique.py", title="Mon historique", icon="📜"),
    "recettes": st.Page("views/recettes.py", title="Noter des recettes", icon="⭐"),
    "profil": st.Page("views/profil.py", title="Mon profil", icon="👤"),
    "admin": st.Page("views/admin.py", title="Administration", icon="🛠️"),
    "about": st.Page("views/a_propos.py", title="À propos", icon="ℹ️"),
}
if user and user.is_admin:
    visible = ["accords", "frigo", "recettes", "historique", "profil", "admin", "about"]
elif user:
    visible = ["accords", "frigo", "recettes", "historique", "profil", "about"]
elif guest:
    visible = ["accords", "frigo", "recettes", "about"]
else:
    visible = ["login", "about"]
nav = st.navigation([P[k] for k in visible], position="hidden")

# Pages can tell a fresh visit from a rerun (e.g. a new recipe to rate at every visit).
st.session_state.page_changed = st.session_state.get("current_page") != nav.url_path
st.session_state.current_page = nav.url_path

# Just signed up: go straight to the profile so allergies and diet are filled in first
# (the recipe-rating onboarding then respects them).
if user and st.session_state.get("onboarding") and "profil" in visible:
    st.session_state.onboarding_banner = True
    st.session_state.onboarding_redirected = True
    st.session_state.pop("onboarding")
    st.switch_page(P["profil"])

# Later logins until onboarding is done: send the user once to the recipe ratings,
# so the model learns their taste.
if user and not user.onboarding_done and not st.session_state.get("onboarding_redirected"):
    st.session_state.onboarding_redirected = True
    st.session_state.flash = ("Bienvenue ! Note quelques recettes pour que les suggestions "
                              "collent à tes goûts (tu peux passer).")
    st.switch_page(P["recettes"])

# -------------------------------------------------------------- sidebar top
uri = logo_data_uri(st.session_state.dark_mode)  # ECE logo, cut out and recoloured per theme
logo_html = f"<img src='{uri}' style='width:150px;margin-bottom:.8rem'><br>" if uri else ""

with st.sidebar:
    st.markdown(f"<div style='margin-top:-2.5rem'>{logo_html}"
                "<span style='font-size:1.6rem;font-weight:700'>🍽️ Food Pairing</span><br>"
                "<span style='opacity:.65;font-size:.9rem'>Les ingrédients qui vont bien "
                "ensemble</span></div>", unsafe_allow_html=True)
    st.write("")
    for key in visible:
        st.page_link(P[key])
    st.divider()
    st.toggle("🌙 Mode sombre", key="dark_mode")

if "flash" in st.session_state:  # one-shot message set before a st.rerun()
    st.toast(st.session_state.pop("flash"))

nav.run()

# ----------------------------------------------------------- sidebar bottom
with st.sidebar:
    st.divider()
    if user or guest:
        name = user.display_name if user else "Invité"
        initials = user.initials if user else "?"
        role = auth.ROLES[user.role] if user else "Sans compte"
        color = "#E0A040" if user and user.is_admin else "#7FAF3A"
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:.7rem;margin-bottom:.6rem'>"
            f"<div style='width:2.4rem;height:2.4rem;border-radius:50%;background:{color};"
            f"color:#111;display:flex;align-items:center;justify-content:center;font-weight:700'>"
            f"{initials}</div><div style='line-height:1.2'><b>{name}</b><br>"
            f"<span style='font-size:.8rem;opacity:.7'>{role}</span></div></div>",
            unsafe_allow_html=True)
        if user:
            st.button("Se déconnecter", icon="🚪", on_click=auth.logout, width="stretch")
        else:
            st.button("Se connecter / créer un compte", icon="🔑", on_click=auth.logout,
                      width="stretch")
    ok, ms = api_status(api_url())
    st.caption(f"🟢 Serveur connecté · {ms:.0f} ms" if ok
               else "🔴 Serveur de recommandations injoignable")
