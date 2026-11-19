"""Page "À propos": school project context, team, how the recommender works."""
from __future__ import annotations

import json

import streamlit as st

from app.common import PROCESSED, logo_data_uri

TEAM = [
    ("Hugo Riviere", "Chef de projet", "🧭"),
    ("Mael Vaudin", "Data Engineer", "🗄️"),
    ("Jade Planterose", "ML Engineer — Modèle", "🧠"),
    ("Thibault Rabbé", "ML Engineer — Ops", "⚙️"),
]


@st.cache_data
def load_stats() -> dict:
    try:
        return json.loads((PROCESSED / "stats.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


head, logo = st.columns([3, 1], vertical_alignment="center")
with head:
    st.title("ℹ️ À propos")
    st.markdown("#### Un projet étudiant de l'**ECE**, ING5, majeure **Data & IA** (2026-2027)")
uri = logo_data_uri(st.session_state.get("dark_mode", True))
if uri:  # logo cut out of its white background, recoloured for the current theme
    logo.markdown(f'<img src="{uri}" style="width:100%;max-width:260px;height:auto">',
                  unsafe_allow_html=True)

st.markdown(
    "**Food Pairing** répond à une question de tous les jours : *« J'ai tel ingrédient, "
    "qu'est-ce qui va bien avec ? »*. Tu donnes un ou plusieurs ingrédients, l'application te "
    "propose ceux qui s'accordent le mieux, en tenant compte de tes **allergies**, de ton "
    "**régime** et de tes **goûts**, avec une **explication** pour chaque suggestion.\n\n"
    "Ce projet est réalisé dans le cadre du cours de **systèmes de recommandation** "
    "par le **groupe 5, « Pull_request »**."
)

st.subheader("👩‍🍳 L'équipe")
for col, (name, role, icon) in zip(st.columns(len(TEAM)), TEAM):
    with col.container(border=True):
        st.markdown(f"### {icon}")
        st.markdown(f"**{name}**")
        st.caption(role)

st.subheader("🔬 Comment ça marche ?")
c1, c2, c3 = st.columns(3)
with c1.container(border=True):
    st.markdown("**1. Les données**")
    st.markdown("Des centaines de milliers de recettes et d'avis d'utilisateurs du site "
                "**Food.com**, complétées par les **molécules aromatiques** de chaque "
                "ingrédient (FlavorGraph).")
with c2.container(border=True):
    st.markdown("**2. Le modèle**")
    st.markdown("Un **réseau de neurones sur graphe** apprend quels ingrédients vont bien "
                "ensemble. Il s'inspire de **NLGCL** (Xu et al., RecSys 2025), une méthode "
                "d'apprentissage contrastif récente.")
with c3.container(border=True):
    st.markdown("**3. Le classement**")
    st.markdown("Chaque suggestion combine trois signaux : les **habitudes culinaires**, la "
                "**chimie des arômes** et **tes goûts**. Les ingrédients incompatibles avec tes "
                "allergies ou ton régime sont retirés.")

stats = load_stats()
if stats:
    st.subheader("📊 En chiffres")
    m = st.columns(4)
    m[0].metric("Recettes analysées", f"{stats.get('recipes_kept', 0):,}".replace(",", " "))
    m[1].metric("Ingrédients connus", f"{stats.get('ingredients_vocab_size', 0):,}".replace(",", " "))
    m[2].metric("Utilisateurs", f"{stats.get('n_users', 0):,}".replace(",", " "))
    m[3].metric("Avis utilisés", f"{stats.get('interactions_after_kcore', 0):,}".replace(",", " "))

st.divider()
st.caption("Projet pédagogique, sans but commercial. Les suggestions sont générées "
           "automatiquement : vérifie toujours la composition des produits si tu as une allergie.  \n"
           "Photos des ingrédients : [TheMealDB](https://www.themealdb.com) (usage pédagogique).")
