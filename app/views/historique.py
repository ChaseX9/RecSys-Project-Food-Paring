"""Page "Mon historique": the logged-in user's feedback, read back through GET /interactions."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app import api_client, auth
from app.common import API_DOWN_MSG, FEEDBACK, PROFILE_QUERY, api_url, fr

user = auth.current_user()
if user is None:
    st.warning("Connecte-toi pour voir ton historique.")
    st.stop()

head, refresh = st.columns([4, 1], vertical_alignment="bottom")
head.title("📜 Mon historique")
refresh.button("⟳ Actualiser", width="stretch")
st.markdown("Tous les avis que tu as donnés sur les suggestions.")

try:
    rows = api_client.interactions(user_id=user.username, limit=1000, base_url=api_url())
except api_client.APIError:
    st.error(API_DOWN_MSG)
    st.stop()

if not rows:
    st.info("Aucun avis pour l'instant. Va sur **Trouver des accords** et clique sur 👍 ou 👎.")
    st.stop()

df = pd.DataFrame(rows)
counts = df["feedback"].value_counts()
shown = {fb: v for fb, v in FEEDBACK.items() if counts.get(fb, 0) or fb in ("like", "dislike")}
for col, (fb, (icon, label)) in zip(st.columns(len(shown)), shown.items()):
    col.metric(f"{icon} {label}", int(counts.get(fb, 0)))

st.dataframe(pd.DataFrame({
    "Date": pd.to_datetime(df["created_at"], utc=True).dt.tz_convert("Europe/Paris")
              .dt.strftime("%d/%m/%Y %H:%M"),
    "Recherche": df["query_ingredients"].map(
        lambda q: "depuis mon profil" if q == [PROFILE_QUERY] else " + ".join(fr(i) for i in q)),
    "Suggestion": df["recommended_ingredient"].map(lambda n: fr(n) if isinstance(n, str) else "—"),
    "Avis": df["feedback"].map(lambda f: " ".join(FEEDBACK.get(f, ("", f)))),
    "Note": df["note"].fillna(""),
}), hide_index=True, width="stretch")
