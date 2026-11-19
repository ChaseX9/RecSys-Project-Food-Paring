"""Page "Administration" (admins only): overview, users, logs, services & configuration."""
from __future__ import annotations

import platform
import subprocess
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

from app import api_client, auth
from app.common import FEEDBACK, ROOT, api_url, fr, user_label
from src.ranking.hybrid import RankingConfig

admin = auth.current_user()
if admin is None or not admin.is_admin:
    st.error("⛔ Cette page est réservée aux administrateurs.")
    st.stop()

LEVELS = {"info": "ℹ️ info", "warning": "⚠️ attention", "error": "❌ erreur"}


def local(iso: str | None) -> str:
    return datetime.fromisoformat(iso).astimezone().strftime("%d/%m/%Y %H:%M:%S") if iso else "—"


def run(action, *args, success: str) -> None:
    """Run an admin action and report the outcome after the rerun."""
    try:
        action(admin.username, *args)
        st.session_state.flash = success
        st.rerun()
    except auth.AuthError as e:
        st.error(str(e))


@st.cache_data(ttl=30, show_spinner=False)
def model_mode(url: str) -> bool | None:
    """True if /recommend still serves the mock, False if the real model, None if unreachable."""
    try:
        body, _ = api_client.recommend(["tomato"], auth.GUEST, 1, base_url=url)
        return bool(body.get("mock"))
    except api_client.APIError:
        return None


def fetch_interactions() -> list[dict] | None:
    try:
        return api_client.interactions(limit=1000, base_url=api_url())
    except api_client.APIError:
        return None


st.title("🛠️ Administration")
st.caption(f"Connecté en tant que {admin.display_name} (@{admin.username})")

t_overview, t_users, t_logs, t_services = st.tabs(
    ["📊 Vue d'ensemble", "👥 Utilisateurs", "📜 Journaux", "🔌 Services & configuration"])

# ------------------------------------------------------------------ overview
with t_overview:
    users = auth.all_users()
    events = auth.events()
    ok, ms = api_client.ping(api_url())
    mode = model_mode(api_url()) if ok else None
    rows = fetch_interactions() if ok else None
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    recent = [e for e in events if datetime.fromisoformat(e["date"]) >= since]

    c = st.columns(4)
    c[0].metric("Serveur (API)", "🟢 En ligne" if ok else "🔴 Hors ligne",
                help=f"Temps de réponse de /health : {ms:.0f} ms" if ok else None)
    c[0].caption(f"Réponse en {ms:.0f} ms" if ok else "Lancer `uvicorn api.main:app`")
    c[1].metric("Modèle", {True: "🧪 Factice", False: "✅ Réel", None: "—"}[mode],
                help="Factice tant que le vrai modèle n'est pas branché sur /recommend")
    c[2].metric("Comptes actifs", f"{sum(u['active'] for u in users)} / {len(users)}")
    c[2].caption(f"dont {sum(u['role'] == 'admin' for u in users)} administrateur(s)")
    c[3].metric("Avis enregistrés", "—" if rows is None else
                (f"{len(rows)}+" if len(rows) >= 1000 else len(rows)))

    c = st.columns(4)
    c[0].metric("Connexions (24 h)", sum(e["message"] == "Connexion" for e in recent))
    c[1].metric("Échecs de connexion (24 h)",
                sum(e["message"].startswith("Échec de connexion") for e in recent))
    calls = list(api_client.CALLS)
    c[2].metric("Appels API (session serveur)", len(calls))
    c[3].metric("Appels API en erreur", sum(not x["ok"] for x in calls))

    st.subheader("Dernière activité")
    st.dataframe(pd.DataFrame([{"Date": local(e["date"]), "Niveau": LEVELS.get(e["niveau"]),
                                "Qui": e["acteur"], "Événement": e["message"]}
                               for e in events[:8]]), hide_index=True, width="stretch")

# --------------------------------------------------------------------- users
with t_users:
    users = auth.all_users()
    st.dataframe(pd.DataFrame([{
        "Identifiant": u["username"], "Nom": u["display_name"], "E-mail": u["email"],
        "Rôle": auth.ROLES[u["role"]], "Statut": "✅ actif" if u["active"] else "⛔ désactivé",
        "Régime": u["diet"] or "—", "Profil Food.com": u["food_com_id"] or "—",
        "Créé le": local(u["created_at"])[:10], "Dernière connexion": local(u["last_login"]),
    } for u in users]), hide_index=True, width="stretch")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Gérer un compte")
        target = st.selectbox("Compte", [u["username"] for u in users],
                              format_func=lambda n: f"{user_label(n)} (@{n})")
        tu = next(u for u in users if u["username"] == target)
        is_me = target == admin.username

        role = st.radio("Rôle", list(auth.ROLES), index=list(auth.ROLES).index(tu["role"]),
                        format_func=auth.ROLES.get, horizontal=True, key=f"role_{target}")
        if role != tu["role"]:
            run(auth.admin_set_role, target, role,
                success=f"Rôle de @{target} : {auth.ROLES[role]}")

        b1, b2 = st.columns(2)
        if tu["active"]:
            if b1.button("⛔ Désactiver le compte", disabled=is_me, width="stretch"):
                run(auth.admin_set_active, target, False, success=f"@{target} désactivé")
        elif b1.button("✅ Réactiver le compte", width="stretch"):
            run(auth.admin_set_active, target, True, success=f"@{target} réactivé")
        if b2.button("🔑 Réinitialiser le mot de passe", width="stretch"):
            try:
                temp = auth.admin_reset_password(admin.username, target)
                st.success(f"Nouveau mot de passe temporaire pour @{target} : `{temp}`  \n"
                           "Transmets-le à l'utilisateur : il pourra le changer dans son profil.")
            except auth.AuthError as e:
                st.error(str(e))
        with st.popover("🗑️ Supprimer le compte", disabled=is_me, width="stretch"):
            st.warning(f"Supprimer définitivement @{target} ? Ses avis déjà enregistrés "
                       "sont conservés.")
            if st.button("Oui, supprimer", type="primary", key=f"del_{target}"):
                run(auth.admin_delete, target, success=f"@{target} supprimé")
        if is_me:
            st.caption("Tu ne peux pas désactiver ni supprimer ton propre compte.")

    with right:
        st.subheader("Créer un compte")
        with st.form("create_user", clear_on_submit=True):
            c1, c2 = st.columns(2)
            nu = c1.text_input("Identifiant")
            nd = c2.text_input("Prénom et nom")
            ne = st.text_input("Adresse e-mail")
            c3, c4 = st.columns(2)
            npw = c3.text_input("Mot de passe", type="password")
            nr = c4.selectbox("Rôle", list(auth.ROLES), format_func=auth.ROLES.get)
            if st.form_submit_button("Créer", type="primary"):
                try:
                    auth.register(nu, nd, ne, npw, npw, role=nr, actor=admin.username)
                    st.session_state.flash = f"Compte @{nu.strip().lower()} créé"
                    st.rerun()
                except auth.AuthError as e:
                    st.error(str(e))

# ---------------------------------------------------------------------- logs
with t_logs:
    l_app, l_feedback, l_api = st.tabs(
        ["Journal de l'application", "Avis des utilisateurs", "Appels à l'API"])

    with l_app:
        c1, c2 = st.columns([1, 2])
        levels = c1.multiselect("Niveau", list(LEVELS), default=list(LEVELS),
                                format_func=LEVELS.get)
        search = c2.text_input("Rechercher", placeholder="ex. connexion, camille…")
        ev = [e for e in auth.events() if e["niveau"] in levels and
              (not search or search.lower() in (e["acteur"] + " " + e["message"]).lower())]
        df = pd.DataFrame([{"Date": local(e["date"]), "Niveau": LEVELS.get(e["niveau"]),
                            "Qui": e["acteur"], "Événement": e["message"]} for e in ev])
        st.dataframe(df, hide_index=True, width="stretch", height=380)
        st.download_button("⬇️ Exporter (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                           "journal_application.csv", "text/csv", disabled=df.empty)

    with l_feedback:
        rows = fetch_interactions()
        if rows is None:
            st.error("Serveur de recommandations injoignable : impossible de lire les avis.")
        elif not rows:
            st.info("Aucun avis enregistré pour l'instant.")
        else:
            df = pd.DataFrame(rows)
            who = st.selectbox("Utilisateur", ["Tous"] + sorted(df["user_id"].unique()),
                               format_func=lambda u: u if u == "Tous" else
                               f"{user_label(u)} ({u})")
            if who != "Tous":
                df = df[df["user_id"] == who]
            counts = df["feedback"].value_counts()
            for col, (fb, (icon, label)) in zip(st.columns(len(FEEDBACK)), FEEDBACK.items()):
                col.metric(f"{icon} {label}", int(counts.get(fb, 0)))
            view = pd.DataFrame({
                "N°": df["id"], "Utilisateur": df["user_id"].map(user_label),
                "Recherche": df["query_ingredients"].map(lambda q: " + ".join(fr(i) for i in q)),
                "Suggestion": df["recommended_ingredient"].map(
                    lambda n: fr(n) if isinstance(n, str) else "—"),
                "Avis": df["feedback"].map(lambda f: " ".join(FEEDBACK.get(f, ("", f)))),
                "Note": df["note"].fillna(""), "Date": df["created_at"].map(local),
            })
            st.dataframe(view, hide_index=True, width="stretch", height=380)
            st.download_button("⬇️ Exporter (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                               "avis_utilisateurs.csv", "text/csv")

    with l_api:
        calls = list(api_client.CALLS)
        hide_health = st.toggle("Masquer les vérifications de santé (/health)", value=True)
        if hide_health:
            calls = [c for c in calls if c["route"] != "/health"]
        if not calls:
            st.info("Aucun appel enregistré depuis le démarrage du serveur Streamlit.")
        else:
            df = pd.DataFrame(calls)
            summary = df.groupby(["méthode", "route"]).agg(
                appels=("ms", "size"), erreurs=("ok", lambda s: int((~s).sum())),
                moyenne_ms=("ms", "mean"), p95_ms=("ms", lambda s: s.quantile(.95)),
                max_ms=("ms", "max")).round(1).reset_index()
            st.markdown("**Par route**")
            st.dataframe(summary, hide_index=True, width="stretch")
            st.markdown("**Derniers appels**")
            df["date"] = df["date"].map(local)
            df["ok"] = df["ok"].map({True: "✅", False: "❌"})
            st.dataframe(df, hide_index=True, width="stretch", height=300)

# ------------------------------------------------------- services & config
with t_services:
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Serveur de recommandations (API)")
        with st.form("api_url"):
            url = st.text_input("Adresse de l'API", value=api_url(),
                                help="Appliquée à tous les utilisateurs de l'application.")
            if st.form_submit_button("Enregistrer l'adresse"):
                run(auth.admin_set_api_url, url, success="Adresse de l'API mise à jour")

        if st.button("🔌 Tester la connexion", type="primary"):
            checks = []
            def check_health():
                if not api_client.health(api_url()):
                    raise api_client.APIError("injoignable ou statut ≠ ok")

            for label, fn in [
                ("GET /health", check_health),
                ("POST /recommend", lambda: api_client.recommend(["tomato"], auth.GUEST, 3,
                                                                 base_url=api_url())),
                ("GET /interactions", lambda: api_client.interactions(limit=1,
                                                                      base_url=api_url())),
            ]:
                start = time.perf_counter()
                try:
                    fn()
                    checks.append({"Route": label, "Résultat": "✅ OK",
                                   "Temps": f"{(time.perf_counter() - start) * 1000:.0f} ms"})
                except api_client.APIError as e:
                    checks.append({"Route": label, "Résultat": f"❌ {e}"[:120], "Temps": "—"})
            st.dataframe(checks, hide_index=True, width="stretch")
            st.caption("Objectif du projet : moins de 2 s par recommandation.")

    with right:
        st.subheader("Modèle et données")
        emb = ROOT / "artifacts" / "ingredient_embeddings.npy"
        if emb.exists():
            trained = datetime.fromtimestamp(emb.stat().st_mtime).strftime("%d/%m/%Y %H:%M")
            st.markdown(f"- Embeddings entraînés : ✅ `artifacts/{emb.name}` ({trained})")
        else:
            st.markdown("- Embeddings entraînés : ❌ absents (lancer `scripts/train_nlgcl.py`)")
        cfg = RankingConfig()
        st.markdown(f"- Pondération du classement : habitudes culinaires **{cfg.alpha_emb}** · "
                    f"chimie des arômes **{cfg.beta_chem}** · goûts **{cfg.gamma_pref}**")
        try:
            commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                    capture_output=True, text=True, timeout=5).stdout.strip()
            branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT,
                                    capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            commit = branch = ""
        st.markdown(f"- Version du code : `{branch or '?'}` @ `{commit or '?'}`")
        st.markdown(f"- Python {platform.python_version()} · Streamlit {st.__version__}")

        st.subheader("Comptes")
        st.info("🧪 Les comptes sont factices et gardés en mémoire (pas encore de base de "
                "données) : ils sont remis à zéro au redémarrage du serveur Streamlit. "
                "Les avis, eux, sont bien stockés dans la base SQLite de l'API.")
