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
from app.i18n import t, t_actor, t_log
from src.ranking.hybrid import RankingConfig

admin = auth.current_user()
if admin is None or not admin.is_admin:
    st.error(t("⛔ Cette page est réservée aux administrateurs."))
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
        st.error(t(str(e)))


@st.cache_data(ttl=30, show_spinner=False)
def model_mode(url: str) -> bool | None:
    """True if /recommend still serves the mock, False if the real model, None if unreachable."""
    try:
        body, _ = api_client.recommend(["tomato"], auth.GUEST, 1, base_url=url)
        return bool(body.get("mock"))
    except api_client.APIError:
        return None


def fetch_users() -> list[dict]:
    """All accounts, or [] if the API cannot be reached (the page must still render, e.g. to
    fix the API address in the last tab)."""
    try:
        return auth.all_users()
    except auth.AuthError:
        return []


def fetch_interactions() -> list[dict] | None:
    try:
        return api_client.interactions(limit=1000, base_url=api_url())
    except api_client.APIError:
        return None


st.title(t("🛠️ Administration"))
st.caption(t("Connecté en tant que {name} (@{user})", name=admin.display_name, user=admin.username))

t_overview, t_users, t_logs, t_services = st.tabs(
    [t("📊 Vue d'ensemble"), t("👥 Utilisateurs"), t("📜 Journaux"),
     t("🔌 Services & configuration")])

# ------------------------------------------------------------------ overview
with t_overview:
    users = fetch_users()
    events = auth.events()
    ok, ms = api_client.ping(api_url())
    mode = model_mode(api_url()) if ok else None
    rows = fetch_interactions() if ok else None
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    recent = [e for e in events if datetime.fromisoformat(e["date"]) >= since]

    c = st.columns(4)
    c[0].metric(t("Serveur (API)"), t("🟢 En ligne") if ok else t("🔴 Hors ligne"),
                help=t("Temps de réponse de /health : {ms} ms", ms=f"{ms:.0f}") if ok else None)
    c[0].caption(t("Réponse en {ms} ms", ms=f"{ms:.0f}") if ok else t("Lancer `uvicorn api.main:app`"))
    c[1].metric(t("Modèle"), t({True: "🧪 Factice", False: "✅ Réel", None: "—"}[mode]),
                help=t("Factice tant que le vrai modèle n'est pas branché sur /recommend"))
    c[2].metric(t("Comptes actifs"), f"{sum(u['active'] for u in users)} / {len(users)}")
    c[2].caption(t("dont {n} administrateur(s)", n=sum(u["role"] == "admin" for u in users)))
    c[3].metric(t("Avis enregistrés"), "—" if rows is None else
                (f"{len(rows)}+" if len(rows) >= 1000 else len(rows)))

    c = st.columns(4)
    c[0].metric(t("Connexions (24 h)"), sum(e["message"] == "Connexion" for e in recent))
    c[1].metric(t("Échecs de connexion (24 h)"),
                sum(e["message"].startswith("Échec de connexion") for e in recent))
    calls = list(api_client.CALLS)
    c[2].metric(t("Appels API (session serveur)"), len(calls))
    c[3].metric(t("Appels API en erreur"), sum(not x["ok"] for x in calls))

    st.subheader(t("Dernière activité"))
    st.dataframe(pd.DataFrame([{t("Date"): local(e["date"]), t("Niveau"): t(LEVELS.get(e["niveau"])),
                                t("Qui"): t_actor(e["acteur"]), t("Événement"): t_log(e["message"])}
                               for e in events[:8]]), hide_index=True, width="stretch")

# --------------------------------------------------------------------- users
with t_users:
    users = fetch_users()
    st.dataframe(pd.DataFrame([{
        t("Identifiant"): u["username"], t("Nom"): u["display_name"], t("E-mail"): u["email"],
        t("Rôle"): t(auth.ROLES[u["role"]]),
        t("Statut"): t("✅ actif") if u["active"] else t("⛔ désactivé"),
        t("Régime"): u["diet"] or "—", t("Profil Food.com"): u["food_com_id"] or "—",
        t("Créé le"): local(u["created_at"])[:10], t("Dernière connexion"): local(u["last_login"]),
    } for u in users]), hide_index=True, width="stretch")

    if not users:
        st.error(t("Serveur injoignable : impossible de lire les comptes. Vérifie l'adresse de l'API "
                   "dans « Services & configuration »."))
    else:
        left, right = st.columns(2, gap="large")
        with left:
            st.subheader(t("Gérer un compte"))
            target = st.selectbox(t("Compte"), [u["username"] for u in users],
                                  format_func=lambda n: f"{user_label(n)} (@{n})")
            tu = next(u for u in users if u["username"] == target)
            is_me = target == admin.username

            role = st.radio(t("Rôle"), list(auth.ROLES), index=list(auth.ROLES).index(tu["role"]),
                            format_func=lambda r: t(auth.ROLES[r]), horizontal=True, key=f"role_{target}")
            if role != tu["role"]:
                run(auth.admin_set_role, target, role,
                    success=t("Rôle de @{user} : {role}", user=target, role=t(auth.ROLES[role])))

            b1, b2 = st.columns(2)
            if tu["active"]:
                if b1.button(t("⛔ Désactiver le compte"), disabled=is_me, width="stretch"):
                    run(auth.admin_set_active, target, False, success=t("@{user} désactivé", user=target))
            elif b1.button(t("✅ Réactiver le compte"), width="stretch"):
                run(auth.admin_set_active, target, True, success=t("@{user} réactivé", user=target))
            if b2.button(t("🔑 Réinitialiser le mot de passe"), width="stretch"):
                try:
                    temp = auth.admin_reset_password(admin.username, target)
                    st.success(t("Nouveau mot de passe temporaire pour @{user} : `{temp}`  \nTransmets-le à "
                                 "l'utilisateur : il pourra le changer dans son profil.",
                                 user=target, temp=temp))
                except auth.AuthError as e:
                    st.error(t(str(e)))
            with st.popover(t("🗑️ Supprimer le compte"), disabled=is_me, width="stretch"):
                st.warning(t("Supprimer définitivement @{user} ? Ses avis (j'aime, allergies…) sont "
                             "supprimés avec le compte.", user=target))
                if st.button(t("Oui, supprimer"), type="primary", key=f"del_{target}"):
                    run(auth.admin_delete, target, success=t("@{user} supprimé", user=target))
            if is_me:
                st.caption(t("Tu ne peux pas désactiver ni supprimer ton propre compte."))

        with right:
            st.subheader(t("Créer un compte"))
            with st.form("create_user", clear_on_submit=True):
                c1, c2 = st.columns(2)
                nu = c1.text_input(t("Identifiant"))
                nd = c2.text_input(t("Prénom et nom"))
                ne = st.text_input(t("Adresse e-mail"))
                c3, c4 = st.columns(2)
                npw = c3.text_input(t("Mot de passe"), type="password")
                nr = c4.selectbox(t("Rôle"), list(auth.ROLES), format_func=lambda r: t(auth.ROLES[r]))
                if st.form_submit_button(t("Créer"), type="primary"):
                    try:
                        auth.register(nu, nd, ne, npw, npw, role=nr, actor=admin.username)
                        st.session_state.flash = t("Compte @{user} créé", user=nu.strip().lower())
                        st.rerun()
                    except auth.AuthError as e:
                        st.error(t(str(e)))

# ---------------------------------------------------------------------- logs
with t_logs:
    l_app, l_feedback, l_api = st.tabs(
        [t("Journal de l'application"), t("Avis des utilisateurs"), t("Appels à l'API")])

    with l_app:
        c1, c2 = st.columns([1, 2])
        levels = c1.multiselect(t("Niveau"), list(LEVELS), default=list(LEVELS),
                                format_func=lambda x: t(LEVELS[x]))
        search = c2.text_input(t("Rechercher"), placeholder=t("ex. connexion, camille…"))
        ev = [e for e in auth.events() if e["niveau"] in levels and
              (not search or search.lower() in
               (t_actor(e["acteur"]) + " " + t_log(e["message"])).lower())]
        df = pd.DataFrame([{t("Date"): local(e["date"]), t("Niveau"): t(LEVELS.get(e["niveau"])),
                            t("Qui"): t_actor(e["acteur"]), t("Événement"): t_log(e["message"])}
                           for e in ev])
        st.dataframe(df, hide_index=True, width="stretch", height=380)
        st.download_button(t("⬇️ Exporter (CSV)"), df.to_csv(index=False).encode("utf-8-sig"),
                           "journal_application.csv", "text/csv", disabled=df.empty)

    with l_feedback:
        rows = fetch_interactions()
        if rows is None:
            st.error(t("Serveur de recommandations injoignable : impossible de lire les avis."))
        elif not rows:
            st.info(t("Aucun avis enregistré pour l'instant."))
        else:
            df = pd.DataFrame(rows)
            who = st.selectbox(t("Utilisateur"), ["Tous"] + sorted(df["user_id"].unique()),
                               format_func=lambda u: t(u) if u == "Tous" else
                               f"{user_label(u)} ({u})")
            if who != "Tous":
                df = df[df["user_id"] == who]
            counts = df["feedback"].value_counts()
            for col, (fb, (icon, label)) in zip(st.columns(len(FEEDBACK)), FEEDBACK.items()):
                col.metric(f"{icon} {t(label)}", int(counts.get(fb, 0)))
            view = pd.DataFrame({
                t("N°"): df["id"], t("Utilisateur"): df["user_id"].map(user_label),
                t("Recherche"): df["query_ingredients"].map(lambda q: " + ".join(fr(i) for i in q)),
                t("Suggestion"): df["recommended_ingredient"].map(
                    lambda n: fr(n) if isinstance(n, str) else "—"),
                t("Avis"): df["feedback"].map(
                    lambda f: " ".join(t(x) for x in FEEDBACK.get(f, ("", f)))),
                t("Note"): df["note"].fillna(""), t("Date"): df["created_at"].map(local),
            })
            st.dataframe(view, hide_index=True, width="stretch", height=380)
            st.download_button(t("⬇️ Exporter (CSV)"), df.to_csv(index=False).encode("utf-8-sig"),
                               "avis_utilisateurs.csv", "text/csv")

    with l_api:
        calls = list(api_client.CALLS)
        hide_health = st.toggle(t("Masquer les vérifications de santé (/health)"), value=True)
        if hide_health:
            calls = [c for c in calls if c["route"] != "/health"]
        if not calls:
            st.info(t("Aucun appel enregistré depuis le démarrage du serveur Streamlit."))
        else:
            df = pd.DataFrame(calls)
            summary = df.groupby(["méthode", "route"]).agg(
                appels=("ms", "size"), erreurs=("ok", lambda s: int((~s).sum())),
                moyenne_ms=("ms", "mean"), p95_ms=("ms", lambda s: s.quantile(.95)),
                max_ms=("ms", "max")).round(1).reset_index()
            st.markdown(t("**Par route**"))
            summary = summary.rename(columns={c: t(c) for c in summary.columns})
            st.dataframe(summary, hide_index=True, width="stretch")
            st.markdown(t("**Derniers appels**"))
            df["date"] = df["date"].map(local)
            df["ok"] = df["ok"].map({True: "✅", False: "❌"})
            df = df.rename(columns={"méthode": t("méthode")})
            st.dataframe(df, hide_index=True, width="stretch", height=300)

# ------------------------------------------------------- services & config
with t_services:
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader(t("Serveur de recommandations (API)"))
        with st.form("api_url"):
            url = st.text_input(t("Adresse de l'API"), value=api_url(),
                                help=t("Appliquée à tous les utilisateurs de l'application."))
            if st.form_submit_button(t("Enregistrer l'adresse")):
                run(auth.admin_set_api_url, url, success=t("Adresse de l'API mise à jour"))

        if st.button(t("🔌 Tester la connexion"), type="primary"):
            checks = []
            def check_health():
                if not api_client.health(api_url()):
                    raise api_client.APIError(t("injoignable ou statut ≠ ok"))

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
                    checks.append({t("Route"): label, t("Résultat"): "✅ OK",
                                   t("Temps"): f"{(time.perf_counter() - start) * 1000:.0f} ms"})
                except api_client.APIError as e:
                    checks.append({t("Route"): label, t("Résultat"): f"❌ {e}"[:120], t("Temps"): "—"})
            st.dataframe(checks, hide_index=True, width="stretch")
            st.caption(t("Objectif du projet : moins de 2 s par recommandation."))

    with right:
        st.subheader(t("Modèle et données"))
        emb = ROOT / "artifacts" / "ingredient_embeddings.npy"
        if emb.exists():
            trained = datetime.fromtimestamp(emb.stat().st_mtime).strftime("%d/%m/%Y %H:%M")
            st.markdown(t("- Embeddings entraînés : ✅ `artifacts/{file}` ({date})",
                          file=emb.name, date=trained))
        else:
            st.markdown(t("- Embeddings entraînés : ❌ absents (lancer `scripts/train_nlgcl.py`)"))
        cfg = RankingConfig()
        st.markdown(t("- Pondération du classement : habitudes culinaires **{emb}** · chimie des "
                      "arômes **{chem}** · goûts **{pref}**", emb=cfg.alpha_emb, chem=cfg.beta_chem,
                      pref=cfg.gamma_pref))
        try:
            commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                    capture_output=True, text=True, timeout=5).stdout.strip()
            branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT,
                                    capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            commit = branch = ""
        st.markdown(t("- Version du code : `{branch}` @ `{commit}`", branch=branch or "?",
                      commit=commit or "?"))
        st.markdown(f"- Python {platform.python_version()} · Streamlit {st.__version__}")

        st.subheader(t("Comptes"))
        st.info(t("Les comptes et les avis sont stockés dans la base SQLite de l'API "
                "(`data/app.db`, ou `FOODPAIRING_DB`). Les mots de passe y sont salés et hachés. "
                "Le journal de l'application, lui, est gardé en mémoire et repart de zéro au "
                "redémarrage de Streamlit."))
