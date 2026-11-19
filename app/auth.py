"""User accounts (stored by the API, see api/users.py) + the app's event log and settings.

Accounts live in the API's SQLite database: they survive restarts, and every rule (validation,
unique e-mail, "at least one active admin", password hashing) is enforced there. This module
only keeps the session (who is logged in, in st.session_state) and turns API errors into
AuthError with the API's French message.

The event log and the app settings (API address) stay in the Streamlit server process
(st.cache_resource): shared by all tabs, reset when Streamlit restarts.
"""
from __future__ import annotations

import re
import secrets
import threading
from collections import deque
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone

import streamlit as st

from api.users import DEMO_ACCOUNTS, MIN_PASSWORD  # noqa: F401  (shown on the login page)
from app import api_client

ROLES = {"user": "Utilisateur", "admin": "Administrateur"}
GUEST = "anonymous"  # generic id for API checks; each guest session gets its own guest-xxxx id
GUEST_PREFIX = "guest-"
EDITABLE = {"display_name", "email", "diet", "allergies", "disliked", "top_k", "food_com_id",
            "fridge", "onboarding_done"}
API_DOWN = "Le serveur ne répond pas : réessaie dans un instant (ou lance `uvicorn api.main:app`)."


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class User:
    username: str
    display_name: str
    email: str
    role: str = "user"
    active: bool = True
    diet: str | None = None                 # vegan / vegetarian / pescatarian
    allergies: list[str] = field(default_factory=list)
    disliked: list[str] = field(default_factory=list)   # ingredients never to suggest
    fridge: list[str] = field(default_factory=list)     # ingredients the user has at home
    top_k: int = 10
    food_com_id: str | None = None          # linked Food.com profile, used for personalization
    onboarding_done: bool = False           # recipe-rating onboarding done or skipped
    created_at: str = ""
    last_login: str | None = None

    @classmethod
    def from_api(cls, d: dict) -> "User":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def initials(self) -> str:
        parts = self.display_name.split() or [self.username]
        return "".join(p[0] for p in parts[:2]).upper()


class AuthError(ValueError):
    """User-facing error (message is in French and safe to display)."""


class Store:
    """App-wide, in-memory: event log + settings. Accounts are NOT here (they are in the API)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.events: deque[dict] = deque(maxlen=1000)
        self.settings = {"api_url": api_client.DEFAULT_API_URL}  # editable by admins
        self.log("info", "système", "Application démarrée")

    def log(self, level: str, actor: str, message: str) -> None:
        with self._lock:
            self.events.appendleft({"date": _now(), "niveau": level, "acteur": actor,
                                    "message": message})


@st.cache_resource
def store() -> Store:
    return Store()


def _url() -> str:
    return store().settings["api_url"]


def _call(fn, *args, **kwargs):
    """Run an api_client call; rule violations and outages become AuthError."""
    try:
        return fn(*args, base_url=_url(), **kwargs)
    except api_client.APIError as e:
        if e.status is not None and e.status < 500 and e.detail:
            raise AuthError(e.detail) from e
        raise AuthError(API_DOWN) from e


# ------------------------------------------------------------------ session
def current_user() -> User | None:
    """The logged-in user, read from the API (so changes made elsewhere, e.g. by an admin,
    show up). If the API is briefly unreachable, the last known copy is used."""
    name = st.session_state.get("auth_user")
    if not name:
        return None
    try:
        d = api_client.get_user(name, base_url=_url())
    except api_client.APIError as e:
        if e.status == 404:          # deleted meanwhile
            d = None
        else:
            d = st.session_state.get("auth_user_cache")
    if d is None or not d.get("active", True):  # deleted or deactivated meanwhile
        st.session_state.pop("auth_user", None)
        st.session_state.pop("auth_user_cache", None)
        return None
    st.session_state.auth_user_cache = d
    return User.from_api(d)


def is_guest() -> bool:
    return st.session_state.get("auth_guest", False) and current_user() is None


def is_authenticated() -> bool:
    return current_user() is not None or is_guest()


def login(username: str, password: str) -> User:
    username = username.strip().lower()
    try:
        d = _call(api_client.login, username, password)
    except AuthError as e:
        store().log("warning", username or "?", f"Échec de connexion ({e})")
        raise
    store().log("info", username, "Connexion")
    st.session_state.auth_user = username
    st.session_state.auth_user_cache = d
    st.session_state.pop("auth_guest", None)
    return User.from_api(d)


def login_as_guest() -> None:
    st.session_state.pop("auth_user", None)
    st.session_state.auth_guest = True
    store().log("info", "invité", "Connexion en invité")


def logout() -> None:
    who = st.session_state.get("auth_user") or "invité"
    store().log("info", who, "Déconnexion")
    # Drop everything tied to the previous user (results, feedback state, ...), keep the theme.
    keep = {k: st.session_state[k] for k in ("dark_mode",) if k in st.session_state}
    st.session_state.clear()
    st.session_state.update(keep)


def log_user_id(u: User | None) -> str:
    """Id sent to the API as ``user_id`` (feedback, history, /recommend): always the account, so
    each user's feedback stays theirs. Guests get one id per browser session, so one guest's
    dislikes are never applied to another guest."""
    if u is not None:
        return u.username
    if "guest_id" not in st.session_state:
        st.session_state.guest_id = GUEST_PREFIX + secrets.token_hex(4)
    return st.session_state.guest_id


def history_user_id(u: User | None) -> str | None:
    """Id sent as ``history_user_id``: the linked Food.com profile whose ratings personalize
    the ranking, or None (the account itself, which has no rating history)."""
    return u.food_com_id if u else None


# ---------------------------------------------------------------- accounts
def register(username: str, display_name: str, email: str, password: str, confirm: str,
             role: str = "user", actor: str | None = None) -> User:
    if password != confirm:  # the API only gets one password: the check belongs to the form
        raise AuthError("Les deux mots de passe ne correspondent pas.")
    d = _call(api_client.create_user, username.strip().lower(), display_name, email, password,
              role=role, actor=actor)
    store().log("info", actor or d["username"], f"Compte créé : {d['username']} ({ROLES[role]})")
    return User.from_api(d)


def update_profile(username: str, **changes) -> User:
    unknown = set(changes) - EDITABLE
    if unknown:
        raise ValueError(f"Champs non modifiables : {unknown}")
    d = _call(api_client.update_user, username, changes)
    store().log("info", username, f"Profil mis à jour ({', '.join(sorted(changes))})")
    return User.from_api(d)


def change_password(username: str, current: str, new: str, confirm: str) -> None:
    if new != confirm:
        raise AuthError("Les deux mots de passe ne correspondent pas.")
    try:
        _call(api_client.change_password, username, current, new)
    except AuthError as e:
        store().log("warning", username, f"Changement de mot de passe refusé ({e})")
        raise
    store().log("info", username, "Mot de passe changé")


# ------------------------------------------------------------------- admin
def admin_set_role(actor: str, username: str, role: str) -> None:
    _call(api_client.admin_action, "role", actor, username, role=role)
    store().log("info", actor, f"Rôle de {username} → {ROLES[role]}")


def admin_set_active(actor: str, username: str, active: bool) -> None:
    _call(api_client.admin_action, "active", actor, username, active=active)
    store().log("warning" if not active else "info", actor,
                f"Compte {username} {'réactivé' if active else 'désactivé'}")


def admin_reset_password(actor: str, username: str) -> str:
    """Set a random temporary password and return it (to hand over to the user)."""
    temp = _call(api_client.admin_action, "reset_password", actor, username)["temporary_password"]
    store().log("warning", actor, f"Mot de passe de {username} réinitialisé")
    return temp


def admin_delete(actor: str, username: str) -> None:
    _call(api_client.admin_delete_user, actor, username)
    store().log("warning", actor, f"Compte {username} supprimé (avec ses avis)")


def admin_set_api_url(actor: str, url: str) -> None:
    # The logged-in admin's own session is enough (it still works when the current address is
    # wrong, which is exactly when an admin needs to change it).
    u = current_user() if st.session_state.get("auth_user") == actor else \
        User.from_api(_call(api_client.get_user, actor))
    if u is None or u.username != actor or not u.is_admin:
        raise AuthError("Action réservée aux administrateurs.")
    url = url.strip().rstrip("/")
    if not re.match(r"^https?://[^\s/]+", url):
        raise AuthError("Adresse invalide (ex. http://127.0.0.1:8000).")
    store().settings["api_url"] = url
    store().log("warning", actor, f"Adresse de l'API changée : {url}")


def all_users() -> list[dict]:
    return _call(api_client.list_users)


def events() -> list[dict]:
    return list(store().events)
