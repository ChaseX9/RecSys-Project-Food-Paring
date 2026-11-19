"""Mock user accounts + app event log, kept in memory until the real user database exists.

Everything here lives in the Streamlit server process (st.cache_resource): it is shared by all
browser tabs and survives page reloads, but is reset when the server restarts. The public
functions (login, update_profile, ...) are the interface the future DB-backed version must keep.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import threading
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import streamlit as st

from app import api_client

ROLES = {"user": "Utilisateur", "admin": "Administrateur"}
GUEST = "anonymous"  # user_id sent to the API for guests
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
USERNAME_RE = re.compile(r"^[a-z0-9_.-]{3,30}$")
MIN_PASSWORD = 8


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 100_000).hex()


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
    created_at: str = field(default_factory=_now)
    last_login: str | None = None
    salt: str = ""
    pw_hash: str = ""

    def set_password(self, password: str) -> None:
        self.salt = secrets.token_hex(16)
        self.pw_hash = _hash(password, self.salt)

    def check_password(self, password: str) -> bool:
        return bool(self.pw_hash) and hmac.compare_digest(self.pw_hash, _hash(password, self.salt))

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def initials(self) -> str:
        parts = self.display_name.split() or [self.username]
        return "".join(p[0] for p in parts[:2]).upper()

    def public(self) -> dict:
        d = asdict(self)
        d.pop("salt"), d.pop("pw_hash")
        return d


# Demo accounts, shown on the login page. Passwords are only meaningful for this mock store.
DEMO_ACCOUNTS = [
    ("admin", "admin1234", dict(display_name="Admin Food Pairing", email="admin@foodpairing.local",
                                role="admin")),
    ("demo", "demo1234", dict(display_name="Alex Démo", email="alex@foodpairing.local",
                              food_com_id="424680",
                              fridge=["salmon", "lemon", "egg", "butter", "garlic", "spinach",
                                      "cream cheese", "potato", "dill", "onion", "rice"])),
    ("camille", "camille1234", dict(display_name="Camille Végé", email="camille@foodpairing.local",
                                    diet="vegetarian", allergies=["nut"], disliked=["cilantro"],
                                    fridge=["tomato", "basil", "mozzarella cheese", "zucchini",
                                            "egg", "garlic", "olive oil", "chickpea", "lemon"])),
]


class AuthError(ValueError):
    """User-facing error (message is in French and safe to display)."""


class Store:
    def __init__(self):
        self._lock = threading.Lock()
        self.users: dict[str, User] = {}
        self.events: deque[dict] = deque(maxlen=1000)
        self.settings = {"api_url": api_client.DEFAULT_API_URL}  # app-wide, editable by admins
        for username, pw, extra in DEMO_ACCOUNTS:
            u = User(username=username, **extra)
            u.set_password(pw)
            self.users[username] = u
        self.log("info", "système", "Comptes de démonstration créés")

    def log(self, level: str, actor: str, message: str) -> None:
        self.events.appendleft({"date": _now(), "niveau": level, "acteur": actor, "message": message})


@st.cache_resource
def store() -> Store:
    return Store()


# ------------------------------------------------------------------ session
def current_user() -> User | None:
    """The logged-in user (a fresh copy, so pages can't mutate the store by accident)."""
    name = st.session_state.get("auth_user")
    u = store().users.get(name) if name else None
    if name and (u is None or not u.active):  # deleted or deactivated meanwhile
        st.session_state.pop("auth_user", None)
        return None
    return deepcopy(u) if u else None


def is_guest() -> bool:
    return st.session_state.get("auth_guest", False) and current_user() is None


def is_authenticated() -> bool:
    return current_user() is not None or is_guest()


def login(username: str, password: str) -> User:
    s = store()
    username = username.strip().lower()
    with s._lock:
        u = s.users.get(username)
        if u is None or not u.check_password(password):
            s.log("warning", username or "?", "Échec de connexion")
            raise AuthError("Identifiant ou mot de passe incorrect.")
        if not u.active:
            s.log("warning", username, "Connexion refusée : compte désactivé")
            raise AuthError("Ce compte est désactivé. Contacte un administrateur.")
        u.last_login = _now()
        s.log("info", username, "Connexion")
    st.session_state.auth_user = username
    st.session_state.pop("auth_guest", None)
    return deepcopy(u)


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


def api_user_id(u: User | None) -> str:
    """Id sent to /recommend: the linked Food.com profile if any (personalization), else the account."""
    if u is None:
        return GUEST
    return u.food_com_id or u.username


def log_user_id(u: User | None) -> str:
    """Id sent to /log_interaction: always the account, so each user's history stays theirs."""
    return u.username if u else GUEST


# ---------------------------------------------------------------- accounts
def _validate_email(email: str) -> str:
    email = email.strip()
    if not EMAIL_RE.match(email):
        raise AuthError("Adresse e-mail invalide.")
    return email


def _validate_password(password: str, confirm: str) -> None:
    if len(password) < MIN_PASSWORD:
        raise AuthError(f"Le mot de passe doit faire au moins {MIN_PASSWORD} caractères.")
    if password != confirm:
        raise AuthError("Les deux mots de passe ne correspondent pas.")


def register(username: str, display_name: str, email: str, password: str, confirm: str,
             role: str = "user", actor: str | None = None) -> User:
    s = store()
    username = username.strip().lower()
    if not USERNAME_RE.match(username):
        raise AuthError("Identifiant : 3 à 30 caractères, lettres minuscules, chiffres, . _ -")
    if not display_name.strip():
        raise AuthError("Indique un nom.")
    email = _validate_email(email)
    _validate_password(password, confirm)
    with s._lock:
        if username in s.users:
            raise AuthError("Cet identifiant est déjà pris.")
        if any(u.email.lower() == email.lower() for u in s.users.values()):
            raise AuthError("Cette adresse e-mail est déjà utilisée.")
        u = User(username=username, display_name=display_name.strip(), email=email, role=role)
        u.set_password(password)
        s.users[username] = u
        s.log("info", actor or username, f"Compte créé : {username} ({ROLES[role]})")
    return deepcopy(u)


def update_profile(username: str, **changes) -> User:
    allowed = {"display_name", "email", "diet", "allergies", "disliked", "top_k", "food_com_id",
               "fridge"}
    unknown = set(changes) - allowed
    if unknown:
        raise ValueError(f"Champs non modifiables : {unknown}")
    s = store()
    if "email" in changes:
        changes["email"] = _validate_email(changes["email"])
    if "display_name" in changes and not changes["display_name"].strip():
        raise AuthError("Le nom ne peut pas être vide.")
    if changes.get("food_com_id") and not str(changes["food_com_id"]).isdigit():
        raise AuthError("L'identifiant Food.com doit être un nombre.")
    with s._lock:
        if "email" in changes and any(o.email.lower() == changes["email"].lower()
                                      for n, o in s.users.items() if n != username):
            raise AuthError("Cette adresse e-mail est déjà utilisée.")
        u = s.users[username]
        for k, v in changes.items():
            setattr(u, k, v)
        s.log("info", username, f"Profil mis à jour ({', '.join(sorted(changes))})")
    return deepcopy(u)


def change_password(username: str, current: str, new: str, confirm: str) -> None:
    s = store()
    with s._lock:
        u = s.users[username]
        if not u.check_password(current):
            s.log("warning", username, "Changement de mot de passe refusé (mot de passe actuel faux)")
            raise AuthError("Mot de passe actuel incorrect.")
        _validate_password(new, confirm)
        u.set_password(new)
        s.log("info", username, "Mot de passe changé")


# ------------------------------------------------------------------- admin
def _require_admin(actor: str) -> None:
    a = store().users.get(actor)
    if a is None or not a.is_admin:
        raise AuthError("Action réservée aux administrateurs.")


def _admin_count(s: Store) -> int:
    return sum(1 for u in s.users.values() if u.is_admin and u.active)


def admin_set_role(actor: str, username: str, role: str) -> None:
    _require_admin(actor)
    s = store()
    with s._lock:
        u = s.users[username]
        if u.is_admin and role != "admin" and _admin_count(s) <= 1:
            raise AuthError("Impossible : il doit rester au moins un administrateur actif.")
        u.role = role
        s.log("info", actor, f"Rôle de {username} → {ROLES[role]}")


def admin_set_active(actor: str, username: str, active: bool) -> None:
    _require_admin(actor)
    if username == actor and not active:
        raise AuthError("Tu ne peux pas désactiver ton propre compte.")
    s = store()
    with s._lock:
        u = s.users[username]
        if u.is_admin and not active and _admin_count(s) <= 1:
            raise AuthError("Impossible : il doit rester au moins un administrateur actif.")
        u.active = active
        s.log("warning" if not active else "info", actor,
              f"Compte {username} {'réactivé' if active else 'désactivé'}")


def admin_reset_password(actor: str, username: str) -> str:
    """Set a random temporary password and return it (to hand over to the user)."""
    _require_admin(actor)
    temp = secrets.token_urlsafe(9)
    s = store()
    with s._lock:
        s.users[username].set_password(temp)
        s.log("warning", actor, f"Mot de passe de {username} réinitialisé")
    return temp


def admin_delete(actor: str, username: str) -> None:
    _require_admin(actor)
    if username == actor:
        raise AuthError("Tu ne peux pas supprimer ton propre compte.")
    s = store()
    with s._lock:
        u = s.users[username]
        if u.is_admin and _admin_count(s) <= 1:
            raise AuthError("Impossible : il doit rester au moins un administrateur actif.")
        del s.users[username]
        s.log("warning", actor, f"Compte {username} supprimé")


def admin_set_api_url(actor: str, url: str) -> None:
    _require_admin(actor)
    url = url.strip().rstrip("/")
    if not re.match(r"^https?://[^\s/]+", url):
        raise AuthError("Adresse invalide (ex. http://127.0.0.1:8000).")
    store().settings["api_url"] = url
    store().log("warning", actor, f"Adresse de l'API changée : {url}")


def all_users() -> list[dict]:
    return [u.public() for u in store().users.values()]


def events() -> list[dict]:
    return list(store().events)
