"""User accounts of the app, stored in the same SQLite database as the interactions.

Passwords are salted and hashed (PBKDF2-SHA256, 100k rounds); hashes never leave this module.
Every rule lives here (validation, unique e-mail, "at least one active admin", admin-only
actions) so any client gets the same behaviour. Error messages are in French: the Streamlit
app shows them as is.

The API has no authentication of its own yet: it trusts its caller (the Streamlit server,
running next to it). Admin actions take the ``actor`` doing them and check that this account is
an admin, which enforces the rules but is not a security boundary.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import sqlite3
from datetime import datetime, timezone

from .db import connect

ROLES = ("user", "admin")
DIETS = (None, "vegetarian", "vegan", "pescatarian")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
USERNAME_RE = re.compile(r"^[a-z0-9_.-]{3,30}$")
MIN_PASSWORD = 8
LIST_FIELDS = ("allergies", "disliked", "fridge")
EDITABLE = {"display_name", "email", "diet", "allergies", "disliked", "top_k", "food_com_id",
            "fridge", "onboarding_done"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username     TEXT PRIMARY KEY,
    display_name TEXT    NOT NULL,
    email        TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    role         TEXT    NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')),
    active       INTEGER NOT NULL DEFAULT 1,
    diet         TEXT,                          -- vegetarian / vegan / pescatarian
    allergies    TEXT    NOT NULL DEFAULT '[]', -- JSON list of allergy keys (gluten, nut, ...)
    disliked     TEXT    NOT NULL DEFAULT '[]', -- JSON list of ingredients never to suggest
    fridge       TEXT    NOT NULL DEFAULT '[]', -- JSON list of ingredients the user has at home
    top_k        INTEGER NOT NULL DEFAULT 10,
    food_com_id  TEXT,                          -- linked Food.com profile (rating history)
    onboarding_done INTEGER NOT NULL DEFAULT 0, -- recipe-rating onboarding done or skipped
    created_at   TEXT    NOT NULL,
    last_login   TEXT,
    salt         TEXT    NOT NULL,
    pw_hash      TEXT    NOT NULL
);
"""

# Created on the first start (empty table), shown on the login page. Demo only.
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


class UserError(ValueError):
    """A rule was broken. ``status`` is the HTTP status the API answers with."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 100_000).hex()


def _new_password(password: str) -> tuple[str, str]:
    salt = secrets.token_hex(16)
    return salt, _hash(password, salt)


def _public(row: sqlite3.Row) -> dict:
    d = dict(row)
    d.pop("salt"), d.pop("pw_hash")
    d["active"] = bool(d["active"])
    d["onboarding_done"] = bool(d["onboarding_done"])
    for k in LIST_FIELDS:
        d[k] = json.loads(d[k])
    return d


def init_db(db_path, seed_demo: bool = True) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(users)")}
        if "onboarding_done" not in columns:  # databases created before the recipe ratings
            conn.execute("ALTER TABLE users ADD COLUMN onboarding_done INTEGER NOT NULL DEFAULT 0")
        if seed_demo and conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
            for username, password, extra in DEMO_ACCOUNTS:
                # demo profiles come pre-filled: no recipe-rating onboarding on their first login
                _insert(conn, username, password, onboarding_done=True, **extra)


# ------------------------------------------------------------------ validation
def _validate_username(username: str) -> str:
    username = username.strip().lower()
    if not USERNAME_RE.match(username):
        raise UserError(400, "Identifiant : 3 à 30 caractères, lettres minuscules, chiffres, . _ -")
    return username


def _validate_email(email: str) -> str:
    email = email.strip()
    if not EMAIL_RE.match(email):
        raise UserError(400, "Adresse e-mail invalide.")
    return email


def _validate_password(password: str) -> None:
    if len(password) < MIN_PASSWORD:
        raise UserError(400, f"Le mot de passe doit faire au moins {MIN_PASSWORD} caractères.")


def _validate_changes(changes: dict) -> dict:
    unknown = set(changes) - EDITABLE
    if unknown:
        raise UserError(400, f"Champs non modifiables : {', '.join(sorted(unknown))}")
    out = dict(changes)
    if "email" in out:
        out["email"] = _validate_email(out["email"])
    if "display_name" in out:
        out["display_name"] = out["display_name"].strip()
        if not out["display_name"]:
            raise UserError(400, "Le nom ne peut pas être vide.")
    if "food_com_id" in out:
        fid = str(out["food_com_id"]).strip() if out["food_com_id"] is not None else ""
        if fid and not fid.isdigit():
            raise UserError(400, "L'identifiant Food.com doit être un nombre.")
        out["food_com_id"] = fid or None
    if "diet" in out and out["diet"] not in DIETS:
        raise UserError(400, "Régime inconnu.")
    if "onboarding_done" in out:
        out["onboarding_done"] = int(bool(out["onboarding_done"]))
    if "top_k" in out and not 1 <= int(out["top_k"]) <= 50:
        raise UserError(400, "Nombre de suggestions : entre 1 et 50.")
    for k in LIST_FIELDS:
        if k in out:
            out[k] = list(dict.fromkeys(str(x).strip() for x in out[k] or [] if str(x).strip()))
    return out


def _email_taken(conn, email: str, except_user: str | None = None) -> bool:
    row = conn.execute("SELECT username FROM users WHERE email = ? COLLATE NOCASE", (email,)).fetchone()
    return row is not None and row["username"] != except_user


def _insert(conn, username: str, password: str, display_name: str, email: str, role: str = "user",
            **profile) -> None:
    salt, pw_hash = _new_password(password)
    fields = {"diet": None, "top_k": 10, "food_com_id": None, "onboarding_done": False, **profile}
    conn.execute(
        "INSERT INTO users (username, display_name, email, role, diet, allergies, disliked, fridge, "
        "top_k, food_com_id, onboarding_done, created_at, salt, pw_hash) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (username, display_name, email, role, fields["diet"],
         *(json.dumps(fields.get(k, [])) for k in LIST_FIELDS),
         fields["top_k"], fields["food_com_id"], int(fields["onboarding_done"]), _now(), salt,
         pw_hash),
    )


def _row(conn, username: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if row is None:
        raise UserError(404, "Compte introuvable.")
    return row


def _active_admins(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM users WHERE role = 'admin' AND active = 1").fetchone()[0]


def _require_admin(conn, actor: str | None) -> None:
    row = conn.execute("SELECT role, active FROM users WHERE username = ?", (actor or "",)).fetchone()
    if row is None or row["role"] != "admin" or not row["active"]:
        raise UserError(403, "Action réservée aux administrateurs.")


# ------------------------------------------------------------------ accounts
def create_user(db_path, username: str, display_name: str, email: str, password: str,
                role: str = "user", actor: str | None = None) -> dict:
    """Self sign-up (role "user"), or an admin (``actor``) creating any account."""
    username = _validate_username(username)
    display_name = display_name.strip()
    if not display_name:
        raise UserError(400, "Indique un nom.")
    email = _validate_email(email)
    _validate_password(password)
    if role not in ROLES:
        raise UserError(400, "Rôle inconnu.")
    with connect(db_path) as conn:
        if role != "user":
            _require_admin(conn, actor)
        if conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
            raise UserError(409, "Cet identifiant est déjà pris.")
        if _email_taken(conn, email):
            raise UserError(409, "Cette adresse e-mail est déjà utilisée.")
        _insert(conn, username, password, display_name, email, role)
        return _public(_row(conn, username))


def authenticate(db_path, username: str, password: str) -> dict:
    username = username.strip().lower()
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if row is None or not hmac.compare_digest(row["pw_hash"], _hash(password, row["salt"])):
            raise UserError(401, "Identifiant ou mot de passe incorrect.")
        if not row["active"]:
            raise UserError(403, "Ce compte est désactivé. Contacte un administrateur.")
        conn.execute("UPDATE users SET last_login = ? WHERE username = ?", (_now(), username))
        return _public(_row(conn, username))


def get_user(db_path, username: str) -> dict:
    with connect(db_path) as conn:
        return _public(_row(conn, username))


def list_users(db_path) -> list[dict]:
    with connect(db_path) as conn:
        return [_public(r) for r in conn.execute("SELECT * FROM users ORDER BY created_at, username")]


def update_user(db_path, username: str, changes: dict) -> dict:
    changes = _validate_changes(changes)
    with connect(db_path) as conn:
        _row(conn, username)
        if "email" in changes and _email_taken(conn, changes["email"], except_user=username):
            raise UserError(409, "Cette adresse e-mail est déjà utilisée.")
        if changes:
            values = [json.dumps(v) if k in LIST_FIELDS else v for k, v in changes.items()]
            conn.execute(f"UPDATE users SET {', '.join(f'{k} = ?' for k in changes)} WHERE username = ?",
                         (*values, username))
        return _public(_row(conn, username))


def change_password(db_path, username: str, current: str, new: str) -> None:
    with connect(db_path) as conn:
        row = _row(conn, username)
        if not hmac.compare_digest(row["pw_hash"], _hash(current, row["salt"])):
            raise UserError(401, "Mot de passe actuel incorrect.")
        _validate_password(new)
        conn.execute("UPDATE users SET salt = ?, pw_hash = ? WHERE username = ?",
                     (*_new_password(new), username))


# ------------------------------------------------------------------- admin
def set_role(db_path, actor: str, username: str, role: str) -> dict:
    if role not in ROLES:
        raise UserError(400, "Rôle inconnu.")
    with connect(db_path) as conn:
        _require_admin(conn, actor)
        row = _row(conn, username)
        if row["role"] == "admin" and role != "admin" and row["active"] and _active_admins(conn) <= 1:
            raise UserError(409, "Impossible : il doit rester au moins un administrateur actif.")
        conn.execute("UPDATE users SET role = ? WHERE username = ?", (role, username))
        return _public(_row(conn, username))


def set_active(db_path, actor: str, username: str, active: bool) -> dict:
    with connect(db_path) as conn:
        _require_admin(conn, actor)
        if username == actor and not active:
            raise UserError(409, "Tu ne peux pas désactiver ton propre compte.")
        row = _row(conn, username)
        if row["role"] == "admin" and row["active"] and not active and _active_admins(conn) <= 1:
            raise UserError(409, "Impossible : il doit rester au moins un administrateur actif.")
        conn.execute("UPDATE users SET active = ? WHERE username = ?", (int(active), username))
        return _public(_row(conn, username))


def reset_password(db_path, actor: str, username: str) -> str:
    """Set a random temporary password and return it (to hand over to the user)."""
    temp = secrets.token_urlsafe(9)
    with connect(db_path) as conn:
        _require_admin(conn, actor)
        _row(conn, username)
        conn.execute("UPDATE users SET salt = ?, pw_hash = ? WHERE username = ?",
                     (*_new_password(temp), username))
    return temp


def delete_user(db_path, actor: str, username: str) -> None:
    """Deletes the account, its interactions and its recipe ratings (a new account reusing the
    username must not inherit the old one's likes, dislikes, allergies and tastes)."""
    with connect(db_path) as conn:
        _require_admin(conn, actor)
        if username == actor:
            raise UserError(409, "Tu ne peux pas supprimer ton propre compte.")
        row = _row(conn, username)
        if row["role"] == "admin" and row["active"] and _active_admins(conn) <= 1:
            raise UserError(409, "Impossible : il doit rester au moins un administrateur actif.")
        conn.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.execute("DELETE FROM interactions WHERE user_id = ?", (username,))
        conn.execute("DELETE FROM recipe_ratings WHERE user_id = ?", (username,))
