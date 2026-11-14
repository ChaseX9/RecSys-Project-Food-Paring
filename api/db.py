import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "app.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id                TEXT    NOT NULL,
    query_ingredients      TEXT    NOT NULL,  -- JSON list of the ingredients the user searched for
    recommended_ingredient TEXT,              -- the pairing the feedback is about (NULL for free-text logs)
    feedback               TEXT    NOT NULL CHECK (feedback IN ('click', 'like', 'dislike', 'not_for_me', 'cooked')),
    note                   TEXT,              -- optional free text (what they actually cooked)
    created_at             TEXT    NOT NULL   -- ISO 8601, UTC
);
CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions (user_id, created_at);
"""


def connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def insert_interaction(db_path, user_id, query_ingredients, recommended_ingredient, feedback, note) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO interactions (user_id, query_ingredients, recommended_ingredient, feedback, note, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, json.dumps(query_ingredients), recommended_ingredient, feedback, note,
             datetime.now(timezone.utc).isoformat()),
        )
        return cur.lastrowid


def list_interactions(db_path, user_id: str | None = None, limit: int = 100) -> list[dict]:
    query, params = "SELECT * FROM interactions", []
    if user_id:
        query += " WHERE user_id = ?"
        params.append(user_id)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["query_ingredients"] = json.loads(d["query_ingredients"])
        out.append(d)
    return out
