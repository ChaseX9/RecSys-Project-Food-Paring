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
    feedback               TEXT    NOT NULL CHECK (feedback IN ('click', 'like', 'dislike', 'not_for_me', 'cooked', 'allergic')),
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
        _migrate_feedback_check(conn)


def _migrate_feedback_check(conn: sqlite3.Connection) -> None:
    """Databases created before 'allergic' existed have a CHECK that rejects it: rebuild the
    table with the current schema and copy every row (ids and timestamps are kept)."""
    row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='interactions'").fetchone()
    if row is None or "'allergic'" in row["sql"]:
        return
    conn.executescript("""
        ALTER TABLE interactions RENAME TO interactions_old;
        DROP INDEX IF EXISTS idx_interactions_user;
    """ + SCHEMA + """
        INSERT INTO interactions (id, user_id, query_ingredients, recommended_ingredient, feedback, note, created_at)
            SELECT id, user_id, query_ingredients, recommended_ingredient, feedback, note, created_at
            FROM interactions_old;
        DROP TABLE interactions_old;
    """)


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


POSITIVE_FEEDBACK = {"like", "cooked"}
NEGATIVE_FEEDBACK = {"dislike", "not_for_me"}


def _query_key(ingredients) -> tuple[str, ...]:
    return tuple(sorted({str(i).strip().lower() for i in ingredients}))


def feedback_for_user(db_path, user_id: str, query_ingredients=None) -> dict[str, list[str]]:
    """Explicit feedback of ``user_id`` on ingredients, for the feedback loop.

    - like / cooked -> liked (everywhere)
    - dislike       -> disliked (everywhere: the user does not want this ingredient)
    - not_for_me    -> disliked only for the same query ("not with this dish")
    - allergic      -> allergic (everywhere, permanent: a later like does NOT lift it;
                       the ranker also bans every ingredient containing the word)
    - click         -> ignored (implicit, too weak)
    Otherwise the most recent feedback on an ingredient wins, so a user can change their mind.
    """
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT query_ingredients, recommended_ingredient, feedback FROM interactions "
            "WHERE user_id = ? AND recommended_ingredient IS NOT NULL ORDER BY id ASC",
            (user_id,),
        ).fetchall()
    key = _query_key(query_ingredients) if query_ingredients else None
    state: dict[str, str] = {}
    allergic: set[str] = set()
    for r in rows:
        name = r["recommended_ingredient"].strip().lower()
        fb = r["feedback"]
        if fb == "allergic":
            allergic.add(name)
        elif fb in POSITIVE_FEEDBACK or fb == "dislike":
            state[name] = "liked" if fb in POSITIVE_FEEDBACK else "disliked"
        elif fb == "not_for_me" and key is not None and _query_key(json.loads(r["query_ingredients"])) == key:
            state[name] = "disliked"
    return {"liked": sorted(n for n, s in state.items() if s == "liked" and n not in allergic),
            "disliked": sorted(n for n, s in state.items() if s == "disliked" and n not in allergic),
            "allergic": sorted(allergic)}
