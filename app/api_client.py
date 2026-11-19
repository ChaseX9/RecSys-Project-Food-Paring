"""Thin HTTP client for the Food Pairing API. The Streamlit app only talks to the model through here."""
from __future__ import annotations

import os
import time
from collections import deque
from datetime import datetime, timezone

import httpx

# 127.0.0.1 rather than "localhost": on Windows "localhost" tries IPv6 (::1) first, and since
# uvicorn listens on IPv4 only, every call would wait ~2 s for the fallback.
DEFAULT_API_URL = os.environ.get("FOODPAIRING_API_URL", "http://127.0.0.1:8000")
TIMEOUT = 10.0

_client = httpx.Client(timeout=TIMEOUT)  # one pooled client: no new TCP connection per call

# Last API calls made by this app process (shown in the admin panel).
CALLS: deque[dict] = deque(maxlen=500)


class APIError(RuntimeError):
    """``status`` is the HTTP status (None if the API could not be reached), ``detail`` the
    API's own message when it sent one (e.g. a French validation message for accounts)."""

    def __init__(self, message: str, status: int | None = None, detail: str | None = None):
        super().__init__(message)
        self.status = status
        self.detail = detail


def _record(method: str, path: str, status: int | None, start: float, error: str | None) -> None:
    CALLS.appendleft({"date": datetime.now(timezone.utc).isoformat(), "méthode": method,
                      "route": path, "statut": status, "ms": round((time.perf_counter() - start) * 1000, 1),
                      "ok": error is None, "erreur": error})


def _request(method: str, base_url: str, path: str, **kwargs) -> httpx.Response:
    start = time.perf_counter()
    try:
        r = _client.request(method, base_url.rstrip("/") + path, **kwargs)
    except httpx.HTTPError as e:
        _record(method, path, None, start, f"injoignable : {type(e).__name__}")
        raise APIError(f"API injoignable ({base_url}) : {e}") from e
    if r.status_code >= 400:
        _record(method, path, r.status_code, start, r.text[:200])
        try:
            detail = r.json().get("detail")
        except ValueError:
            detail = None
        raise APIError(f"{method} {path} -> HTTP {r.status_code} : {r.text[:300]}",
                       status=r.status_code, detail=detail if isinstance(detail, str) else None)
    _record(method, path, r.status_code, start, None)
    return r


def ping(base_url: str = DEFAULT_API_URL) -> tuple[bool, float]:
    """(reachable, latency in ms) of GET /health."""
    start = time.perf_counter()
    ok = health(base_url)
    return ok, (time.perf_counter() - start) * 1000


def health(base_url: str = DEFAULT_API_URL) -> bool:
    try:
        return _request("GET", base_url, "/health").json().get("status") == "ok"
    except APIError:
        return False


def recommend(
    ingredients: list[str],
    user_id: str,
    top_k: int = 10,
    allergies: list[str] | None = None,
    diet: str | None = None,
    base_url: str = DEFAULT_API_URL,
    candidates: list[str] | None = None,
    exclude: list[str] | None = None,
    history_user_id: str | None = None,
) -> tuple[dict, float]:
    """POST /recommend. Returns (response body, latency in seconds).

    The API applies `user_id`'s past feedback by itself (feedback loop).
    `candidates` restricts the suggestions to these ingredients (e.g. the user's fridge),
    `exclude` removes some, `history_user_id` is the Food.com profile used for personalization."""
    payload = {"user_id": user_id, "ingredients": ingredients, "top_k": top_k,
               "allergies": allergies or None, "diet": diet}
    if candidates is not None:
        payload["candidates"] = candidates
    if exclude:
        payload["exclude"] = exclude
    if history_user_id:
        payload["history_user_id"] = history_user_id
    start = time.perf_counter()
    body = _request("POST", base_url, "/recommend", json=payload).json()
    return body, time.perf_counter() - start


def log_interaction(
    user_id: str,
    query_ingredients: list[str],
    feedback: str,
    recommended_ingredient: str | None = None,
    note: str | None = None,
    base_url: str = DEFAULT_API_URL,
) -> int:
    """POST /log_interaction. Returns the id of the new row."""
    payload = {"user_id": user_id, "query_ingredients": query_ingredients,
               "recommended_ingredient": recommended_ingredient, "feedback": feedback,
               "note": note or None}
    return _request("POST", base_url, "/log_interaction", json=payload).json()["id"]


def interactions(user_id: str | None = None, limit: int = 500,
                 base_url: str = DEFAULT_API_URL) -> list[dict]:
    params = {"limit": limit}
    if user_id:
        params["user_id"] = user_id
    return _request("GET", base_url, "/interactions", params=params).json()


def feedback(user_id: str, base_url: str = DEFAULT_API_URL) -> dict:
    """GET /feedback/{user_id}: what the feedback loop applies ({liked, disliked, allergic})."""
    return _request("GET", base_url, f"/feedback/{user_id}").json()


def profile(user_id: str, history_user_id: str | None = None, n: int = 10,
            base_url: str = DEFAULT_API_URL) -> dict:
    """GET /profile/{user_id}: taste profile as the model sees it (ratings + feedback)."""
    params = {"n": n}
    if history_user_id:
        params["history_user_id"] = history_user_id
    return _request("GET", base_url, f"/profile/{user_id}", params=params).json()



# ------------------------------------------------------------------ accounts
def create_user(username: str, display_name: str, email: str, password: str, role: str = "user",
                actor: str | None = None, base_url: str = DEFAULT_API_URL) -> dict:
    payload = {"username": username, "display_name": display_name, "email": email,
               "password": password, "role": role, "actor": actor}
    return _request("POST", base_url, "/users", json=payload).json()


def login(username: str, password: str, base_url: str = DEFAULT_API_URL) -> dict:
    return _request("POST", base_url, "/login",
                    json={"username": username, "password": password}).json()


def get_user(username: str, base_url: str = DEFAULT_API_URL) -> dict:
    return _request("GET", base_url, f"/users/{username}").json()


def list_users(base_url: str = DEFAULT_API_URL) -> list[dict]:
    return _request("GET", base_url, "/users").json()


def update_user(username: str, changes: dict, base_url: str = DEFAULT_API_URL) -> dict:
    return _request("PATCH", base_url, f"/users/{username}", json=changes).json()


def change_password(username: str, current: str, new: str,
                    base_url: str = DEFAULT_API_URL) -> None:
    _request("POST", base_url, f"/users/{username}/password",
             json={"current": current, "new": new})


def admin_action(action: str, actor: str, username: str, base_url: str = DEFAULT_API_URL,
                 **fields) -> dict:
    """action: role (role=...), active (active=...) or reset_password."""
    return _request("POST", base_url, f"/admin/users/{username}/{action}",
                    json={"actor": actor, **fields}).json()


def admin_delete_user(actor: str, username: str, base_url: str = DEFAULT_API_URL) -> None:
    _request("DELETE", base_url, f"/admin/users/{username}", params={"actor": actor})



# ----------------------------------------------------------- recipe ratings
def _filters(user_id: str | None, diet: str | None, allergies: list[str] | None) -> dict:
    params = {}
    if user_id:
        params["user_id"] = user_id
    if diet:
        params["diet"] = diet
    if allergies:
        params["allergies"] = list(allergies)
    return params


def recipes_search(query: str, user_id: str | None = None, diet: str | None = None,
                   allergies: list[str] | None = None, limit: int = 20,
                   base_url: str = DEFAULT_API_URL) -> list[dict]:
    params = {"q": query, "limit": limit} | _filters(user_id, diet, allergies)
    return _request("GET", base_url, "/recipes/search", params=params).json()


def recipes_onboarding(user_id: str | None = None, diet: str | None = None,
                       allergies: list[str] | None = None, n: int = 10,
                       base_url: str = DEFAULT_API_URL) -> list[dict]:
    params = {"n": n} | _filters(user_id, diet, allergies)
    return _request("GET", base_url, "/recipes/onboarding", params=params).json()


def recipe_next(user_id: str, diet: str | None = None, allergies: list[str] | None = None,
                skip: list[int] | None = None, seed: int | None = None,
                base_url: str = DEFAULT_API_URL) -> dict | None:
    """One recipe to rate, or None when there is nothing left to propose."""
    params = _filters(user_id, diet, allergies)
    if skip:
        params["skip"] = list(skip)
    if seed is not None:
        params["seed"] = seed
    try:
        return _request("GET", base_url, "/recipes/next", params=params).json()
    except APIError as e:
        if e.status == 404:
            return None
        raise


def recipe_ratings(user_id: str, base_url: str = DEFAULT_API_URL) -> list[dict]:
    return _request("GET", base_url, f"/recipe_ratings/{user_id}").json()


def rate_recipe(user_id: str, recipe_id: int, rating: int, base_url: str = DEFAULT_API_URL) -> None:
    _request("PUT", base_url, f"/recipe_ratings/{user_id}/{recipe_id}", json={"rating": rating})


def unrate_recipe(user_id: str, recipe_id: int, base_url: str = DEFAULT_API_URL) -> None:
    _request("DELETE", base_url, f"/recipe_ratings/{user_id}/{recipe_id}")
