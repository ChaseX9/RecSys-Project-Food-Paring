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
    pass


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
        raise APIError(f"{method} {path} -> HTTP {r.status_code} : {r.text[:300]}")
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
) -> tuple[dict, float]:
    """POST /recommend. Returns (response body, latency in seconds).

    `candidates` restricts the suggestions to these ingredients (e.g. the user's fridge)."""
    payload = {"user_id": user_id, "ingredients": ingredients, "top_k": top_k,
               "allergies": allergies or None, "diet": diet}
    if candidates is not None:
        payload["candidates"] = candidates
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
