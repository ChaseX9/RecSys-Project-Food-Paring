#!/usr/bin/env python3
"""Rehearse the showcase demo: replay the live scenario on the API and check every effect.

    python scripts/demo_check.py                              # in-process API, temporary DB
    python scripts/demo_check.py --api-url http://127.0.0.1:8000   # the API run_app.py started

The scenario follows the demo script: a new user (no history) asks what goes with chicken,
likes thyme (it jumps to the top with its neighbours), finds thyme again on lamb (the
feedback carries over to other dishes), dislikes butter (gone everywhere), declares a walnut
allergy (every walnut product banned on every query), then lifts it. Every step prints what
the audience sees (French names, as in the app) and PASS / FAIL, so a change to the
model or the weights that breaks the story is caught before the day.

Also the terminal plan B: it needs no browser and no network (in-process mode), only the
laptop. Exit code 1 if a step fails.

Each run uses a fresh user id (demo-check-xxxx), so it never touches real accounts. Against a
live API this user's rows stay in data/app.db (harmless; listed under Administration).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOP_K = 15          # the demo sets the "Nombre de suggestions" slider to 15
LIKE = "thyme"      # shown at rank 11-15 for chicken: the like visibly moves it up
DISLIKE = "butter"
ALLERGY = "walnut"


def load_fr() -> dict[str, str]:
    path = ROOT / "app" / "ingredients_fr.csv"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["en"]: r["fr"] for r in csv.DictReader(f)}


FR = load_fr()


def label(name: str) -> str:
    return f"{FR[name]} ({name})" if name in FR and FR[name] != name else name


class Demo:
    def __init__(self, client):
        self.c = client
        self.user = f"demo-check-{uuid.uuid4().hex[:6]}"
        self.failures: list[str] = []

    def recommend(self, ingredients, user=None, top_k=TOP_K, **extra) -> list[dict]:
        body = {"user_id": user or self.user, "ingredients": ingredients, "top_k": top_k, **extra}
        r = self.c.post("/recommend", json=body)
        r.raise_for_status()
        return r.json()["recommendations"]

    def feedback(self, query, ingredient, feedback):
        r = self.c.post("/log_interaction", json={"user_id": self.user, "query_ingredients": query,
                                                  "recommended_ingredient": ingredient,
                                                  "feedback": feedback})
        r.raise_for_status()
        print(f"  -> POST /log_interaction  {feedback} {label(ingredient)}  (id {r.json()['id']})")

    def check(self, ok: bool, what: str):
        print(f"  [{'PASS' if ok else 'FAIL'}] {what}")
        if not ok:
            self.failures.append(what)


def show(recs: list[dict], highlight=()):
    for r in recs:
        mark = " <==" if r["name"] in highlight else ""
        note = f"   [{r['feedback']}]" if r.get("feedback") else ""
        print(f"    {r['rank']:>2}. {label(r['name'])}{note}{mark}")


def names(recs):
    return [r["name"] for r in recs]


def run(d: Demo) -> None:
    fresh = f"{d.user}-control"   # same queries, no feedback: what a new visitor sees

    print("\n0. Health")
    health = d.c.get("/health").json()
    d.check(health.get("model") == "nlgcl", f"API serves the trained model (model = {health.get('model')})")

    print(f"\n1. New user, no history: what goes with chicken? (top {TOP_K})")
    before = d.recommend(["chicken"])
    show(before, (LIKE,))
    d.check(LIKE in names(before) and names(before).index(LIKE) >= 3,
            f"{label(LIKE)} is visible but not in the top 3 (rank "
            f"{names(before).index(LIKE) + 1 if LIKE in names(before) else '-'})")

    print(f"\n2. Like {label(LIKE)} -> search again")
    d.feedback(["chicken"], LIKE, "like")
    after = d.recommend(["chicken"])
    show(after, (LIKE,))
    d.check(LIKE in names(after)[:3], f"{label(LIKE)} moves to the top 3")
    neighbours = [r["name"] for r in after if r.get("feedback", "") and r["name"] != LIKE]
    d.check(bool(neighbours), "its neighbours in the NLGCL space go up too ('close to thyme, "
                              f"which you liked'): {', '.join(map(label, neighbours)) or 'none'}")

    print(f"\n3. Another dish: lamb (top 10). Left: new visitor, right: this user")
    ctrl, mine = d.recommend(["lamb"], user=fresh, top_k=10), d.recommend(["lamb"], top_k=10)
    for a, b in zip(ctrl, mine):
        print(f"    {a['rank']:>2}. {label(a['name']):<34s} {label(b['name'])}"
              + (f"   [{b['feedback']}]" if b.get("feedback") else ""))
    gained = [n for n in names(mine) if n not in names(ctrl)]
    d.check(LIKE in gained or any(r.get("feedback") for r in mine),
            f"the like carries over to another dish: new in the top 10 = {', '.join(map(label, gained)) or 'none'}")

    print(f"\n4. Dislike {label(DISLIKE)} (on chicken) -> potato")
    ctrl = d.recommend(["potato"], user=fresh, top_k=10)
    d.check(DISLIKE in names(ctrl), f"{label(DISLIKE)} is in a new visitor's top 10 for potato")
    d.feedback(["chicken"], DISLIKE, "dislike")
    mine = d.recommend(["potato"], top_k=10)
    show(mine)
    d.check(DISLIKE not in names(d.recommend(["potato"], top_k=50)),
            f"{label(DISLIKE)} is never suggested again, on any dish")

    print(f"\n5. Allergy: banana, then 'I'm allergic' to {label(ALLERGY)}")
    banana = d.recommend(["banana"], top_k=10)
    show(banana, (ALLERGY,))
    d.check(ALLERGY in names(banana), f"{label(ALLERGY)} is in the top 10 for banana")
    wide_before = [n for n in names(d.recommend(["apple"], user=fresh, top_k=50)) if ALLERGY in n]
    d.feedback(["banana"], ALLERGY, "allergic")
    banana_after = d.recommend(["banana"], top_k=10)
    show(banana_after)
    leaked = [n for q in (["banana"], ["apple"], ["chocolate"])
              for n in names(d.recommend(q, top_k=50)) if ALLERGY in n]
    d.check(not leaked, f"no '{ALLERGY}' product left on banana / apple / chocolate (top 50); "
                        f"banned on apple: {', '.join(map(label, wide_before)) or '-'}")

    print(f"\n6. Mistake: 'not allergic' to {label(ALLERGY)}")
    d.feedback(["banana"], ALLERGY, "not_allergic")
    d.check(ALLERGY in names(d.recommend(["banana"], top_k=10)), f"{label(ALLERGY)} is back")

    print("\n7. What the model learned (GET /profile)")
    prof = d.c.get(f"/profile/{d.user}").json()
    likes = [x["name"] for x in prof.get("likes", [])]
    dislikes = [x["name"] for x in prof.get("dislikes", [])]
    print(f"    likes:    {', '.join(map(label, likes[:6]))}")
    print(f"    dislikes: {', '.join(map(label, dislikes[:6]))}")
    d.check(LIKE in likes and DISLIKE in dislikes, "the profile shows the like and the dislike")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--api-url", default=None,
                   help="running API (e.g. http://127.0.0.1:8000); default: in-process, temporary DB")
    args = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):   # French names (œ, é) on a Windows console
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.api_url:
        import httpx
        with httpx.Client(base_url=args.api_url, timeout=30) as c:
            d = Demo(c)
            print(f"API {args.api_url}, user {d.user}")
            run(d)
    else:
        os.environ.setdefault("FOODPAIRING_SEED_DEMO", "0")
        from fastapi.testclient import TestClient
        from api.main import create_app
        # ignore_cleanup_errors: on Windows SQLite may still hold demo.db when the dir is removed
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp, \
                TestClient(create_app(str(Path(tmp) / "demo.db"))) as c:
            d = Demo(c)
            print(f"In-process API (temporary DB), user {d.user}")
            run(d)

    print("\n" + ("All demo steps OK." if not d.failures else
                  f"{len(d.failures)} step(s) FAILED:\n  - " + "\n  - ".join(d.failures)))
    sys.exit(1 if d.failures else 0)


if __name__ == "__main__":
    main()
