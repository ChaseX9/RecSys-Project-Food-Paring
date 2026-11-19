"""Page "Mon frigo": what goes well together, using only what the user already has at home.

Two modes:
- around one ingredient (e.g. salmon): which fridge items pair best with it;
- no base ingredient: the best pairs that can be made from the fridge alone.
Both rely on the API's `candidates` whitelist, so only fridge items are ever suggested.
The API also applies the user's feedback (likes, dislikes, allergies) by itself.
"""
from __future__ import annotations

import streamlit as st

from app import api_client, auth
from app.common import (ALLERGIES, API_DOWN_MSG, DIET_LABELS, api_url, explanation_fr, fr,
                        ingredient_img, ingredient_label, load_vocab, mode_badge)

QUICK_ADD = ["egg", "butter", "milk", "lemon", "garlic", "onion", "tomato", "cheese", "potato",
             "rice", "pasta", "chicken", "olive oil", "carrot", "honey", "basil"]
MIN_FRIDGE = 2

user = auth.current_user()
diet = user.diet if user else None
allergies = list(user.allergies) if user else []
disliked = set(user.disliked) if user else set()

# The fridge widget starts from the saved profile (or stays in the session for guests).
owner = user.username if user else auth.GUEST
if st.session_state.get("fridge_owner") != owner or "fridge_sel" not in st.session_state:
    st.session_state.fridge_owner = owner
    st.session_state.fridge_sel = list(user.fridge) if user else \
        st.session_state.get("guest_fridge", [])


def save_fridge() -> None:
    items = list(st.session_state.fridge_sel)
    u = auth.current_user()
    if u:
        auth.update_profile(u.username, fridge=items)
    else:
        st.session_state.guest_fridge = items
    st.session_state.pop("fridge_results", None)  # results no longer match the fridge


def quick_add(name: str) -> None:
    if name not in st.session_state.fridge_sel:
        st.session_state.fridge_sel = st.session_state.fridge_sel + [name]
        save_fridge()


def empty_fridge() -> None:
    st.session_state.fridge_sel = []
    save_fridge()


def like(query: list[str], name: str) -> None:
    try:
        api_client.log_interaction(auth.log_user_id(auth.current_user()), query, "like", name,
                                   base_url=api_url())
        st.session_state.setdefault("fridge_liked", set()).add((tuple(query), name))
        st.toast(f"Noté : {fr(query[0])} + {fr(name)} 👍")
    except api_client.APIError:
        st.toast("Impossible d'enregistrer ton avis : le serveur ne répond pas.", icon="❌")


def ask(query: list[str], candidates: list[str] | None, top_k: int) -> dict:
    return api_client.recommend(query, auth.log_user_id(user), top_k, allergies, diet,
                                base_url=api_url(), candidates=candidates,
                                exclude=sorted(disliked),
                                history_user_id=auth.history_user_id(user))[0]


# ---------------------------------------------------------------- my fridge
st.title("🧊 Mon frigo")
st.markdown("Dis-nous ce que tu as chez toi : on te dit **quoi associer avec ce que tu as "
            "déjà**, sans rien avoir à acheter.")

st.multiselect("Ce que j'ai dans mon frigo et mes placards", load_vocab(), key="fridge_sel",
               on_change=save_fridge, format_func=ingredient_label,
               placeholder="Ajoute tes ingrédients (saumon, œufs, citron…)")
fridge = list(st.session_state.fridge_sel)
if fridge:  # the fridge at a glance
    st.markdown(
        "<div style='display:flex;flex-wrap:wrap;gap:.5rem;margin:.2rem 0 .6rem'>" + "".join(
            f"<div style='display:flex;flex-direction:column;align-items:center;width:4.6rem;"
            f"font-size:.75rem;text-align:center;line-height:1.1'>"
            f"{ingredient_img(f, 44).replace('margin-right:.4rem', 'margin:0 0 .2rem')}"
            f"{fr(f)}</div>" for f in fridge) + "</div>", unsafe_allow_html=True)

missing = [q for q in QUICK_ADD if q not in fridge and q in set(load_vocab())][:8]
if missing:
    cols = st.columns([1.3] + [1] * len(missing))
    cols[0].caption("Ajout rapide :")
    for col, name in zip(cols[1:], missing):
        col.button(f"+ {fr(name)}", key=f"qa_{name}", on_click=quick_add, args=(name,),
                   width="stretch")

info, clear = st.columns([4, 1], vertical_alignment="center")
constraints = ([DIET_LABELS[diet].lower()] if diet else []) + \
    [f"sans {ALLERGIES[a].lower()}" for a in allergies]
info.caption(f"🧊 {len(fridge)} ingrédient(s)"
             + (f" · selon ton profil : {', '.join(constraints)}" if constraints else "")
             + ("" if user else " · 💡 crée un compte pour garder ton frigo d'une visite à "
                "l'autre"))
if fridge:
    with clear.popover("🗑️ Vider", width="stretch"):
        st.write("Retirer tous les ingrédients du frigo ?")
        st.button("Oui, vider", on_click=empty_fridge, type="primary")

if len(fridge) < MIN_FRIDGE:
    st.info(f"Ajoute au moins {MIN_FRIDGE} ingrédients pour voir ce que tu peux associer.")
    st.stop()

# Disliked items are never suggested, even if they are in the fridge.
usable = [f for f in fridge if f not in disliked]

# -------------------------------------------------------------- what to do
st.divider()
st.subheader("🍳 Qu'est-ce que je peux faire avec ?")
mode = st.radio("Mode", ["Autour d'un ingrédient", "Les meilleures associations de mon frigo"],
                horizontal=True, label_visibility="collapsed",
                captions=["Ex. « j'ai du saumon, je le mets avec quoi ? »",
                          "On cherche les duos qui marchent le mieux dans ton frigo"])

if mode == "Autour d'un ingrédient":
    others = [v for v in load_vocab() if v not in set(fridge)]
    base = st.selectbox(
        "Ingrédient de base", fridge + others, format_func=lambda n:
        f"🧊 {ingredient_label(n)}" if n in fridge else ingredient_label(n),
        help="Un ingrédient de ton frigo (🧊) ou n'importe quel autre.")
    if st.button("🔍 Trouver les associations", type="primary"):
        cands = [f for f in usable if f != base]
        try:
            with st.spinner("On fouille ton frigo…"):
                in_fridge = ask([base], cands, min(max(len(cands), 1), 50)) if cands else \
                    {"recommendations": [], "mock": None}
                extra = ask([base], None, min(5 + len(fridge), 50))
            to_buy = [r for r in extra["recommendations"] if r["name"] not in set(fridge)][:5]
            returned = {r["name"] for r in in_fridge["recommendations"]}
            st.session_state.fridge_results = {
                "mode": "base", "base": base, "recs": in_fridge["recommendations"],
                "to_buy": to_buy, "mock": extra.get("mock"),
                "filtered": [c for c in cands if c not in returned],
            }
        except api_client.APIError:
            st.error(API_DOWN_MSG)
else:
    if st.button("🔍 Trouver les meilleurs duos", type="primary"):
        pairs: dict[frozenset, dict] = {}
        mock = None
        try:
            with st.spinner("On teste toutes les combinaisons de ton frigo…"):
                for item in usable:
                    cands = [f for f in usable if f != item]
                    body = ask([item], cands, min(len(cands), 50))
                    mock = body.get("mock")
                    for r in body["recommendations"]:
                        key = frozenset((item, r["name"]))
                        p = pairs.setdefault(key, {"a": item, "b": r["name"], "scores": [],
                                                   "explanation": r["explanation"]})
                        p["scores"].append(r["score"])
            ranked = sorted(({**p, "score": sum(p["scores"]) / len(p["scores"])}
                             for p in pairs.values()), key=lambda p: -p["score"])
            # Star = the item that pairs well with the most things (mean of its 3 best duos).
            per_item: dict[str, list[float]] = {}
            for p in ranked:
                per_item.setdefault(p["a"], []).append(p["score"])
                per_item.setdefault(p["b"], []).append(p["score"])
            star = max(per_item, key=lambda i: sum(sorted(per_item[i])[-3:]) /
                       len(sorted(per_item[i])[-3:])) if per_item else None
            st.session_state.fridge_results = {"mode": "pairs", "pairs": ranked[:10],
                                               "star": star, "mock": mock}
        except api_client.APIError:
            st.error(API_DOWN_MSG)

# ------------------------------------------------------------------ results
res = st.session_state.get("fridge_results")
liked = st.session_state.get("fridge_liked", set())


def card(title: str, level: float, explanation: str, query: list[str], name: str,
         pictured: list[str]) -> None:
    with st.container(border=True):
        left, mid, right = st.columns([3, 2, 1], vertical_alignment="center")
        pics = "".join(ingredient_img(p, 40) for p in pictured)
        left.markdown(f"{pics}**{title}**  \n<span style='opacity:.75;font-size:.9em'>💡 "
                      f"{explanation_fr(explanation)}</span>", unsafe_allow_html=True)
        mid.progress(max(0.0, min(1.0, level)), text="Niveau d'accord")
        if (tuple(query), name) in liked:
            right.caption("👍 noté")
        else:
            right.button("👍", key=f"like_{'+'.join(query)}_{name}", on_click=like,
                         args=(query, name), width="stretch", help="Bonne idée !")


if res:
    st.divider()
    mode_badge(res.get("mock"))
    if res["mode"] == "base":
        base, recs = res["base"], res["recs"]
        st.subheader(f"Avec ce que tu as, {fr(base)} va le mieux avec :")
        if not recs:
            st.warning("Rien dans ton frigo ne s'accorde avec cet ingrédient en respectant "
                       "tes contraintes. Regarde les idées à acheter juste en dessous.")
        best = recs[0]["score"] if recs else 1.0
        for i, r in enumerate(recs, 1):
            card(f"{i}. {fr(r['name']).capitalize()}", r["score"] / best if best > 0 else 0,
                 r["explanation"], [base], r["name"], [r["name"]])
        if res["filtered"]:
            st.caption("Non proposés à cause de ton régime, de tes allergies ou de tes avis : "
                       + ", ".join(fr(f) for f in res["filtered"]) + ".")
        if res["to_buy"]:
            st.markdown("#### 🛒 Pour aller plus loin, à acheter")
            st.caption("Ces ingrédients iraient très bien aussi, mais tu ne les as pas.")
            st.markdown(" &nbsp; ".join(f"{ingredient_img(r['name'], 32)}**{fr(r['name'])}**"
                                        for r in res["to_buy"]), unsafe_allow_html=True)
    else:
        pairs = res["pairs"]
        if res["star"]:
            st.success(f"⭐ **L'ingrédient star de ton frigo : {fr(res['star'])}**. C'est "
                       "celui qui s'accorde avec le plus de choses que tu as.")
        st.subheader("Les meilleurs duos avec ce que tu as")
        if not pairs:
            st.warning("Aucune association possible avec tes contraintes actuelles.")
        best = pairs[0]["score"] if pairs else 1.0
        for i, p in enumerate(pairs, 1):
            card(f"{i}. {fr(p['a']).capitalize()} + {fr(p['b'])}",
                 p["score"] / best if best > 0 else 0, p["explanation"], [p["a"]], p["b"],
                 [p["a"], p["b"]])
