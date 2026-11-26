"""Shared helpers for the Streamlit pages: data loaders, French labels, theme, small UI pieces."""
from __future__ import annotations

import re
import sys
import unicodedata
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st
from streamlit import config as st_config

from app import api_client
from app.i18n import is_en, t

PROCESSED = ROOT / "data" / "processed"
APP_DIR = Path(__file__).resolve().parent

ANONYMOUS = "anonymous"  # sent as user_id for guests (the API requires a non-empty id)
# query_ingredients sent with feedback given from the profile page (not after a search)
PROFILE_QUERY = "(profil)"

# Keys are what the API / ranker expect, values are what the user sees.
ALLERGIES = {"gluten": "Gluten", "dairy": "Produits laitiers", "nut": "Fruits à coque",
             "shellfish": "Crustacés", "egg": "Œufs", "soy": "Soja"}
DIETS = {"Aucun": None, "Végétarien": "vegetarian", "Vegan": "vegan", "Pescétarien": "pescatarian"}
DIET_LABELS = {v: k for k, v in DIETS.items() if v}
EXAMPLES = {
    "Saumon + citron": ["salmon", "lemon"],
    "Tomate + basilic": ["tomato", "basil"],
    "Poulet": ["chicken"],
    "Chocolat + fraise": ["chocolate", "strawberry"],
}
FEEDBACK = {  # api value -> (icon, French label). Effects: see api/db.py feedback_for_user.
    "like": ("👍", "J'aime"),             # boosts it and close ingredients on every search
    "dislike": ("👎", "Je n'aime pas"),   # never suggested again
    "not_for_me": ("🚫", "Pas avec ça"),  # not suggested again with this same search
    "cooked": ("🍳", "Cuisiné"),          # counts as a like
    "allergic": ("⚠️", "Allergique"),     # permanent ban of it and its variants
    "not_allergic": ("✅", "Plus allergique"),
    "click": ("➕", "Ajouté à la liste"),
}

# ------------------------------------------------------------------ theme
THEMES = {
    True: {"base": "dark", "primaryColor": "#9CC54A", "backgroundColor": "#111613",
           "secondaryBackgroundColor": "#1C241F", "textColor": "#E6EBE4"},
    False: {"base": "light", "primaryColor": "#55771C", "backgroundColor": "#FFFFFF",
            "secondaryBackgroundColor": "#F1F4EC", "textColor": "#1D261D"},
}


def ensure_theme(dark: bool) -> None:
    """Apply the dark/light theme. Streamlit sends the theme at the start of each run,
    so after changing it we rerun once for the browser to pick it up.
    Note: the theme lives in the server config, i.e. it is shared by all open tabs."""
    wanted = THEMES[dark]
    if any(st_config.get_option(f"theme.{k}") != v for k, v in wanted.items()):
        for k, v in wanted.items():
            st_config.set_option(f"theme.{k}", v)
        st.rerun()


# ------------------------------------------------------------ data loaders
@st.cache_data
def load_vocab() -> list[str]:
    """Ingredient names (English, as the model knows them), most frequent first."""
    path = PROCESSED / "ingredients_vocab.csv"
    if not path.exists():
        return sorted({i for ex in EXAMPLES.values() for i in ex})
    df = pd.read_csv(path)
    if "n_recipes" in df.columns:
        df = df.sort_values("n_recipes", ascending=False)
    return df["name"].astype(str).tolist()


@st.cache_data
def load_translations() -> dict[str, str]:
    path = APP_DIR / "ingredients_fr.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["en"], df["fr"]))


def fr(name: str) -> str:
    """Display name: the French one if we have it and the app is in French, else the original
    (English) name."""
    return name if is_en() else load_translations().get(name, name)


def fold(text: str) -> str:
    """Lowercase without accents, ligatures spelled out: "Œuf" -> "oeuf", "épinard" -> "epinard"."""
    text = text.lower().replace("œ", "oe").replace("æ", "ae")
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


def ingredient_label(name: str) -> str:
    """Label for the search box: French first, English in brackets so both can be typed.

    The box only matches the typed text against this label (plain substring, accents
    matter), so the accent-free spelling is added when it differs: "oeuf" finds "œuf".
    """
    tr = None if is_en() else load_translations().get(name)
    if not tr or tr == name:
        return name
    plain = fold(tr)
    return f"{tr} ({name}, {plain})" if plain != tr.lower() else f"{tr} ({name})"


@st.cache_data
def load_demo_users(n: int = 5) -> list[str]:
    """The n users with the richest preference profiles (most non-zero ingredients)."""
    try:
        profiles = sp.load_npz(PROCESSED / "user_profiles.npz").tocsr()
        ids = np.load(PROCESSED / "user_profile_ids.npy")
    except FileNotFoundError:
        return []
    order = np.argsort(-np.diff(profiles.indptr))[:n]
    return [str(int(u)) for u in ids[order]]


@st.cache_data(ttl=30, show_spinner=False)
def _display_names(url: str) -> dict[str, str]:
    """username -> display name of every account (cached: user_label runs once per table row)."""
    from app import auth

    try:
        return {u["username"]: u["display_name"] for u in auth.all_users()}
    except auth.AuthError:
        return {}


def user_label(user_id: str) -> str:
    """Readable name for any id seen by the API: guest, app account or Food.com profile."""
    from app import auth

    if user_id == ANONYMOUS or user_id.startswith(auth.GUEST_PREFIX):
        return t("Invité")
    account = _display_names(api_url()).get(user_id)
    if account:
        return account
    demo = load_demo_users()
    if user_id in demo:
        return t("Profil Food.com n°{n}", n=demo.index(user_id) + 1)
    return t("Utilisateur {id}", id=user_id)


def food_com_label(food_com_id: str) -> str:
    demo = load_demo_users()
    if food_com_id in demo:
        return t("Profil exemple n°{n} (id {id})", n=demo.index(food_com_id) + 1, id=food_com_id)
    return f"id {food_com_id}"


# ------------------------------------------------------------------- logo
@st.cache_data
def logo_data_uri(dark: bool) -> str | None:
    """ECE logo cut out of its white background, recoloured for the current theme."""
    path = APP_DIR / "images" / ("logo_ece_dark.png" if dark else "logo_ece_light.png")
    if not path.exists():
        return None
    import base64
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


# ------------------------------------------------------ ingredient photos
# Photos come from TheMealDB (see scripts/fetch_ingredient_images.py). Ingredients without one
# get an emoji for their family, checked in this order (first keyword found wins).
_FAMILY_EMOJI = [
    ("🧂", "salt pepper powder spice seasoning paprika cumin cinnamon nutmeg"),
    ("🍫", "chocolate cocoa"), ("🧀", "cheese parmesan mozzarella"),
    ("🥛", "milk cream yogurt yoghurt buttermilk"), ("🧈", "butter margarine"),
    ("🥚", "egg"), ("🍯", "honey syrup molasses"), ("🍬", "sugar candy marshmallow"),
    ("🍷", "wine sherry vodka rum brandy liqueur whiskey beer"),
    ("🫒", "oil olive"), ("🥫", "sauce ketchup mustard mayonnaise vinegar broth stock paste"),
    ("🦐", "shrimp prawn crab lobster scallop clam mussel oyster squid calamari octopus"),
    ("🐟", "fish salmon tuna cod trout anchovy sardine tilapia halibut"),
    ("🥓", "bacon ham prosciutto pancetta"), ("🌭", "sausage chorizo salami pepperoni"),
    ("🍗", "chicken turkey duck"), ("🥩", "beef steak pork lamb veal venison meat"),
    ("🍞", "bread bun baguette roll tortilla pita cracker"),
    ("🌾", "flour oat wheat cornmeal bran"), ("🍚", "rice quinoa couscous"),
    ("🍝", "pasta spaghetti macaroni noodle lasagna penne"),
    ("🥜", "nut peanut almond walnut pecan cashew pistachio hazelnut"),
    ("🫘", "bean lentil chickpea pea"), ("🌿", "basil parsley cilantro mint thyme rosemary "
                                         "oregano dill sage tarragon chive herb leaf"),
    ("🍋", "lemon lime"), ("🍊", "orange tangerine clementine mandarin"),
    ("🍓", "strawberr raspberr blueberr berr cherr cranberr"),
    ("🍎", "apple pear"), ("🍌", "banana"), ("🍍", "pineapple mango papaya"),
    ("🧄", "garlic"), ("🧅", "onion shallot leek scallion"), ("🍅", "tomato"),
    ("🌶️", "chili chile chilli jalapeno"), ("🫑", "bell pepper"),
    ("🥔", "potato"), ("🥕", "carrot"), ("🌽", "corn"), ("🍄", "mushroom"),
    ("🥦", "broccoli cauliflower cabbage kale"), ("🥬", "lettuce spinach greens arugula chard"),
    ("🥒", "cucumber zucchini courgette pickle"), ("🍆", "eggplant aubergine"),
    ("🥥", "coconut"), ("💧", "water ice"),
]


@st.cache_data
def _image_paths() -> dict[str, str]:
    path = APP_DIR / "ingredient_images.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["name"], df["image"]))


@st.cache_data(max_entries=2000)
def _image_uri(rel_path: str) -> str | None:
    import base64

    path = APP_DIR / rel_path
    if not path.exists():
        return None
    return "data:image/webp;base64," + base64.b64encode(path.read_bytes()).decode()


def ingredient_emoji(name: str) -> str:
    words = name.lower()
    for emoji, keywords in _FAMILY_EMOJI:
        if any(re.search(rf"\b{k}", words) for k in keywords.split()):
            return emoji
    return "🍽️"


def ingredient_img(name: str, size: int = 40) -> str:
    """HTML for an ingredient's thumbnail (photo, or family emoji when we have none)."""
    rel = _image_paths().get(name)
    uri = _image_uri(rel) if rel else None
    if uri:
        return (f"<img src='{uri}' alt='' style='width:{size}px;height:{size}px;"
                f"object-fit:contain;vertical-align:middle;margin-right:.4rem'>")
    return (f"<span style='display:inline-block;width:{size}px;text-align:center;"
            f"font-size:{size * 0.7:.0f}px;vertical-align:middle;margin-right:.4rem'>"
            f"{ingredient_emoji(name)}</span>")


# --------------------------------------------------------- recipe photos
# 1. The recipe's own photo on Food.com (scripts/link_foodcom_photos.py, ~123k recipes), loaded
#    from Food.com's CDN by the browser (the photos are not stored in the repo).
# 2. Below it, shown if 1 is missing or cannot load (offline): a photo of a similar dish from
#    TheMealDB (scripts/fetch_recipe_images.py, ~7k recipes), else a collage of the recipe's
#    ingredient photos. Pantry staples make a poor picture of a dish: they come last.
PHOTO_URL = ("https://img.sndimg.com/food/image/upload/w_{px},h_{px},c_fill,q_80/v1/img/recipes/"
             "{path}")
_STAPLES = {"salt", "pepper", "black pepper", "water", "sugar", "flour", "all-purpose flour",
            "baking soda", "baking powder", "oil", "vegetable oil", "olive oil", "butter",
            "egg", "milk", "kosher salt", "table salt", "ground black pepper", "cooking spray",
            "vanilla", "vanilla extract", "cornstarch", "ice"}


@st.cache_data
def _recipe_photo_paths() -> dict[int, str]:
    path = APP_DIR / "recipe_images.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["recipe_id"].astype(int), df["image"]))


@st.cache_data
def _foodcom_photo_paths() -> dict[int, str]:
    path = APP_DIR / "recipe_photos.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    return dict(zip(df["recipe_id"].astype(int), df["path"]))


def recipe_img(recipe_id: int, ingredients: list[str], size: int = 96) -> str:
    """HTML for a recipe thumbnail: the recipe's Food.com photo on top of the local picture
    (TheMealDB photo or ingredient collage), which shows when the photo is missing or fails."""
    local = _local_recipe_img(recipe_id, ingredients, size)
    path = _foodcom_photo_paths().get(int(recipe_id))
    if not path:
        return local
    url = escape(PHOTO_URL.format(px=min(2 * size, 320), path=path.replace(" ", "%20")), quote=True)
    return (f"<span style='position:relative;display:inline-block;width:{size}px;"
            f"height:{size}px;vertical-align:middle;flex:none'>{local}"
            f"<img src='{url}' alt='' loading='lazy' "
            f"title='{escape(t('Photo de la recette (Food.com)'), quote=True)}' "
            f"style='position:absolute;inset:0;width:{size}px;height:{size}px;object-fit:cover;"
            f"border-radius:10px'></span>")


def _local_recipe_img(recipe_id: int, ingredients: list[str], size: int = 96) -> str:
    """Picture available offline: TheMealDB dish photo, or a 2x2 collage of the recipe's
    ingredient photos (main ingredients first), or a family emoji when none has a photo."""
    box = (f"width:{size}px;height:{size}px;border-radius:10px;overflow:hidden;flex:none;"
           f"background:rgba(128,128,128,.15);display:inline-block;vertical-align:middle")
    rel = _recipe_photo_paths().get(int(recipe_id))
    uri = _image_uri(rel) if rel else None
    if uri:
        title = escape(t("Photo d'un plat similaire (TheMealDB)"), quote=True)
        return (f"<img src='{uri}' alt='' title='{title}' "
                f"style='{box};object-fit:cover'>")
    ordered = sorted(ingredients, key=lambda n: n in _STAPLES)  # stable: keeps recipe order
    uris = [u for u in (_image_uri(p) for p in (_image_paths().get(n) for n in ordered) if p) if u]
    uris = list(dict.fromkeys(uris))[:4]
    if not uris:
        emoji = ingredient_emoji(" ".join(ingredients)) if ingredients else "🍽️"
        return (f"<span style='{box};text-align:center;line-height:{size}px;"
                f"font-size:{size * 0.5:.0f}px'>{emoji}</span>")
    cell = size // 2 if len(uris) > 1 else size
    imgs = "".join(f"<img src='{u}' alt='' style='width:{cell}px;height:{cell}px;"
                   f"object-fit:contain;padding:3px;box-sizing:border-box'>" for u in uris)
    title = escape(t("Les ingrédients de la recette"), quote=True)
    return (f"<span style='{box};display:inline-flex;flex-wrap:wrap;align-content:center;"
            f"justify-content:center' title='{title}'>{imgs}</span>")


# ---------------------------------------------------------- explanations
_EXPLANATIONS = [
    (re.compile(r"often paired together in recipes"), "souvent cuisinés ensemble dans les recettes"),
    (re.compile(r"share aromatic compounds"), "partagent des molécules aromatiques"),
    (re.compile(r"chemically similar flavor profile"), "ont un profil aromatique proche"),
    (re.compile(r"close in the learned pairing embedding space"),
     "le modèle a appris qu'ils vont bien ensemble"),
    (re.compile(r"suggested by the pairing model"), "suggéré par le modèle"),
    # feedback loop (HybridRanker._feedback_reason)
    (re.compile(r"^you liked it$"), "tu l'as aimé"),
    (re.compile(r"^close to (.+), which you liked$"),
     lambda m: f"proche de {fr(m[1])}, que tu as aimé"),
]


def explanation_fr(text: str) -> str:
    """Turn the API's (English, technical) explanation into a short sentence, in French or
    left in English according to the app's language."""
    if text.startswith("[mock]"):
        return t("Suggestion d'exemple : le vrai modèle n'est pas encore branché.")
    if is_en():
        sentence = " and ".join(dict.fromkeys(p.strip() for p in text.split(" + ")))
        return sentence[:1].upper() + sentence[1:] + "."
    out = []
    for part in text.split(" + "):
        for pattern, french in _EXPLANATIONS:
            m = pattern.search(part.strip())
            if m:
                out.append(french(m) if callable(french) else french)
                break
        else:
            out.append(part)
    sentence = " et ".join(dict.fromkeys(out))
    return sentence[:1].upper() + sentence[1:] + "."


# ------------------------------------------------------------- UI pieces
def api_url() -> str:
    """App-wide API address (set by admins in the admin panel)."""
    from app import auth

    return auth.store().settings["api_url"]


@st.cache_data(ttl=10, show_spinner=False)
def api_status(url: str) -> tuple[bool, float]:
    """Cached /health ping for the sidebar, so every click doesn't hit the API."""
    return api_client.ping(url)


def mode_badge(is_mock: bool | None) -> None:
    if is_mock is None:
        return
    if is_mock:
        title = t("🧪 Résultats d'exemple")
        st.markdown(f":orange-badge[{title}]",
                    help=t("Le vrai modèle n'est pas encore branché sur l'API : "
                           "les suggestions sont tirées au hasard (mais toujours les mêmes)."))
    else:
        title = t("✅ Modèle réel")
        st.markdown(f":green-badge[{title}]")


API_DOWN_MSG = ("Le serveur de recommandations ne répond pas. "
                "Lance-le dans un terminal : `uvicorn api.main:app`")
