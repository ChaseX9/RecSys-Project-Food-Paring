# Food Pairing — Ingredient Recommendation Engine

**Group 5: Pull_request** 

A recommender system that answers the everyday question *"I have X, what goes well with it?"*. The user enters one or more ingredients (e.g. `salmon`) and receives complementary ingredients that pair well together, filtered by the user's constraints (allergies, diet) and re-ranked by their tastes.

## Team

| Role | Member |
|---|---|
| Project Lead | Hugo Riviere |
| Data Engineer | Mael Vaudin |
| ML Engineer — Core | Jade Planterose |
| ML Engineer — Ops | Thibault Rabbé |
| Systems Engineer | - |

## Approach

- **Data:** Food.com recipes and user interactions (Kaggle). Optional chemical flavor relationships from FlavorDB / FlavorGraph.
- **Model:** graph-based recommender inspired by NLGCL (Xu et al., RecSys '25): a LightGCN-style encoder trained with a BPR loss plus a contrastive loss that uses the naturally existing neighbor layers of the GNN as contrastive views, instead of noisy graph augmentations.
- **Hybrid ranking:** embedding similarity + chemical score + hard constraint filtering. TODO: confirm final components.
- **Personalization:** user preference vector built from highly rated recipes.
- **Explanations:** short reasons for each recommendation (e.g. frequently paired together, shared aromatic compounds).

## Quick start

1. Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux or Mac
python3 -m venv .venv
source ./venv/bin/activate
```

2. Install Dependences
```bash
pip install -r requirements.txt
```

3. Build ingredient graph (NPMI + chemical)
```bash
python scripts/build_graph.py
```

4. Train NLGCL embeddings (~40 epochs, CPU is fine for 2.6k nodes)
```bash
python scripts/train_nlgcl.py --epochs 40 --device cpu
# or GPU:
python scripts/train_nlgcl.py --epochs 40 --device cuda
```

5. Recommend
```bash
python scripts/recommend.py salmon lemon --top-k 10
python scripts/recommend.py chicken --allergies gluten dairy --diet vegetarian
python scripts/recommend.py tomato basil --user-id 12345
```

6. Web interface (two terminals, from the repo root)
```bash
uvicorn api.main:app                 # API on http://localhost:8000
streamlit run app/streamlit_app.py   # UI on http://localhost:8501
```
The UI (in French, dark/light toggle) only talks to the API (`FOODPAIRING_API_URL`, default
`http://127.0.0.1:8000`). Pages: **Trouver des accords** (query + recommendations + feedback, with
the user's diet/allergies applied), **Mon frigo** (pairings using only what the user has
at home, via the `candidates` field of `/recommend`), **Mon historique**, **Mon profil**
(diet, allergies, disliked ingredients, Food.com profile link, e-mail, password),
**Administration** (admins only: overview, user management, logs, API checks), **À propos**.

Accounts are **mock** for now (kept in memory in `app/auth.py`, reset when Streamlit restarts)
until the user database exists. Demo logins: `admin` / `admin1234` (admin), `demo` / `demo1234`,
`camille` / `camille1234`, or "Continuer sans compte". `/recommend` is also still a mock: a
"Résultats d'exemple" badge is shown until the real ranker is wired into the API.

Ingredient photos come from [TheMealDB](https://www.themealdb.com) (free for educational use).
`python scripts/fetch_ingredient_images.py` matches the vocab to their ingredient list (exact,
then without qualifiers like "fresh"/"ground", then the head noun) and stores 96 px thumbnails in
`app/images/ingredients/` with the mapping in `app/ingredient_images.csv` (~93 % of ingredient
uses covered). Ingredients without a photo show a family emoji instead.

