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
- **Hybrid ranking:** embedding similarity + co-occurrence in train recipes + chemical score, each min-max normalized per query, then hard constraint filtering (allergies, diet). Weights in `configs/model.yaml`, chosen on the val split.
- **Personalization:** user preference vector built from highly rated recipes.
- **Explanations:** short reasons for each recommendation (e.g. frequently paired together, shared aromatic compounds).


## Evaluation

The evaluation harness (`src/eval/`) scores any recommender on held-out recipes and reports **NDCG@10**, **Recall@20**, HitRate@k and MRR.

**Protocol: masked-ingredient completion.** For each held-out user → recipe interaction, about 20% of the recipe's ingredients are hidden (at least one). The remaining ingredients are the query, the hidden ones are the ground truth, and the user id is passed for personalization. Query ingredients are removed from every prediction before scoring.

**No leakage.** By default (`protocol: unseen_recipes`) only recipes with **no train interaction** are evaluated (a recipe seen in both val and test interactions is kept for test only, so tuning on val never touches a test recipe), so the graph, NPMI, popularity and user profiles (all built on train recipes) have never seen them. A stricter recipe-level holdout (`protocol: recipe_holdout`) is also available; it requires the graph and model to be rebuilt without the held-out recipes (`src.eval.protocol.holdout_recipe_ids`, same `holdout_frac` and `seed`).

Settings (protocol, split, share of hidden ingredients, k, seed, baselines) are in `configs/eval.yaml`.

```bash
# Baselines (random, popularity, co-occurrence) on the test split
python scripts/evaluate.py

# Validation split, for tuning
python scripts/evaluate.py --split val

# Add the trained model (after scripts/train_nlgcl.py)
python scripts/evaluate.py --embeddings artifacts/ingredient_embeddings.npy
```

Results are printed and saved to `artifacts/eval/<protocol>_<split>.json`.

From Python, mock or precomputed predictions can be scored directly:

```python
from src.eval.protocol import build_cases
from src.eval.harness import evaluate_predictions

setup = build_cases("data/processed", split="test")
preds = {case_id: [12, 7, 40] for case_id in setup.cases.case_id}   # {case_id: ranked ingredient ids}
print(evaluate_predictions(preds, setup.cases, ks=(10, 20)))         # {'ndcg@10': ..., 'recall@20': ..., ...}
```

Tests: `pytest tests/test_eval.py` (hand-computed metric values, mock predictions, anti-leakage checks).

### Ablations

`scripts/ablations.py` trains each model variant listed in `configs/ablations.yaml` and scores it, with each ranking variant, on the same cases as the baselines:

- **Model:** NLGCL (reference, `lambda_cl = 0.1`) vs LightGCN (`lambda_cl = 0`, i.e. the innovation switched off), the former `lambda_cl = 1e-5`, and a graph built without chemical edges. Every variant uses the same validation and early stopping.
- **Ranking:** served hybrid score vs without the embedding (is the GNN useful next to co-occurrence?), without co-occurrence, without chemistry, with the user profile, a model-led score (no co-occurrence) and the embedding alone.
- **Beyond accuracy (top-10):** catalog coverage, long-tail share, novelty, and `grounded@10`, the share of recommendations backed by a real explanation signal (strong NPMI with a query ingredient or shared aromatic compounds).

```bash
python scripts/ablations.py                    # val + test, about 35 min on CPU
python scripts/ablations.py --max-cases 300    # quick run
```

Trained variants are cached in `artifacts/ablations/<variant>/` (use `--retrain` to retrain them). Tables are written to `artifacts/eval/ablations_unseen_recipes_<split>.{json,md}`. Pick settings on `val`, report `test`.

### Ranking weights

`scripts/tune_ranking.py` grid-searches the hybrid weights on **val** (1,630 points: embedding, chemistry, co-occurrence, cosine or dot, for NLGCL and LightGCN), then prints **test** for the chosen points only. Each case's score components are computed once and every weight set is scored with numpy, which takes about 10 min instead of hours. At startup it checks that its NDCG@10 matches the real harness. Results are written to `artifacts/eval/tune_ranking.{json,md}`.

NDCG@10, NLGCL, served personalization (`gamma_pref = 0.2`):

| ranking | alpha | beta (chem) | delta (cooc) | val | test |
|---|---:|---:|---:|---:|---:|
| previous served | 0.35 | 0.1 | 0.7 | 0.2985 | 0.2995 |
| **served now** | 0.35 | **0.025** | 0.7 | **0.3134** | **0.3168** |
| no chemistry | 0.35 | 0 | 0.7 | 0.3159 | 0.3190 |
| model-led (alpha = delta) | 0.5 | 0.025 | 0.5 | 0.3001 | 0.3001 |
| embedding alone (dot) | 1 | 0 | 0 | 0.2342 | 0.2352 |

- **Chemistry:** every `beta > 0` lowers NDCG on val, steadily (0.1: 0.2985, 0.05: 0.3106, 0.025: 0.3134, 0: 0.3159). FlavorGraph's shared compounds are not what Food.com recipes put together. We keep `beta = 0.025`, the best val point with chemistry on. It gets 5 % more NDCG than before and loses 0.8 % against no chemistry. In exchange it keeps grounded@10 at 0.925 (0.916 without chemistry) and keeps the "shared aromatic compounds" explanation in the app.
- **Model-led hybrid:** we tested whether giving the embedding at least as much weight as co-occurrence would carry NLGCL's gain into the served ranking. It does not. The best model-led point (0.3001 to 0.3023 on val) stays below the co-occurrence-led hybrid. NLGCL beats LightGCN on the embedding alone (val 0.2342 vs 0.1974, +19 %), but LightGCN is slightly ahead in every hybrid (best val 0.3250 vs 0.3173). Co-occurrence already gives the hybrid the frequency signal that LightGCN's embedding adds. NLGCL's contrastive term spreads the embedding toward rarer ingredients, and in the hybrid that spread is redundant or counts against it. We keep NLGCL as the served encoder because its embedding is better on its own, and the embedding drives cold queries, feedback neighbours and "close to X" explanations.
- No model change: same vocabulary, same `artifacts/ingredient_embeddings.npy`, no retraining needed.

```bash
python scripts/tune_ranking.py                    # full grid, about 10 min on CPU
python scripts/tune_ranking.py --max-cases 300    # quick run
```

### Personalization (ratings 1-5)

User profiles use **every train rating**, not only the 5/5 ones (`src/data/rating_profiles.py`):

- **Centered, shrunk ratings:** `weight = rating - b_u`, where `b_u` is the user's mean rating pulled toward the global mean (4.69). A user who rates everything 5/5 (26% of users) still gets a positive profile. Only ratings **strictly below 3** (1/5, 2/5) count as dislikes. Weights are scaled so that one typical rating weighs about 1 for every user.
- **Consistency per ingredient:** a rating is about a whole recipe, not about each of its ingredients. `pref(i) = (P_i - N_i) / (P_i + N_i + alpha)`, with `P_i` / `N_i` the weight of liked / disliked recipes containing `i`. Flour in 25 liked and 2 disliked cakes stays liked; an ingredient only seen in disliked recipes becomes negative; `alpha = 5` keeps a single recipe from being decisive. (Normalizing likes and dislikes separately let 2 disliked cakes weigh as much as 25 liked ones: a baker ended up "disliking" flour.)
- **Confidence:** `n / (n + 10)`: a user with 5 ratings is trusted less than one with 50.
- **Multiplicative gating:** `score = pairing * (1 + gamma * pref)`. The preference only re-orders plausible pairings: a baker's love of flour no longer pushes flour next to salmon.
- **Level 2 (embedding user vector):** `pref_source: embedding | hybrid` builds the user vector in the NLGCL space (liked recipes minus disliked ones). Implemented and evaluated, not served (no gain on val).

Settings: `ranking` block of `configs/model.yaml`. Study (val to choose, test to report, with a **shuffled-user control**: same cases with permuted user ids; the gap is the information that really comes from the user's tastes):

```bash
python scripts/eval_personalization.py                         # standard protocol
python scripts/eval_personalization.py --config <copy of configs/personalization.yaml with eval_config: configs/eval_short_query.yaml>
```
`configs/eval_short_query.yaml` mimics real usage ("I have salmon"): 70% of the recipe is hidden, so the query is short and the user's taste has more room to matter.

### Feedback loop

Every explicit feedback sent to `POST /log_interaction` (with `recommended_ingredient`) **updates the user's taste profile**, and the next `POST /recommend` uses it. No retraining needed (`api/db.py: feedback_for_user`, `HybridRanker.preference(user_id, liked, disliked)`):

| feedback | effect on the profile |
|---|---|
| `like`, `cooked` | preference of the ingredient set to 1.5 (rating-based values are in [-1, 1]); its close neighbours in the NLGCL space (cosine > `feedback_sim_threshold`) are raised proportionally |
| `dislike` | preference set to -1 and the ingredient is never recommended again; its neighbours are **not** touched |
| `not_for_me` | ingredient excluded only for the same query ("not with this dish") |
| `allergic` | **permanent** ban of the ingredient and of every ingredient containing it as a word (`peanut` -> peanut butter, creamy peanut butter, peanut oil...), on every query; a later `like` does **not** lift it |
| `not_allergic` | removes the allergy (the only way to lift it, e.g. after a mistake); the ingredient goes back to its previous like / dislike state |
| `click` | logged, not applied (implicit, too weak) |

- **Latest feedback wins:** the state of an ingredient is the most recent feedback on it, so a user who disliked cilantro and later likes it gets cilantro back. Allergies are the exception: only `not_allergic` lifts them.
- **Existing databases:** `data/app.db` files created before `allergic` / `not_allergic` existed are migrated automatically at startup (rows, ids and timestamps kept).
- **Cold start:** a new app user (no Food.com history) starts with an empty profile, built from their very first like.
- The preference stays **multiplicative**: a liked ingredient goes up where it pairs well (coconut milk with chicken), not next to anything.
- `GET /profile/{user_id}` shows the current profile (ratings history + feedback), `GET /feedback/{user_id}` the raw feedback state. `/recommend` returns `applied_feedback` and, per item, `feedback` ("you liked it", "close to thyme, which you liked").
- **Recipe ratings from the app:** app users rate Food.com recipes 1-5 (`PUT /recipe_ratings/{user_id}/{recipe_id}`, stored in `recipe_ratings`). On every `/recommend` and `/profile`, their ratings (plus the train ratings of a linked Food.com user) are turned into the same rating profile as offline (`RatingData.profile_for`, same formula and train global mean; replaying a Food.com user's ratings gives exactly the precomputed profile). This is the cold start for app users: about 10 ratings already personalize. It changes the preference score only, not the embeddings (no retraining).
- Recipe routes: `GET /recipes/search?q=` (every word in the name, most rated first), `GET /recipes/onboarding` (popular recipes, as different as possible from each other: greedy on ingredient and title-word Jaccard), `GET /recipes/next` (one popular recipe made of ingredients the user's ratings don't cover yet, randomized by `seed`, `skip` to pass), `GET|DELETE /recipe_ratings/...`. They take `diet` / `allergies` and apply the ranker's keyword rules to the ingredients **and** to the recipe title, plus title rules for meat / fish (the cleaned ingredient lists sometimes lose the main ingredient: "Pulled Pork (Crock Pot)" = onion, ginger ale, barbecue sauce).
- Optional `/recommend` fields: `candidates` (only these ingredients may be returned, e.g. a fridge), `exclude` (never returned, not stored as feedback) and `history_user_id` (whose rating history personalizes, e.g. a Food.com profile linked to an app account; feedback is still read from `user_id`). `/profile` takes `history_user_id` too.

`/recommend` serves the trained model when `artifacts/ingredient_embeddings.npy` exists (`"mock": false`, check `GET /health`), and falls back to the mock otherwise or with `FOODPAIRING_MOCK=1`.

## Quick start: run the app (about 5 minutes)

Needs Python 3.10+ and git. Nothing to download or train: the processed data (`data/processed/`) and the trained model (`artifacts/ingredient_embeddings.npy`) are in the repo.

1. Clone and create a virtual environment
```bash
git clone https://github.com/ChaseX9/RecSys-Project-Food-Paring.git
cd RecSys-Project-Food-Paring

# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\activate

# Linux or Mac
python3 -m venv .venv
source ./.venv/bin/activate
```

2. Install the dependencies (no PyTorch needed to run the app)
```bash
pip install -r requirements.txt
```

3. Start the API and the app with one command
```bash
python scripts/run_app.py
```
Open http://localhost:8501 and log in with `demo` / `demo1234` (or `admin` / `admin1234`, or "Continuer sans compte"). The terminal must show `API ready: model = nlgcl` (the real model, not the mock). `Ctrl+C` stops everything. If a port is already used by another program (another Streamlit app on 8501, an API on 8000...), the script takes the next free one and says so: open the URL it prints. To choose: `python scripts/run_app.py --api-port 8010 --ui-port 8510`.

4. Check (optional): `pytest` (all tests, about 8 min) or `pytest tests/test_artifacts.py tests/test_api.py` (1 min).

### Showcase: share the app on the local network

Free, no cloud account: one laptop serves the app, every phone or laptop on the **same Wi-Fi** opens it.
```bash
python scripts/run_app.py --lan
```
The terminal prints `Showcase URL (same Wi-Fi): http://<laptop IP>:8501` and saves a QR code of it to
`artifacts/showcase_qr.png` (show it on the stand). Visitors scan it and pick "Continuer sans compte":
each guest gets their own profile.

- **Only the app is exposed.** The Streamlit app listens on `0.0.0.0`; the API stays on `127.0.0.1`
  (it has no authentication of its own), and the app calls it locally.
- **Windows firewall:** on the first `--lan` start, allow Python in the popup (tick Private *and*
  Public, school Wi-Fi is often classified Public). If the popup was dismissed, as admin:
  `netsh advfirewall firewall add rule name=FoodPairing dir=in action=allow protocol=TCP localport=8501`.
- **School Wi-Fi that blocks device-to-device traffic** (eduroam and many guest networks do): the URL
  answers on the laptop but not on phones. Use a **phone hotspot** instead: connect the laptop and the
  visitors to it and restart with `--lan` (the IP changes, so does the QR code).
- **Wrong IP shown** (VPN, Ethernet + Wi-Fi): `python scripts/run_app.py --lan --ip 192.168.1.10`.

### Demo rehearsal

`scripts/demo_check.py` replays the live demo on the API and checks that each effect shows. A new user asks what goes with chicken, then likes thyme. Thyme jumps to the top, and so do parsley and rosemary ("close to thyme, which you liked"). The like carries over to lamb. A dislike of butter removes it from every dish. A walnut allergy bans walnut products on every query, and "not allergic" lifts it. Each step prints what the audience sees (French names, as in the app) followed by PASS or FAIL. Run it after any change to the model or the weights, and before the showcase:
```bash
python scripts/demo_check.py                                  # in-process API, temporary DB (~30 s)
python scripts/demo_check.py --api-url http://127.0.0.1:8000  # the API started by run_app.py
```
It needs neither a browser nor a network, so it is also the plan B if the app cannot be shown. It runs as a fresh `demo-check-xxxx` user and never touches real accounts.

## Rebuild and retrain (optional)

Only needed to change the data or the model. Install the training dependencies first:
```bash
pip install -r requirements-train.txt
```

1. Build ingredient graph (NPMI + chemical)
```bash
python scripts/build_graph.py
```

2. Train NLGCL embeddings (CPU, about 20 min)

**Whenever `data/processed/ingredients_vocab.csv` changes, retrain and commit `artifacts/ingredient_embeddings.npy` and `artifacts/train_meta.json`**: an out-of-date model stops the API at startup with an explicit message, and `tests/test_artifacts.py` fails.
```bash
python scripts/train_nlgcl.py                 # settings from configs/model.yaml
python scripts/train_nlgcl.py --device cuda   # GPU
python scripts/train_nlgcl.py --no-val        # skip per-epoch validation (faster, no early stopping)
```
Every epoch the model is validated on the **val split only**: BPR loss on ingredient pairs of val recipes the graph never saw, and NDCG@10 of the embedding alone and of the hybrid ranker. The embeddings of the best epoch on `val_emb_ndcg@10` are kept, and training stops after `patience` epochs without improvement. Outputs: `artifacts/ingredient_embeddings.npy`, `artifacts/nlgcl_state.pt`, `artifacts/train_meta.json` (full history).

Plot the training / validation curves:
```bash
python scripts/plot_training.py               # -> artifacts/training_curves.png
```

3. Recommend from the command line
```bash
python scripts/recommend.py salmon lemon --top-k 10
python scripts/recommend.py chicken --allergies gluten dairy --diet vegetarian
python scripts/recommend.py tomato basil --user-id 12345
```

## Web interface

`python scripts/run_app.py` starts both parts. To run them separately (two terminals, from the repo root):
```bash
uvicorn api.main:app                 # API on http://localhost:8000
streamlit run app/streamlit_app.py   # UI on http://localhost:8501
```
The UI (in French, dark/light toggle) only talks to the API (`FOODPAIRING_API_URL`, default
`http://127.0.0.1:8000`). Pages: **Trouver des accords** (query + recommendations + feedback, with
the user's diet/allergies applied), **Mon frigo** (pairings using only what the user has
at home, via the `candidates` field of `/recommend`), **Mon historique**, **Mon profil**
(diet, allergies, ingredients never to suggest, what the model learned from the user's feedback,
Food.com profile link, e-mail, password), **Administration** (admins only: overview, user
management, logs, API checks), **À propos**.

**Noter des recettes** (⭐): on the first login, a new account is sent once to 10 starter
recipes to rate (can be skipped; `onboarding_done` on the account). The page also offers one
recipe to rate at each visit, a search, and the list of the user's ratings (editable).

Feedback buttons map to the API feedback loop: 👍 `like`, 👎 `dislike`, 🚫 "Pas avec ça"
`not_for_me`, ⚠️ `allergic` (with confirmation), 🍳 `cooked`. The app does no filtering of its
own: it sends `user_id` = the account (one random `guest-xxxx` id per guest session), the linked
Food.com profile as `history_user_id` and the profile's "never suggest" list as `exclude`. Feedback
can be changed from **Mon profil** (e.g. ✅ `not_allergic` after a mistake).

Accounts are stored by the API, in a `users` table of the same SQLite database as the
interactions (`data/app.db`, or `FOODPAIRING_DB`), so they survive restarts. Passwords are salted
and hashed (PBKDF2-SHA256), and every rule lives in `api/users.py` (validation, unique e-mail,
"at least one active admin", admin-only actions). Deleting an account also deletes its
interactions, so a new account reusing the username starts from scratch. Routes: `POST /users`,
`POST /login`, `GET /users`, `GET|PATCH /users/{username}`, `POST /users/{username}/password`,
and `POST /admin/users/{username}/{role|active|reset_password}`, `DELETE /admin/users/{username}`
(both with the `actor` doing it). The API has **no authentication of its own yet**: it trusts its
caller (the Streamlit server next to it), so do not expose it publicly.

Demo accounts are created on the first start (empty table; `FOODPAIRING_SEED_DEMO=0` to skip):
`admin` / `admin1234` (admin), `demo` / `demo1234`, `camille` / `camille1234`, or "Continuer sans
compte". When the API falls back to the mock `/recommend` (no trained embeddings), a "Résultats
d'exemple" badge is shown.

Ingredient photos come from [TheMealDB](https://www.themealdb.com) (free for educational use).
`python scripts/fetch_ingredient_images.py` matches the vocab to their ingredient list (exact,
then without qualifiers like "fresh"/"ground", then the head noun) and stores 96 px thumbnails in
`app/images/ingredients/` with the mapping in `app/ingredient_images.csv` (~93 % of ingredient
uses covered). Ingredients without a photo show a family emoji instead.

Recipe photos: each recipe shows **its own photo from Food.com** when it has one (54 % of the
recipes, 99.6 % of the 3,000 most rated). `python scripts/link_foodcom_photos.py` reads the Kaggle
dataset [Food.com - Recipes and Reviews](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews)
(`recipes.parquet` in `data/raw/`, same recipe ids as ours) and writes `app/recipe_photos.parquet`
(recipe id → CDN path, 3 MB). The photos belong to the Food.com users who posted them, so they
are **not** stored in the repo: the browser loads a small version from Food.com's CDN. Underneath,
and shown when there is no photo or no internet, the app keeps a local picture:

Local pictures: `python scripts/fetch_recipe_images.py` matches recipe titles to TheMealDB
dish photos, strictly (same words, or every word of a 2+ word dish in the title *and* the
title's head noun in the dish, so "Pumpkin Pie Dip" gets no pumpkin pie photo). About 7.2k
recipes get a 160 px photo of a similar dish (`app/images/recipes/`, mapping in
`app/recipe_images.csv`). Every other recipe shows a 2x2 collage of its ingredient photos,
main ingredients first and pantry staples (salt, flour, oil, ...) last.

