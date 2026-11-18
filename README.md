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

### Personalization (ratings 1-5)

User profiles use **every train rating**, not only the 5/5 ones (`src/data/rating_profiles.py`):

- **Centered, shrunk ratings:** `weight = rating - b_u`, where `b_u` is the user's mean rating pulled toward the global mean (4.69). A user who rates everything 5/5 (26% of users) still gets a positive profile; a 2/5 from a generous user becomes a dislike. Only ratings <= 3 can count as dislikes.
- **Like / dislike profiles:** ingredients of liked recipes go to `like`, ingredients of disliked recipes go to `dislike`; `pref = like - 0.5 * dislike`.
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

Every explicit feedback sent to `POST /log_interaction` (with `recommended_ingredient`) is applied on the **next** `POST /recommend` of the same user, no retraining needed (`api/db.py: feedback_for_user`, `HybridRanker.recommend(liked=, disliked=)`):

| feedback | effect |
|---|---|
| `dislike` | ingredient never recommended again to this user; ingredients close to it in the NLGCL space (cosine > `feedback_sim_threshold`) are ranked lower |
| `not_for_me` | same, but only for the same query ("not with this dish") |
| `like`, `cooked` | ingredient and its close neighbours are ranked higher |
| `click` | logged, not applied (implicit, too weak) |

The latest feedback on an ingredient wins (a user can change their mind). `/recommend` returns the feedback it applied (`applied_feedback`) and, per item, why the feedback moved it (`feedback`, also appended to `explanation`). `GET /feedback/{user_id}` shows the current state. Settings: `feedback_*` in `src/ranking/hybrid.py:RankingConfig`.

`/recommend` serves the trained model when `artifacts/ingredient_embeddings.npy` exists (`"mock": false`, check `GET /health`), and falls back to the mock otherwise or with `FOODPAIRING_MOCK=1`.

## Quick start

1. Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux or Mac
python3 -m venv .venv
source ./.venv/bin/activate
```

2. Install Dependences
```bash
pip install -r requirements.txt
```

3. Build ingredient graph (NPMI + chemical)
```bash
python scripts/build_graph.py
```

4. Train NLGCL embeddings (CPU is fine for 2.6k nodes, about 8 min)
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

5. Recommend
```bash
python scripts/recommend.py salmon lemon --top-k 10
python scripts/recommend.py chicken --allergies gluten dairy --diet vegetarian
python scripts/recommend.py tomato basil --user-id 12345
```


