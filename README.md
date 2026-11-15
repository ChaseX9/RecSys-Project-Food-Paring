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


## Evaluation

The evaluation harness (`src/eval/`) scores any recommender on held-out recipes and reports **NDCG@10**, **Recall@20**, HitRate@k and MRR.

**Protocol: masked-ingredient completion.** For each held-out user → recipe interaction, about 20% of the recipe's ingredients are hidden (at least one). The remaining ingredients are the query, the hidden ones are the ground truth, and the user id is passed for personalization. Query ingredients are removed from every prediction before scoring.

**No leakage.** By default (`protocol: unseen_recipes`) only recipes with **no train interaction** are evaluated, so the graph, NPMI, popularity and user profiles (all built on train recipes) have never seen them. A stricter recipe-level holdout (`protocol: recipe_holdout`) is also available; it requires the graph and model to be rebuilt without the held-out recipes (`src.eval.protocol.holdout_recipe_ids`, same `holdout_frac` and `seed`).

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


