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

