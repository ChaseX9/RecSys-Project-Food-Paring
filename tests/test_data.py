import numpy as np
import pandas as pd

from src.data.build import leave_last_out_split
from src.data.ingredients import normalize_ingredient, normalize_recipe, parse_r_vector
from src.data.profiles import build_recipe_ingredient_matrix, build_user_profiles


def test_parse_r_vector():
    assert parse_r_vector('c("sugar", "lemons, rind of")') == ["sugar", "lemons, rind of"]
    assert parse_r_vector('"salt"') == ["salt"]
    assert parse_r_vector(float("nan")) == []


def test_normalize():
    assert normalize_ingredient("Fresh Tomatoes") == "tomato"
    assert normalize_ingredient("lemons, rind of") == "lemon"
    assert normalize_ingredient("molasses") == "molasses"
    assert normalize_ingredient("bay leaves") == "bay leaf"
    assert normalize_ingredient("scallions", {"scallion": "green onion"}) == "green onion"
    assert normalize_ingredient("2 (14 oz) cans") != ""  # must not crash
    assert normalize_recipe('c("onion", "onions", "salt")') == ["onion", "salt"]  # deduplication


def _inter():
    rows = []
    for u in (1, 2):
        for k in range(5):
            rows.append(dict(review_id=u * 100 + k, user_id=u, recipe_id=k,
                             rating=5, timestamp=pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=k)))
    return pd.DataFrame(rows)


def test_split_leave_last_out_no_leak():
    df = leave_last_out_split(_inter())
    for _, g in df.groupby("user_id"):
        g = g.sort_values("timestamp")
        assert list(g["split"]) == ["train"] * 3 + ["val", "test"]
        # test is strictly after train
        assert g[g.split == "test"].timestamp.iloc[0] > g[g.split == "train"].timestamp.max()


def test_profiles_use_train_only_and_l1():
    df = leave_last_out_split(_inter())
    recipes = pd.DataFrame({"recipe_id": range(5),
                            "ingredient_ids": [np.array([0, 1]), np.array([1]), np.array([2]),
                                               np.array([0]), np.array([2])]})
    M = build_recipe_ingredient_matrix(recipes, 3)
    idx = {r: i for i, r in enumerate(recipes.recipe_id)}
    ids, P = build_user_profiles(df[df.split == "train"], idx, M, 4)
    # train = recipes 0,1,2 -> ingredients {0:1, 1:2, 2:1}; recipes 3,4 (val/test) are ignored
    assert np.allclose(P[0].toarray().ravel(), [0.25, 0.5, 0.25])
    assert np.allclose(np.asarray(P.sum(axis=1)).ravel(), 1.0)


def test_flavor_features():
    import scipy.sparse as sp
    from src.data.flavor import build_compound_matrix, jaccard_similarity, map_vocab_to_flavorgraph

    vocab = pd.DataFrame({"ingredient_id": [0, 1, 2], "name": ["tomato", "basil", "sugar"],
                          "n_recipes": [10, 8, 5]})
    nodes = pd.DataFrame({"node_id": [10, 11, 12, 100, 101, 102],
                          "name": ["tomato", "fresh_basil", "milk", "c1", "c2", "c3"],
                          "node_type": ["ingredient"] * 3 + ["compound"] * 3})
    edges = pd.DataFrame({"id_1": [10, 10, 11, 11], "id_2": [100, 101, 101, 102],
                          "edge_type": ["ingr-fcomp"] * 4})
    mapping = map_vocab_to_flavorgraph(vocab, nodes)
    assert set(mapping["name"]) == {"tomato", "basil"}  # "fresh_basil" -> "basil", sugar absent
    mat, compounds = build_compound_matrix(mapping, nodes, edges, 3)
    assert mat.shape == (3, 3) and mat[2].nnz == 0
    sim = jaccard_similarity(mat).toarray()
    assert np.isclose(sim[0, 1], 1 / 3) and sim[0, 0] == 0


def test_major_ingredients_are_not_split():
    from src.data.ingredients import load_aliases
    aliases = load_aliases("configs/ingredient_aliases.yaml")
    cases = {
        "extra virgin olive oil": "olive oil",
        "extra-virgin olive oil": "olive oil",
        "olive oil": "olive oil",
        "chickpeas": "chickpea",
        "garbanzo beans": "chickpea",
        "garlic cloves": "garlic",
        "garlic": "garlic",
        "purple onion": "red onion",
        "fresh cilantro": "cilantro",
    }
    for raw, expected in cases.items():
        assert normalize_ingredient(raw, aliases) == expected, raw


def test_train_recipe_mask_excludes_val_test_only_recipes():
    from src.data.build import mark_train_recipes

    recipes = pd.DataFrame({"recipe_id": [1, 2, 3], "name": list("abc")})
    inter = pd.DataFrame({"recipe_id": [1, 2, 3, 3], "split": ["train", "val", "test", "train"]})
    marked = mark_train_recipes(recipes, inter)
    # recipe 2 only appears in validation -> must not be used for graph statistics
    assert marked["is_train_recipe"].tolist() == [True, False, True]
    assert "is_train_recipe" not in recipes.columns  # input untouched


def test_duplicate_ingredients_are_merged():
    from src.data.ingredients import load_aliases
    aliases = load_aliases("configs/ingredient_aliases.yaml")
    merged = {
        "Heinz Ketchup": "ketchup", "tomato ketchup": "ketchup", "catsup": "ketchup",
        "no-salt-added ketchup": "ketchup", "Hellmann's Mayonnaise": "mayonnaise",
        "Lea & Perrins Worcestershire Sauce": "worcestershire sauce",
        "Fat-Free Cream Cheese": "cream cheese", "low sodium soy sauce": "soy sauce",
        "hard-boiled eggs": "egg", "jumbo eggs": "egg", "egg tomatoes": "plum tomato",
        "Egg Beaters Egg Substitute": "egg substitute", "ground cinnamon": "cinnamon",
        "flat leaf parsley": "flat-leaf parsley", "self raising flour": "self-rising flour",
        "Cool Whip Lite": "cool whip", "fat free cool whip": "cool whip",
        "low fat mozzarella": "mozzarella cheese",
    }
    for raw, expected in merged.items():
        assert normalize_ingredient(raw, aliases) == expected, raw


def test_distinct_ingredients_are_not_merged():
    from src.data.ingredients import load_aliases
    aliases = load_aliases("configs/ingredient_aliases.yaml")
    for raw in ["eggplant", "egg substitute", "ground ginger", "ground beef", "dark brown sugar",
                "vegan butter", "vegetarian worcestershire sauce", "tabasco sauce"]:
        assert normalize_ingredient(raw, aliases) == raw, raw
