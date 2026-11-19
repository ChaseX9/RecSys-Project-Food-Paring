import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import db, recipes, users
from .recommender import PROCESSED_DIR, ModelRecommender, mock_recommend, model_available


# ------------------------------------------------------------------ users
class UserOut(BaseModel):
    username: str
    display_name: str
    email: str
    role: Literal["user", "admin"]
    active: bool
    diet: str | None = None
    allergies: list[str] = []
    disliked: list[str] = []
    fridge: list[str] = []
    top_k: int = 10
    food_com_id: str | None = None
    onboarding_done: bool = False
    created_at: str
    last_login: str | None = None


class UserCreate(BaseModel):
    username: str = Field(max_length=30)
    display_name: str = Field(max_length=100)
    email: str = Field(max_length=200)
    password: str = Field(max_length=200)
    role: Literal["user", "admin"] = "user"
    actor: str | None = None     # the admin creating the account (needed for role "admin")


class LoginIn(BaseModel):
    username: str = Field(max_length=100)
    password: str = Field(max_length=200)


class UserUpdate(BaseModel):
    """Fields a user can change on their own account (only the ones sent are updated)."""
    onboarding_done: bool | None = None   # recipe-rating onboarding done or skipped
    display_name: str | None = Field(default=None, max_length=100)
    email: str | None = Field(default=None, max_length=200)
    diet: Literal["vegetarian", "vegan", "pescatarian"] | None = None
    allergies: list[str] | None = Field(default=None, max_length=50)
    disliked: list[str] | None = Field(default=None, max_length=500)
    fridge: list[str] | None = Field(default=None, max_length=500)
    top_k: int | None = Field(default=None, ge=1, le=50)
    food_com_id: str | None = Field(default=None, max_length=20)


class PasswordChange(BaseModel):
    current: str = Field(max_length=200)
    new: str = Field(max_length=200)


class RecipeRatingIn(BaseModel):
    rating: int = Field(ge=1, le=5)


class AdminAction(BaseModel):
    actor: str                   # the admin doing it
    role: Literal["user", "admin"] | None = None
    active: bool | None = None


class InteractionIn(BaseModel):
    user_id: str = Field(min_length=1, max_length=100)
    query_ingredients: list[str] = Field(min_length=1)
    recommended_ingredient: str | None = None
    feedback: Literal["click", "like", "dislike", "not_for_me", "cooked", "allergic", "not_allergic"]
    note: str | None = Field(default=None, max_length=1000)


class InteractionOut(BaseModel):
    id: int
    status: str = "logged"


class RecommendIn(BaseModel):
    user_id: str = Field(min_length=1, max_length=100)
    ingredients: list[str] = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)
    allergies: list[str] | None = None
    diet: str | None = None
    # Optional whitelist: only these ingredients may be recommended (e.g. what is in the
    # user's fridge). None = the whole vocabulary.
    candidates: list[str] | None = Field(default=None, max_length=500)
    # Ingredients never to recommend for this request (e.g. the user's "never suggest" list).
    exclude: list[str] | None = Field(default=None, max_length=500)
    # Whose rating history personalizes the ranking (e.g. a linked Food.com profile).
    # Feedback is always read from `user_id`. None = `user_id`.
    history_user_id: str | None = Field(default=None, max_length=100)


class Recommendation(BaseModel):
    rank: int
    name: str
    score: float
    emb_score: float
    chem_score: float
    pref_score: float
    explanation: str
    feedback: str | None = None      # effect of the user's past like / dislike, if any


class AppliedFeedback(BaseModel):
    liked: list[str] = []
    disliked: list[str] = []
    allergic: list[str] = []      # permanent: never recommended again, nor any ingredient containing it


class RecommendOut(BaseModel):
    query: list[str]
    mock: bool = True
    applied_feedback: AppliedFeedback = AppliedFeedback()
    recommendations: list[Recommendation]


def create_app(db_path: str | None = None, recommender="auto", catalog="auto") -> FastAPI:
    """``recommender``: "auto" = trained model if artifacts/ingredient_embeddings.npy exists
    (and FOODPAIRING_MOCK is not set), else the mock; None = mock; or any object with a
    ``recommend(...)`` method (tests). ``catalog``: "auto" = Food.com recipes from
    data/processed if present, None = no recipe routes (503), or a RecipeCatalog (tests)."""
    path = db_path or os.environ.get("FOODPAIRING_DB", str(db.DEFAULT_DB_PATH))
    state = {"model": None, "catalog": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db.init_db(path)
        recipes.init_db(path)
        users.init_db(path, seed_demo=os.environ.get("FOODPAIRING_SEED_DEMO", "1") != "0")
        state["catalog"] = recipes.RecipeCatalog.load(PROCESSED_DIR) if catalog == "auto" else catalog
        if recommender == "auto":
            if model_available() and os.environ.get("FOODPAIRING_MOCK") != "1":
                state["model"] = ModelRecommender()
        elif recommender is not None:
            state["model"] = recommender
        yield

    app = FastAPI(title="Food Pairing API", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "model": "nlgcl" if state["model"] is not None else "mock"}

    @app.post("/log_interaction", response_model=InteractionOut, status_code=201)
    def log_interaction(item: InteractionIn):
        new_id = db.insert_interaction(path, item.user_id, item.query_ingredients,
                                       item.recommended_ingredient, item.feedback, item.note)
        return InteractionOut(id=new_id)

    @app.get("/interactions")
    def get_interactions(user_id: str | None = None, limit: int = Query(100, ge=1, le=1000)):
        return db.list_interactions(path, user_id, limit)

    @app.post("/recommend", response_model=RecommendOut)
    def recommend(req: RecommendIn):
        # feedback loop: every like / dislike logged by this user is applied immediately
        fb = db.feedback_for_user(path, req.user_id, req.ingredients)
        model = state["model"]
        if model is None:
            recs = mock_recommend(req.ingredients, top_k=req.top_k, user_id=req.user_id,
                                  allergies=(req.allergies or []) + fb["allergic"], diet=req.diet,
                                  disliked=fb["disliked"] + (req.exclude or []),
                                  candidates=req.candidates)
        else:
            recs = model.recommend(req.ingredients, top_k=req.top_k, user_id=req.user_id,
                                   allergies=(req.allergies or []) + fb["allergic"], diet=req.diet,
                                   liked=fb["liked"], disliked=fb["disliked"] + fb["allergic"],
                                   candidates=req.candidates, exclude=req.exclude,
                                   history_user_id=req.history_user_id,
                                   recipe_ratings=recipes.rating_pairs(path, req.user_id))
        return RecommendOut(query=req.ingredients, mock=model is None,
                            applied_feedback=AppliedFeedback(**fb), recommendations=recs)

    @app.get("/profile/{user_id}")
    def get_profile(user_id: str, n: int = Query(10, ge=1, le=50),
                    history_user_id: str | None = Query(None, max_length=100)):
        """The user's taste profile as the recommender sees it right now: ratings history
        (Food.com users, or ``history_user_id``) updated with every like / dislike sent to
        /log_interaction by ``user_id``."""
        fb = db.feedback_for_user(path, user_id)
        rated = recipes.rating_pairs(path, user_id)
        model = state["model"]
        if model is None:
            return {"user_id": user_id, "mock": True, "feedback": fb, "likes": [], "dislikes": [],
                    "n_recipe_ratings": len(rated)}
        return {"user_id": user_id, "mock": False, "feedback": fb, "n_recipe_ratings": len(rated),
                **model.profile(history_user_id or user_id, fb["liked"],
                                fb["disliked"] + fb["allergic"], n, recipe_ratings=rated)}

    @app.get("/feedback/{user_id}", response_model=AppliedFeedback)
    def get_feedback(user_id: str):
        """What the feedback loop currently applies for this user (outside any query)."""
        return AppliedFeedback(**db.feedback_for_user(path, user_id))

    # -------------------------------------------------------------- recipes
    # Recipes rated in the app feed the user's taste profile (see api/recipes.py).
    def _catalog() -> recipes.RecipeCatalog:
        if state["catalog"] is None:
            raise users.UserError(503, "Le catalogue de recettes n'est pas disponible "
                                       "(data/processed manquant).")
        return state["catalog"]

    def _banned(user_id: str | None, diet: str | None, allergies: list[str] | None) -> recipes.Ban:
        """What the user must never be offered: diet, allergies (profile + allergic feedback)."""
        model = state["model"]
        title = recipes.title_ban(diet)
        if model is None or not hasattr(model, "ban_for"):
            return recipes.Ban(title=title)    # mock: only the title rules
        allergic = db.feedback_for_user(path, user_id)["allergic"] if user_id else []
        ban = model.ban_for(diet, (allergies or []) + allergic)
        ban.title = title
        return ban

    def _rated(user_id: str | None) -> dict[int, int]:
        return dict(recipes.rating_pairs(path, user_id)) if user_id else {}

    @app.get("/recipes/search")
    def search_recipes(q: str = Query(min_length=2, max_length=100),
                       limit: int = Query(20, ge=1, le=50), user_id: str | None = None,
                       diet: str | None = None, allergies: list[str] | None = Query(None)):
        """Recipes whose name contains every word of ``q``, most rated first."""
        return _catalog().search(q, limit, _banned(user_id, diet, allergies), _rated(user_id))

    @app.get("/recipes/onboarding")
    def onboarding_recipes(user_id: str | None = None, diet: str | None = None,
                           allergies: list[str] | None = Query(None),
                           n: int = Query(recipes.ONBOARDING_SIZE, ge=1, le=20)):
        """Popular and varied recipes to rate first (not yet rated by ``user_id``)."""
        return _catalog().onboarding(_banned(user_id, diet, allergies), _rated(user_id), n)

    @app.get("/recipes/next")
    def next_recipe(user_id: str, diet: str | None = None,
                    allergies: list[str] | None = Query(None),
                    skip: list[int] | None = Query(None), seed: int | None = None):
        """One recipe to rate next, chosen to learn about ingredients not covered yet."""
        card = _catalog().next_to_rate(_banned(user_id, diet, allergies), _rated(user_id),
                                       skip or [], seed)
        if card is None:
            raise users.UserError(404, "Plus aucune recette à te proposer.")
        return card

    @app.get("/recipe_ratings/{user_id}")
    def list_recipe_ratings(user_id: str):
        """The user's recipe ratings, most recent first."""
        cat = state["catalog"]
        out = []
        for r in recipes.ratings_of(path, user_id):
            if cat is not None and r["recipe_id"] in cat:
                card = cat.card(cat.row[r["recipe_id"]])
            else:
                card = {"recipe_id": r["recipe_id"], "name": f"Recette {r['recipe_id']}",
                        "ingredients": []}
            out.append(card | {"user_rating": r["rating"], "rated_at": r["created_at"]})
        return out

    @app.put("/recipe_ratings/{user_id}/{recipe_id}", status_code=204)
    def rate_recipe(user_id: str, recipe_id: int, body: RecipeRatingIn):
        """Rate a recipe 1-5 (rating it again replaces the previous rating)."""
        if recipe_id not in _catalog():
            raise users.UserError(404, "Recette introuvable.")
        recipes.rate(path, user_id, recipe_id, body.rating)
        return Response(status_code=204)

    @app.delete("/recipe_ratings/{user_id}/{recipe_id}", status_code=204)
    def unrate_recipe(user_id: str, recipe_id: int):
        if not recipes.unrate(path, user_id, recipe_id):
            raise users.UserError(404, "Tu n'as pas noté cette recette.")
        return Response(status_code=204)

    # ------------------------------------------------------------- accounts
    # No authentication on the API itself yet: it trusts its caller (see api/users.py).
    @app.exception_handler(users.UserError)
    def user_error(request: Request, exc: users.UserError):
        return JSONResponse(status_code=exc.status, content={"detail": exc.message})

    @app.post("/users", response_model=UserOut, status_code=201)
    def create_user(body: UserCreate):
        return users.create_user(path, body.username, body.display_name, body.email,
                                 body.password, body.role, body.actor)

    @app.post("/login", response_model=UserOut)
    def login(body: LoginIn):
        return users.authenticate(path, body.username, body.password)

    @app.get("/users", response_model=list[UserOut])
    def list_users():
        return users.list_users(path)

    @app.get("/users/{username}", response_model=UserOut)
    def get_user(username: str):
        return users.get_user(path, username)

    @app.patch("/users/{username}", response_model=UserOut)
    def update_user(username: str, body: UserUpdate):
        return users.update_user(path, username, body.model_dump(exclude_unset=True))

    @app.post("/users/{username}/password", status_code=204)
    def change_password(username: str, body: PasswordChange):
        users.change_password(path, username, body.current, body.new)
        return Response(status_code=204)

    @app.post("/admin/users/{username}/role", response_model=UserOut)
    def admin_set_role(username: str, body: AdminAction):
        return users.set_role(path, body.actor, username, body.role or "user")

    @app.post("/admin/users/{username}/active", response_model=UserOut)
    def admin_set_active(username: str, body: AdminAction):
        return users.set_active(path, body.actor, username, bool(body.active))

    @app.post("/admin/users/{username}/reset_password")
    def admin_reset_password(username: str, body: AdminAction):
        return {"temporary_password": users.reset_password(path, body.actor, username)}

    @app.delete("/admin/users/{username}", status_code=204)
    def admin_delete_user(username: str, actor: str = Query(max_length=100)):
        users.delete_user(path, actor, username)
        return Response(status_code=204)

    return app


app = create_app()
