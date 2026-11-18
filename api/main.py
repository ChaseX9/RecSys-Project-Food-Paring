import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

from . import db
from .recommender import ModelRecommender, mock_recommend, model_available


class InteractionIn(BaseModel):
    user_id: str = Field(min_length=1, max_length=100)
    query_ingredients: list[str] = Field(min_length=1)
    recommended_ingredient: str | None = None
    feedback: Literal["click", "like", "dislike", "not_for_me", "cooked", "allergic"]
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


def create_app(db_path: str | None = None, recommender="auto") -> FastAPI:
    """``recommender``: "auto" = trained model if artifacts/ingredient_embeddings.npy exists
    (and FOODPAIRING_MOCK is not set), else the mock; None = mock; or any object with a
    ``recommend(...)`` method (tests)."""
    path = db_path or os.environ.get("FOODPAIRING_DB", str(db.DEFAULT_DB_PATH))
    state = {"model": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db.init_db(path)
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
                                  disliked=fb["disliked"])
        else:
            recs = model.recommend(req.ingredients, top_k=req.top_k, user_id=req.user_id,
                                   allergies=(req.allergies or []) + fb["allergic"], diet=req.diet,
                                   liked=fb["liked"], disliked=fb["disliked"] + fb["allergic"])
        return RecommendOut(query=req.ingredients, mock=model is None,
                            applied_feedback=AppliedFeedback(**fb), recommendations=recs)

    @app.get("/profile/{user_id}")
    def get_profile(user_id: str, n: int = Query(10, ge=1, le=50)):
        """The user's taste profile as the recommender sees it right now: ratings history
        (Food.com users) updated with every like / dislike sent to /log_interaction."""
        fb = db.feedback_for_user(path, user_id)
        model = state["model"]
        if model is None:
            return {"user_id": user_id, "mock": True, "feedback": fb, "likes": [], "dislikes": []}
        return {"user_id": user_id, "mock": False, "feedback": fb,
                **model.profile(user_id, fb["liked"], fb["disliked"] + fb["allergic"], n)}

    @app.get("/feedback/{user_id}", response_model=AppliedFeedback)
    def get_feedback(user_id: str):
        """What the feedback loop currently applies for this user (outside any query)."""
        return AppliedFeedback(**db.feedback_for_user(path, user_id))

    return app


app = create_app()
