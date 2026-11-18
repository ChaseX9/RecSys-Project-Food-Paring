import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

from . import db
from .recommender import mock_recommend


class InteractionIn(BaseModel):
    user_id: str = Field(min_length=1, max_length=100)
    query_ingredients: list[str] = Field(min_length=1)
    recommended_ingredient: str | None = None
    feedback: Literal["click", "like", "dislike", "not_for_me", "cooked"]
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


class Recommendation(BaseModel):
    rank: int
    name: str
    score: float
    emb_score: float
    chem_score: float
    pref_score: float
    explanation: str


class RecommendOut(BaseModel):
    query: list[str]
    mock: bool = True
    recommendations: list[Recommendation]


def create_app(db_path: str | None = None) -> FastAPI:
    path = db_path or os.environ.get("FOODPAIRING_DB", str(db.DEFAULT_DB_PATH))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db.init_db(path)
        yield

    app = FastAPI(title="Food Pairing API", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

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
        recs = mock_recommend(req.ingredients, top_k=req.top_k, user_id=req.user_id,
                              allergies=req.allergies, diet=req.diet,
                              candidates=req.candidates)
        return RecommendOut(query=req.ingredients, mock=True, recommendations=recs)

    return app


app = create_app()
