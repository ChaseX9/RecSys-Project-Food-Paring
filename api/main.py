import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

from . import db


class InteractionIn(BaseModel):
    user_id: str = Field(min_length=1, max_length=100)
    query_ingredients: list[str] = Field(min_length=1)
    recommended_ingredient: str | None = None
    feedback: Literal["click", "like", "dislike", "not_for_me", "cooked"]
    note: str | None = Field(default=None, max_length=1000)


class InteractionOut(BaseModel):
    id: int
    status: str = "logged"


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

    return app


app = create_app()
