from __future__ import annotations
from statistics import mean

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_pipeline import (
    load_product_cache,
    run_rag_pipeline,
)

app = FastAPI(title="Smart Shopping List – RAG API")

load_product_cache()
class Basket(BaseModel):
    products: list[str]
    days: int = 0


class RecommendRequest(BaseModel):
    history_baskets: list[Basket]


class RecommendedProduct(BaseModel):
    id: str
    name: str


class RecommendResponse(BaseModel):
    recommended: list[RecommendedProduct]




@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    if not req.history_baskets:
        raise HTTPException(status_code=422, detail="history_baskets nie może być puste")

    cache = load_product_cache()

    purchase_counts: dict[str, int] = {}
    for b in req.history_baskets:
        for pid in b.products:
            purchase_counts[pid] = purchase_counts.get(pid, 0) + 1

    baskets = [
        {
            "days": b.days,
            "products": [cache[pid]["name"] for pid in b.products if pid in cache],
        }
        for b in req.history_baskets
    ]

    try:
        output = run_rag_pipeline(
            history_baskets=baskets,
            previously_bought_ids=set(purchase_counts.keys()),
            purchase_counts=purchase_counts,

        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    recommended = [
        RecommendedProduct(id=p["id"], name=p["name"])
        for p in output["final_recommendations"]
    ]

    return RecommendResponse(
        recommended=recommended
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
