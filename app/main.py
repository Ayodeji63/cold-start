from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from app.recommender import RecommendationService


app = FastAPI(
    title="Kalm4Rec Persona Recommendation API",
    description="Containerized Task B API: user persona in, personalized recommendations out.",
    version="1.0.0",
)
service = RecommendationService()


class RecommendRequest(BaseModel):
    persona: str = Field(
        ...,
        min_length=5,
        description="Natural-language user persona or current recommendation context.",
        examples=[
            "A Nigerian student in Philadelphia who likes spicy jollof, halal meat, generous portions, and affordable places for group dinners."
        ],
    )
    domain: str = Field(
        "restaurants",
        description="Recommendation domain: restaurants, amazon_grocery, or amazon_grocery_dense.",
    )
    top_k: int = Field(10, ge=1, le=50, description="Number of recommendations to return.")
    city: Optional[str] = Field(
        None,
        description="Optional city filter for restaurant recommendations, e.g. Philadelphia, Tampa, Nashville.",
    )


class RecommendationResponse(BaseModel):
    domain: str
    top_k: int
    count: int
    recommendations: list[dict]


@app.get("/health")
def health():
    return {"status": "ok", "datasets": service.datasets()}


@app.get("/datasets")
def datasets():
    return service.datasets()


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(payload: RecommendRequest):
    try:
        recommendations = service.recommend(
            persona=payload.persona,
            domain=payload.domain,
            top_k=payload.top_k,
            city_filter=payload.city,
        )
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "domain": payload.domain,
        "top_k": payload.top_k,
        "count": len(recommendations),
        "recommendations": recommendations,
    }
