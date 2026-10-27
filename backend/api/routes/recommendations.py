"""
EcoML — Model Recommendation and Decision Frontiers Routes.
"""

from fastapi import APIRouter, HTTPException, status
from backend.schemas.recommendations import (
    WeightProfilePresetsResponse,
    RecommendationRequest,
    RecommendationResponse,
)
from backend.services.recommendation_service import recommendation_service

router = APIRouter(prefix="/api/recommendations", tags=["Green Intelligence & Recommendations"])


@router.get("/profiles", response_model=WeightProfilePresetsResponse)
def get_weight_profile_presets() -> WeightProfilePresetsResponse:
    """Retrieve predefined operational weight profiles (Balanced, Green-First, Accuracy-First, Edge)."""
    presets = recommendation_service.get_preset_profiles()
    return WeightProfilePresetsResponse(presets=presets)


@router.post("/{benchmark_id}", response_model=RecommendationResponse)
def calculate_recommendations(benchmark_id: str, req: RecommendationRequest = RecommendationRequest()) -> RecommendationResponse:
    """Execute multi-criteria scoring (WSM) and Pareto efficiency analysis for a benchmark."""
    try:
        custom_weights = req.weight_profile.model_dump() if req.weight_profile else None
        res = recommendation_service.generate_recommendations(
            benchmark_id=benchmark_id,
            custom_weights=custom_weights,
        )
        return RecommendationResponse(**res)
    except ValueError as exc:
        err_msg = str(exc)
        if "not found" in err_msg.lower():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=err_msg)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err_msg)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recommendation evaluation failed: {str(exc)}",
        )

