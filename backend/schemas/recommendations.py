"""
EcoML — Model Recommendation and Decision Frontiers Schemas.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class WeightProfileSchema(BaseModel):
    """Multi-criteria scoring weight distribution."""
    name: str = Field("Balanced", max_length=50, description="Profile label.")
    w_performance: float = Field(0.40, ge=0.0, le=1.0, description="Predictive utility weight.")
    w_train_time: float = Field(0.15, ge=0.0, le=1.0, description="Training duration efficiency weight.")
    w_inference_latency: float = Field(0.15, ge=0.0, le=1.0, description="Inference latency efficiency weight.")
    w_ram: float = Field(0.15, ge=0.0, le=1.0, description="Memory footprint efficiency weight.")
    w_carbon: float = Field(0.15, ge=0.0, le=1.0, description="Carbon/energy emissions efficiency weight.")

    @model_validator(mode="after")
    def validate_positive_sum(self) -> "WeightProfileSchema":
        total = (
            self.w_performance
            + self.w_train_time
            + self.w_inference_latency
            + self.w_ram
            + self.w_carbon
        )
        if total <= 0:
            raise ValueError("Total sum of criteria weights must be strictly greater than zero.")
        return self


class RecommendationRequest(BaseModel):
    """Recommendation query request with optional custom weight profile."""
    weight_profile: Optional[WeightProfileSchema] = None


class ModelScoreDetailSchema(BaseModel):
    """Detailed multi-criteria scoring breakdown for an individual algorithm."""
    model_id: str
    model_name: str
    status: str
    rank: int
    composite_score: float
    is_pareto_optimal: bool
    raw_metrics: Dict[str, Any]
    normalized_scores: Dict[str, float]
    effective_weights: Dict[str, float]
    strengths: List[str]
    tradeoff_notes: List[str]


class RecommendationResponse(BaseModel):
    """Comprehensive recommendation synthesis."""
    benchmark_id: str
    dataset_name: str
    task_type: str
    profile_name: str
    ranked_models: List[ModelScoreDetailSchema]
    top_recommended_model: Optional[ModelScoreDetailSchema]
    pareto_frontier_models: List[str]
    effective_weights: Dict[str, float]
    redistribution_note: Optional[str] = None
    tradeoff_synthesis: str
    scoring_methodology: str


class WeightProfilePresetItem(BaseModel):
    """Preset weight profile details."""
    profile_id: str
    name: str
    description: str
    weights: WeightProfileSchema


class WeightProfilePresetsResponse(BaseModel):
    """List of available weight profile presets."""
    presets: List[WeightProfilePresetItem]

