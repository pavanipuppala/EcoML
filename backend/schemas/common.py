"""
EcoML — Common API Schemas.
"""

from typing import Optional
from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Standard error response format."""
    detail: str = Field(..., description="Human-readable error explanation.")
    error_type: Optional[str] = Field(None, description="Categorical error identifier.")


class SystemHealthResponse(BaseModel):
    """System health check and metadata response."""
    status: str = Field("healthy", description="Operational health status.")
    app: str = Field(..., description="Project name.")
    version: str = Field(..., description="Application version.")
    operational_status: str = Field(..., description="Milestone operational state.")
    tagline: str = Field(..., description="Project descriptor.")


class RoadmapStepSchema(BaseModel):
    """Architecture roadmap step."""
    step: str
    title: str
    status: str
    summary: str


class FeatureItemSchema(BaseModel):
    """Feature status descriptor."""
    id: str
    title: str
    icon: str
    description: str
    status: str
    stage: str


class MetadataFeaturesResponse(BaseModel):
    """Project capabilities and roadmap stages."""
    implemented_features: list[FeatureItemSchema]
    future_features: list[FeatureItemSchema]
    roadmap_steps: list[RoadmapStepSchema]

