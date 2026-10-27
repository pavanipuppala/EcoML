"""
EcoML — System Health & Metadata Routes.
"""

from fastapi import APIRouter
from config.settings import (
    PROJECT_NAME,
    PROJECT_VERSION,
    PROJECT_STATUS,
    PROJECT_TAGLINE,
    PLANNED_FEATURES,
    FUTURE_ROADMAP_FEATURES,
    ROADMAP_STEPS,
)
from backend.schemas.common import SystemHealthResponse, MetadataFeaturesResponse

router = APIRouter(tags=["System & Metadata"])


@router.get("/api/health", response_model=SystemHealthResponse)
def get_health() -> SystemHealthResponse:
    """Return health status and project metadata for EcoML."""
    return SystemHealthResponse(
        status="healthy",
        app=PROJECT_NAME,
        version=PROJECT_VERSION,
        operational_status=PROJECT_STATUS,
        tagline=PROJECT_TAGLINE,
    )


@router.get("/api/meta/features", response_model=MetadataFeaturesResponse)
def get_meta_features() -> MetadataFeaturesResponse:
    """Return implemented features, future roadmap capabilities, and milestone roadmap steps."""
    return MetadataFeaturesResponse(
        implemented_features=PLANNED_FEATURES,
        future_features=FUTURE_ROADMAP_FEATURES,
        roadmap_steps=ROADMAP_STEPS,
    )

