"""
EcoML — Backend Services Index.
"""

from backend.services.dataset_service import dataset_service, DatasetService
from backend.services.benchmark_service import benchmark_service, BenchmarkService
from backend.services.recommendation_service import recommendation_service, RecommendationService

__all__ = [
    "dataset_service",
    "DatasetService",
    "benchmark_service",
    "BenchmarkService",
    "recommendation_service",
    "RecommendationService",
]
