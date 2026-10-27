"""
EcoML — API Routers Index.
"""

from backend.api.routes.health import router as health_router
from backend.api.routes.datasets import router as datasets_router
from backend.api.routes.benchmarks import router as benchmarks_router
from backend.api.routes.recommendations import router as recommendations_router
from backend.api.routes.quantum import router as quantum_router

__all__ = [
    "health_router",
    "datasets_router",
    "benchmarks_router",
    "recommendations_router",
    "quantum_router",
]

