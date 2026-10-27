"""
EcoML — FastAPI Application Entry Point.

Centralized REST API backend for EcoML — Carbon and Energy-Aware Machine Learning Model Selector.
Interfaces directly with core scikit-learn preprocessing, benchmarking, and green recommendation modules.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import (
    PROJECT_NAME,
    PROJECT_VERSION,
    PROJECT_STATUS,
    PROJECT_TAGLINE,
    PROJECT_DESCRIPTION,
)
from backend.api.routes import (
    health_router,
    datasets_router,
    benchmarks_router,
    recommendations_router,
    quantum_router,
)

app = FastAPI(
    title=f"{PROJECT_NAME} API",
    description=f"{PROJECT_TAGLINE}\n\n{PROJECT_DESCRIPTION}",
    version=PROJECT_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware enabling Vite frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routers
app.include_router(health_router)
app.include_router(datasets_router)
app.include_router(benchmarks_router)
app.include_router(recommendations_router)
app.include_router(quantum_router)
